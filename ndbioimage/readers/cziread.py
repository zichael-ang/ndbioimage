import re
import warnings
from abc import ABC
from io import BytesIO
from itertools import product
from pathlib import Path

import czifile
import imagecodecs
import numpy as np
from lxml import etree
from ome_types import model
from tifffile import repeat_nd

from .. import AbstractReader

try:
    # TODO: use zoom from imagecodecs implementation when available
    from scipy.ndimage.interpolation import zoom
except ImportError:
    try:
        from ndimage.interpolation import zoom
    except ImportError:
        zoom = None


def zstd_decode(data: bytes) -> bytes:
    """ decode zstd bytes, copied from BioFormats ZeissCZIReader """
    def read_var_int(stream: BytesIO) -> int:
        a = stream.read(1)[0]
        if a & 128:
            b = stream.read(1)[0]
            if b & 128:
                c = stream.read(1)[0]
                return (c << 14) | ((b & 127) << 7) | (a & 127)
            return (b << 7) | (a & 127)
        return a & 255

    try:
        with BytesIO(data) as stream:
            size_of_header = read_var_int(stream)
            high_low_unpacking = False
            while stream.tell() < size_of_header:
                chunk_id = read_var_int(stream)
                # only one chunk ID defined so far
                if chunk_id == 1:
                    high_low_unpacking = (stream.read(1)[0] & 1) == 1
                else:
                    raise ValueError(f'Invalid chunk id: {chunk_id}')
            pointer = stream.tell()
    except Exception:
        high_low_unpacking = False
        pointer = 0

    decoded = imagecodecs.zstd_decode(data[pointer:])
    if high_low_unpacking:
        second_half = len(decoded) // 2
        return bytes([decoded[second_half + i // 2] if i % 2 else decoded[i // 2] for i in range(len(decoded))])
    else:
        return decoded


def data(self, raw=False, resize=True, order=0):
    """Read image data from file and return as numpy array."""
    DECOMPRESS = czifile.czifile.DECOMPRESS
    DECOMPRESS[5] = imagecodecs.zstd_decode
    DECOMPRESS[6] = zstd_decode

    de = self.directory_entry
    fh = self._fh
    if raw:
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)
        return data
    if de.compression:
        # if de.compression not in DECOMPRESS:
        #     raise ValueError('compression unknown or not supported')
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)
        data = DECOMPRESS[de.compression](data)
        if de.compression == 2:
            # LZW
            data = np.fromstring(data, de.dtype)  # noqa
        elif de.compression in (5, 6):
            # ZSTD
            data = np.frombuffer(data, de.dtype)
    else:
        dtype = np.dtype(de.dtype)
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read_array(dtype, self.data_size // dtype.itemsize)

    data = data.reshape(de.stored_shape)
    if de.compression != 4 and de.stored_shape[-1] in (3, 4):
        if de.stored_shape[-1] == 3:
            # BGR -> RGB
            data = data[..., ::-1]
        else:
            # BGRA -> RGBA
            tmp = data[..., 0].copy()
            data[..., 0] = data[..., 2]
            data[..., 2] = tmp
    if de.stored_shape == de.shape or not resize:
        return data

    # sub / supersampling
    factors = [j / i for i, j in zip(de.stored_shape, de.shape)]
    factors = [(int(round(f)) if abs(f - round(f)) < 0.0001 else f)
               for f in factors]

    # use repeat if possible
    if order == 0 and all(isinstance(f, int) for f in factors):
        data = repeat_nd(data, factors).copy()
        data.shape = de.shape
        return data

    # remove leading dimensions with size 1 for speed
    shape = list(de.stored_shape)
    i = 0
    for s in shape:
        if s != 1:
            break
        i += 1
    shape = shape[i:]
    factors = factors[i:]
    data.shape = shape

    # resize RGB components separately for speed
    if zoom is None:
        raise ImportError("cannot import 'zoom' from scipy or ndimage")
    if shape[-1] in (3, 4) and factors[-1] == 1.0:
        factors = factors[:-1]
        old = data
        data = np.empty(de.shape, de.dtype[-2:])
        for i in range(shape[-1]):
            data[..., i] = zoom(old[..., i], zoom=factors, order=order)
    else:
        data = zoom(data, zoom=factors, order=order)

    data.shape = de.shape
    return data


# monkeypatch zstd into czifile
czifile.czifile.SubBlockSegment.data = data


class Reader(AbstractReader, ABC):
    priority = 0
    do_not_pickle = 'reader', 'filedict'

    @staticmethod
    def _can_open(path):
        return isinstance(path, Path) and path.suffix == '.czi'

    def open(self):
        self.reader = czifile.CziFile(self.path)
        filedict = {}
        for directory_entry in self.reader.filtered_subblock_directory:
            idx = self.get_index(directory_entry, self.reader.start)
            if 'S' not in self.reader.axes or self.series in range(*idx[self.reader.axes.index('S')]):
                for c in range(*idx[self.reader.axes.index('C')]):
                    for z in range(*idx[self.reader.axes.index('Z')]):
                        for t in range(*idx[self.reader.axes.index('T')]):
                            if (c, z, t) in filedict:
                                filedict[c, z, t].append(directory_entry)
                            else:
                                filedict[c, z, t] = [directory_entry]
        self.filedict = filedict  # noqa

    def close(self):
        self.reader.close()

    def get_ome(self):
        xml = self.reader.metadata()
        attachments = {i.attachment_entry.name: i.attachment_entry.data_segment()
                       for i in self.reader.attachments()}
        tree = etree.fromstring(xml)
        metadata = tree.find('Metadata')
        version = metadata.find('Version')
        if version is not None:
            version = version.text
        else:
            version = metadata.find('Experiment').attrib['Version']

        if version == '1.0':
            return self.ome_10(tree, attachments)
        elif version == '1.2':
            return self.ome_12(tree, attachments)

    def ome_12(self, tree, attachments):
        def text(item, default=""):
            return default if item is None else item.text

        def def_list(item):
            return [] if item is None else item

        ome = model.OME()

        metadata = tree.find('Metadata')

        information = metadata.find('Information')
        display_setting = metadata.find('DisplaySetting')
        ome.experimenters = [model.Experimenter(id='Experimenter:0',
                                                user_name=information.find('Document').find('UserName').text)]

        instrument = information.find('Instrument')
        for _ in instrument.find('Microscopes'):
            ome.instruments.append(model.Instrument(id='Instrument:0'))

        for detector in instrument.find('Detectors'):
            try:
                detector_type = model.Detector_Type(text(detector.find('Type')).upper() or "")
            except ValueError:
                detector_type = model.Detector_Type.OTHER

            ome.instruments[0].detectors.append(
                model.Detector(
                    id=detector.attrib['Id'].replace(' ', ''), model=text(detector.find('Manufacturer').find('Model')),
                    type=detector_type
                ))

        for objective in instrument.find('Objectives'):
            ome.instruments[0].objectives.append(
                model.Objective(
                    id=objective.attrib['Id'],
                    model=text(objective.find('Manufacturer').find('Model')),
                    immersion=text(objective.find('Immersion')),
                    lens_na=float(text(objective.find('LensNA'))),
                    nominal_magnification=float(text(objective.find('NominalMagnification')))))

        for tubelens in instrument.find('TubeLenses'):
            try:
                nominal_magnification = float(re.findall(r'\d+(?:[,.]\d*)?',
                                                         tubelens.attrib['Name'])[0].replace(',', '.'))
            except Exception:
                nominal_magnification = 1.0

            ome.instruments[0].objectives.append(
                model.Objective(
                    id=f"Objective:{tubelens.attrib['Id']}",
                    model=tubelens.attrib['Name'],
                    nominal_magnification=nominal_magnification))

        for light_source in def_list(instrument.find('LightSources')):
            if light_source.find('LightSourceType').find('Laser') is not None:
                ome.instruments[0].lasers.append(
                    model.Laser(
                        id=f"LightSource:{light_source.attrib['Id']}",
                        power=float(text(light_source.find('Power'))),
                        wavelength=float(light_source.attrib['Id'][-3:])))

        x_min = min([f.start[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_min = min([f.start[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        x_max = max([f.start[f.axes.index('X')] + f.shape[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_max = max([f.start[f.axes.index('Y')] + f.shape[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        size_x = x_max - x_min
        size_y = y_max - y_min
        size_c, size_z, size_t = (self.reader.shape[self.reader.axes.index(directory_entry)]
                                  for directory_entry in 'CZT')

        image = information.find('Image')
        pixel_type = text(image.find('PixelType'), 'Gray16')
        if pixel_type.startswith('Gray'):
            pixel_type = 'uint' + pixel_type[4:]
        objective_settings = image.find('ObjectiveSettings')
        try:  # TODO
            scenes = image.find('Dimensions').find('S').find('Scenes')
            center_position = [float(pos) for pos in text(scenes[0].find('CenterPosition')).split(',')]
        except AttributeError:
            center_position = [0, 0]
        um = model.UnitsLength.MICROMETER
        nm = model.UnitsLength.NANOMETER

        ome.images.append(
            model.Image(
                id='Image:0',
                name=f"{text(information.find('Document').find('Name'))} #1",
                pixels=model.Pixels(
                    id='Pixels:0', size_x=size_x, size_y=size_y,
                    size_c=size_c, size_z=size_z, size_t=size_t,
                    dimension_order='XYCZT', type=pixel_type,
                    significant_bits=int(text(image.find('ComponentBitCount'))),
                    big_endian=False, interleaved=False, metadata_only=True),
                experimenter_ref=model.ExperimenterRef(id='Experimenter:0'),
                instrument_ref=model.InstrumentRef(id='Instrument:0'),
                objective_settings=model.ObjectiveSettings(
                    id=objective_settings.find('ObjectiveRef').attrib['Id'],
                    medium=text(objective_settings.find('Medium')),
                    refractive_index=float(text(objective_settings.find('RefractiveIndex')))),
                stage_label=model.StageLabel(
                    name=f'Scene position #0',
                    x=center_position[0], x_unit=um,
                    y=center_position[1], y_unit=um)))

        for distance in metadata.find('Scaling').find('Items'):
            if distance.attrib['Id'] == 'X':
                ome.images[0].pixels.physical_size_x = float(text(distance.find('Value'))) * 1e6
            elif distance.attrib['Id'] == 'Y':
                ome.images[0].pixels.physical_size_y = float(text(distance.find('Value'))) * 1e6
            elif size_z > 1 and distance.attrib['Id'] == 'Z':
                ome.images[0].pixels.physical_size_z = float(text(distance.find('Value'))) * 1e6

        channels_im = {channel.attrib['Id']: channel for channel in image.find('Dimensions').find('Channels')}
        channels_ds = {channel.attrib['Id']: channel for channel in display_setting.find('Channels')}

        for idx, (key, channel) in enumerate(channels_im.items()):
            detector_settings = channel.find('DetectorSettings')
            laser_scan_info = channel.find('LaserScanInfo')
            detector = detector_settings.find('Detector')
            try:
                binning = model.Binning(text(detector_settings.find('Binning')))
            except ValueError:
                binning = model.Binning.OTHER

            light_sources_settings = channel.find('LightSourcesSettings')
            # no space in ome for multiple lightsources simultaneously
            if light_sources_settings is not None:
                light_source_settings = light_sources_settings[0]
                light_source_settings = model.LightSourceSettings(
                    id='LightSource:' + '_'.join([light_source_settings.find('LightSource').attrib['Id']
                                                  for light_source_settings in light_sources_settings]),
                    attenuation=float(text(light_source_settings.find('Attenuation'))),
                    wavelength=float(text(light_source_settings.find('Wavelength'))),
                    wavelength_unit=nm)
            else:
                light_source_settings = None

            ome.images[0].pixels.channels.append(
                model.Channel(
                    id=f'Channel:{idx}',
                    name=channel.attrib['Name'],
                    acquisition_mode=text(channel.find('AcquisitionMode')).replace('SingleMoleculeLocalisation',
                                                                                   'SingleMoleculeImaging'),
                    color=model.Color(text(channels_ds[channel.attrib['Id']].find('Color'), 'white')),
                    detector_settings=model.DetectorSettings(
                        id=detector.attrib['Id'].replace(' ', ""),
                        binning=binning),
                    emission_wavelength=i if (i := text(channel.find('EmissionWavelength'))) != '0' else '100',
                    excitation_wavelength=text(channel.find('ExcitationWavelength')),
                    # filter_set_ref=model.FilterSetRef(id=ome.instruments[0].filter_sets[filterset_idx].id),
                    illumination_type=text(channel.find('IlluminationType')),
                    light_source_settings=light_source_settings,
                    samples_per_pixel=int(text(laser_scan_info.find('Averaging'), '1'))))

        exposure_times = [float(text(channel.find('LaserScanInfo').find('FrameTime'), '100')) for channel in
                          channels_im.values()]
        delta_ts = attachments['TimeStamps'].data()
        dt = np.diff(delta_ts)
        if np.std(dt) / np.mean(dt) > 0.02:
            dt = np.median(dt[dt > 0])
            delta_ts = dt * np.arange(len(delta_ts))
            warnings.warn(f'delta_t is inconsistent, using median value: {dt}')

        for t, z, c in product(range(size_t), range(size_z), range(size_c)):
            ome.images[0].pixels.planes.append(
                model.Plane(the_c=c, the_z=z, the_t=t, delta_t=delta_ts[t], exposure_time=exposure_times[c]))

        idx = 0
        for layer in [] if (ml := metadata.find('Layers')) is None else ml:
            rectangle = layer.find('Elements').find('Rectangle')
            if rectangle is not None:
                geometry = rectangle.find('Geometry')
                roi = model.ROI(id=f'ROI:{idx}', description=text(layer.find('Usage')))
                roi.union.append(
                    model.Rectangle(
                        id='Shape:0:0',
                        height=float(text(geometry.find('Height'))),
                        width=float(text(geometry.find('Width'))),
                        x=float(text(geometry.find('Left'))),
                        y=float(text(geometry.find('Top')))))
                ome.rois.append(roi)
                ome.images[0].roi_refs.append(model.ROIRef(id=f'ROI:{idx}'))
                idx += 1
        return ome

    def ome_10(self, tree, attachments):
        def text(item, default=""):
            return default if item is None else item.text

        def def_list(item):
            return [] if item is None else item

        ome = model.OME()

        metadata = tree.find('Metadata')

        information = metadata.find('Information')
        display_setting = metadata.find('DisplaySetting')
        experiment = metadata.find('Experiment')
        acquisition_block = experiment.find('ExperimentBlocks').find('AcquisitionBlock')

        ome.experimenters = [model.Experimenter(id='Experimenter:0',
                                                user_name=information.find('User').find('DisplayName').text)]

        instrument = information.find('Instrument')
        ome.instruments.append(model.Instrument(id=instrument.attrib['Id']))

        for detector in instrument.find('Detectors'):
            try:
                detector_type = model.Detector_Type(text(detector.find('Type')).upper() or "")
            except ValueError:
                detector_type = model.Detector_Type.OTHER

            ome.instruments[0].detectors.append(
                model.Detector(
                    id=detector.attrib['Id'], model=text(detector.find('Manufacturer').find('Model')),
                    amplification_gain=float(text(detector.find('AmplificationGain'))),
                    gain=float(text(detector.find('Gain'))), zoom=float(text(detector.find('Zoom'))),
                    type=detector_type
                ))

        for objective in instrument.find('Objectives'):
            ome.instruments[0].objectives.append(
                model.Objective(
                    id=objective.attrib['Id'],
                    model=text(objective.find('Manufacturer').find('Model')),
                    immersion=text(objective.find('Immersion')),
                    lens_na=float(text(objective.find('LensNA'))),
                    nominal_magnification=float(text(objective.find('NominalMagnification')))))

        for light_source in def_list(instrument.find('LightSources')):
            if light_source.find('LightSourceType').find('Laser') is not None:
                ome.instruments[0].lasers.append(
                    model.Laser(
                        id=light_source.attrib['Id'],
                        model=text(light_source.find('Manufacturer').find('Model')),
                        power=float(text(light_source.find('Power'))),
                        wavelength=float(
                            text(light_source.find('LightSourceType').find('Laser').find('Wavelength')))))

        multi_track_setup = acquisition_block.find('MultiTrackSetup')
        for idx, tube_lens in enumerate({text(track_setup.find('TubeLensPosition'))
                                         for track_setup in multi_track_setup}):
            ome.instruments[0].objectives.append(
                model.Objective(id=f'Objective:Tubelens:{idx}', model=tube_lens,
                                nominal_magnification=float(
                                    re.findall(r'\d+[,.]\d*', tube_lens)[0].replace(',', '.'))
                                ))

        for idx, filter_ in enumerate({text(beam_splitter.find('Filter'))
                                       for track_setup in multi_track_setup
                                       for beam_splitter in track_setup.find('BeamSplitters')}):
            ome.instruments[0].filter_sets.append(
                model.FilterSet(id=f'FilterSet:{idx}', model=filter_)
            )

        for idx, collimator in enumerate({text(track_setup.find('FWFOVPosition'))
                                          for track_setup in multi_track_setup}):
            ome.instruments[0].filters.append(model.Filter(id=f'Filter:Collimator:{idx}', model=collimator))

        x_min = min([f.start[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_min = min([f.start[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        x_max = max([f.start[f.axes.index('X')] + f.shape[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_max = max([f.start[f.axes.index('Y')] + f.shape[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        size_x = x_max - x_min
        size_y = y_max - y_min
        size_c, size_z, size_t = (self.reader.shape[self.reader.axes.index(directory_entry)]
                                  for directory_entry in 'CZT')

        image = information.find('Image')
        pixel_type = text(image.find('PixelType'), 'Gray16')
        if pixel_type.startswith('Gray'):
            pixel_type = 'uint' + pixel_type[4:]
        objective_settings = image.find('ObjectiveSettings')
        scenes = image.find('Dimensions').find('S').find('Scenes')
        positions = scenes[0].find('Positions')[0]
        um = model.UnitsLength.MICROMETER
        nm = model.UnitsLength.NANOMETER

        ome.images.append(
            model.Image(
                id='Image:0',
                name=f"{text(information.find('Document').find('Name'))} #1",
                pixels=model.Pixels(
                    id='Pixels:0', size_x=size_x, size_y=size_y,
                    size_c=size_c, size_z=size_z, size_t=size_t,
                    dimension_order='XYCZT', type=pixel_type,
                    significant_bits=int(text(image.find('ComponentBitCount'))),
                    big_endian=False, interleaved=False, metadata_only=True),
                experimenter_ref=model.ExperimenterRef(id='Experimenter:0'),
                instrument_ref=model.InstrumentRef(id='Instrument:0'),
                objective_settings=model.ObjectiveSettings(
                    id=objective_settings.find('ObjectiveRef').attrib['Id'],
                    medium=text(objective_settings.find('Medium')),
                    refractive_index=float(text(objective_settings.find('RefractiveIndex')))),
                stage_label=model.StageLabel(
                    name=f'Scene position #0',
                    x=float(positions.attrib['X']), x_unit=um,
                    y=float(positions.attrib['Y']), y_unit=um,
                    z=float(positions.attrib['Z']), z_unit=um)))

        for distance in metadata.find('Scaling').find('Items'):
            if distance.attrib['Id'] == 'X':
                ome.images[0].pixels.physical_size_x = float(text(distance.find('Value'))) * 1e6
            elif distance.attrib['Id'] == 'Y':
                ome.images[0].pixels.physical_size_y = float(text(distance.find('Value'))) * 1e6
            elif size_z > 1 and distance.attrib['Id'] == 'Z':
                ome.images[0].pixels.physical_size_z = float(text(distance.find('Value'))) * 1e6

        channels_im = {channel.attrib['Id']: channel for channel in image.find('Dimensions').find('Channels')}
        channels_ds = {channel.attrib['Id']: channel for channel in display_setting.find('Channels')}
        channels_ts = {detector.attrib['Id']: track_setup
                       for track_setup in
                       experiment.find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup')
                       for detector in track_setup.find('Detectors')}

        for idx, (key, channel) in enumerate(channels_im.items()):
            detector_settings = channel.find('DetectorSettings')
            laser_scan_info = channel.find('LaserScanInfo')
            detector = detector_settings.find('Detector')
            try:
                binning = model.Binning(text(detector_settings.find('Binning')))
            except ValueError:
                binning = model.Binning.OTHER

            filterset = text(channels_ts[key].find('BeamSplitters')[0].find('Filter'))
            filterset_idx = [filterset.model for filterset in ome.instruments[0].filter_sets].index(filterset)

            light_sources_settings = channel.find('LightSourcesSettings')
            # no space in ome for multiple lightsources simultaneously
            if len(light_sources_settings) > idx:
                light_source_settings = light_sources_settings[idx]
            else:
                light_source_settings = light_sources_settings[0]
            light_source_settings = model.LightSourceSettings(
                id=light_source_settings.find('LightSource').attrib['Id'],
                attenuation=float(text(light_source_settings.find('Attenuation'))),
                wavelength=float(text(light_source_settings.find('Wavelength'))),
                wavelength_unit=nm)

            ome.images[0].pixels.channels.append(
                model.Channel(
                    id=f'Channel:{idx}',
                    name=channel.attrib['Name'],
                    acquisition_mode=text(channel.find('AcquisitionMode')),
                    color=model.Color(text(channels_ds[channel.attrib['Id']].find('Color'), 'white')),
                    detector_settings=model.DetectorSettings(id=detector.attrib['Id'], binning=binning),
                    # emission_wavelength=text(channel.find('EmissionWavelength')),  # TODO: fix
                    excitation_wavelength=light_source_settings.wavelength,
                    filter_set_ref=model.FilterSetRef(id=ome.instruments[0].filter_sets[filterset_idx].id),
                    illumination_type=text(channel.find('IlluminationType')),
                    light_source_settings=light_source_settings,
                    samples_per_pixel=int(text(laser_scan_info.find('Averaging')))))

        exposure_times = [float(text(channel.find('LaserScanInfo').find('FrameTime'))) for channel in
                          channels_im.values()]
        delta_ts = attachments['TimeStamps'].data()
        dt = np.diff(delta_ts)
        if np.std(dt) / np.mean(dt) > 0.02:
            dt = np.median(dt[dt > 0])
            delta_ts = dt * np.arange(len(delta_ts))
            warnings.warn(f'delta_t is inconsistent, using median value: {dt}')

        for t, z, c in product(range(size_t), range(size_z), range(size_c)):
            ome.images[0].pixels.planes.append(
                model.Plane(the_c=c, the_z=z, the_t=t, delta_t=delta_ts[t],
                            exposure_time=exposure_times[c],
                            position_x=float(positions.attrib['X']), position_x_unit=um,
                            position_y=float(positions.attrib['Y']), position_y_unit=um,
                            position_z=float(positions.attrib['Z']), position_z_unit=um))

        idx = 0
        for layer in [] if (ml := metadata.find('Layers')) is None else ml:
            rectangle = layer.find('Elements').find('Rectangle')
            if rectangle is not None:
                geometry = rectangle.find('Geometry')
                roi = model.ROI(id=f'ROI:{idx}', description=text(layer.find('Usage')))
                roi.union.append(
                    model.Rectangle(
                        id='Shape:0:0',
                        height=float(text(geometry.find('Height'))),
                        width=float(text(geometry.find('Width'))),
                        x=float(text(geometry.find('Left'))),
                        y=float(text(geometry.find('Top')))))
                ome.rois.append(roi)
                ome.images[0].roi_refs.append(model.ROIRef(id=f'ROI:{idx}'))
                idx += 1
        return ome

    def __frame__(self, c=0, z=0, t=0):
        f = np.zeros(self.base.shape['yx'], self.dtype)
        if (c, z, t) in self.filedict:
            directory_entries = self.filedict[c, z, t]
            x_min = min([f.start[f.axes.index('X')] for f in directory_entries])
            y_min = min([f.start[f.axes.index('Y')] for f in directory_entries])
            xy_min = {'X': x_min, 'Y': y_min}
            for directory_entry in directory_entries:
                subblock = directory_entry.data_segment()
                tile = subblock.data(resize=True, order=0)
                axes_min = [xy_min.get(ax, 0) for ax in directory_entry.axes]
                index = [slice(i - j - m, i - j + k)
                         for i, j, k, m in zip(directory_entry.start, self.reader.start, tile.shape, axes_min)]
                index = tuple(index[self.reader.axes.index(i)] for i in 'YX')
                f[index] = tile.squeeze()
        return f

    @staticmethod
    def get_index(directory_entry, start):
        return [(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, directory_entry.shape)]
