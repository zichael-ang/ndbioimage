import re
import warnings
from abc import ABC
from functools import cached_property
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import czifile
import imagecodecs
import numpy as np
from lxml import etree
from ome_types import OME, model
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


Element = TypeVar('Element')


def zstd_decode(data: bytes) -> bytes:  # noqa
    """ decode zstd bytes, copied from BioFormats ZeissCZIReader """
    def read_var_int(stream: BytesIO) -> int:  # noqa
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
    except Exception:  # noqa
        high_low_unpacking = False
        pointer = 0

    decoded = imagecodecs.zstd_decode(data[pointer:])
    if high_low_unpacking:
        second_half = len(decoded) // 2
        return bytes([decoded[second_half + i // 2] if i % 2 else decoded[i // 2] for i in range(len(decoded))])
    else:
        return decoded


def data(self, raw: bool = False, resize: bool = True, order: int = 0) -> np.ndarray:
    """Read image data from file and return as numpy array."""
    DECOMPRESS = czifile.czifile.DECOMPRESS  # noqa
    DECOMPRESS[5] = imagecodecs.zstd_decode
    DECOMPRESS[6] = zstd_decode

    de = self.directory_entry
    fh = self._fh
    if raw:
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)  # noqa
        return data
    if de.compression:
        # if de.compression not in DECOMPRESS:
        #     raise ValueError('compression unknown or not supported')
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read(self.data_size)  # noqa
        data = DECOMPRESS[de.compression](data)  # noqa
        if de.compression == 2:
            # LZW
            data = np.fromstring(data, de.dtype)  # noqa
        elif de.compression in (5, 6):
            # ZSTD
            data = np.frombuffer(data, de.dtype)  # noqa
    else:
        dtype = np.dtype(de.dtype)
        with fh.lock:
            fh.seek(self.data_offset)
            data = fh.read_array(dtype, self.data_size // dtype.itemsize)  # noqa

    data = data.reshape(de.stored_shape)  # noqa
    if de.compression != 4 and de.stored_shape[-1] in (3, 4):
        if de.stored_shape[-1] == 3:
            # BGR -> RGB
            data = data[..., ::-1]  # noqa
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
        data = repeat_nd(data, factors).copy()  # noqa
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
        data = np.empty(de.shape, de.dtype[-2:])  # noqa
        for i in range(shape[-1]):
            data[..., i] = zoom(old[..., i], zoom=factors, order=order)
    else:
        data = zoom(data, zoom=factors, order=order)  # noqa

    data.shape = de.shape
    return data


# monkeypatch zstd into czifile
czifile.czifile.SubBlockSegment.data = data


class Reader(AbstractReader, ABC):
    priority = 0
    do_not_pickle = 'reader', 'filedict'

    @staticmethod
    def _can_open(path: Path) -> bool:
        return isinstance(path, Path) and path.suffix == '.czi'

    def open(self) -> None:
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
        if len(filedict) == 0:
            raise FileNotFoundError(f'Series {self.series} not found in {self.path}.')
        self.filedict = filedict  # noqa

    def close(self) -> None:
        self.reader.close()

    def get_ome(self) -> OME:
        return OmeParse.get_ome(self.reader, self.filedict)

    def __frame__(self, c: int = 0, z: int = 0, t: int = 0) -> np.ndarray:
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
    def get_index(directory_entry: czifile.DirectoryEntryDV, start: tuple[int]) -> list[tuple[int, int]]:
        return [(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, directory_entry.shape)]


class OmeParse:
    size_x: int
    size_y: int
    size_c: int
    size_z: int
    size_t: int

    nm = model.UnitsLength.NANOMETER
    um = model.UnitsLength.MICROMETER

    @classmethod
    def get_ome(cls, reader: czifile.CziFile, filedict: dict[tuple[int, int, int], Any]) -> OME:
        new = cls(reader, filedict)
        new.parse()
        return new.ome

    def __init__(self, reader: czifile.CziFile, filedict: dict[tuple[int, int, int], Any]) -> None:
        self.reader = reader
        self.filedict = filedict
        xml = reader.metadata()
        self.attachments = {i.attachment_entry.name: i.attachment_entry.data_segment()
                            for i in reader.attachments()}
        self.tree = etree.fromstring(xml)
        self.metadata = self.tree.find('Metadata')
        version = self.metadata.find('Version')
        if version is not None:
            self.version = version.text
        else:
            self.version = self.metadata.find('Experiment').attrib['Version']

        self.ome = OME()
        self.information = self.metadata.find('Information')
        self.display_setting = self.metadata.find('DisplaySetting')
        self.experiment = self.metadata.find('Experiment')
        self.acquisition_block = self.experiment.find('ExperimentBlocks').find('AcquisitionBlock')
        self.instrument = self.information.find('Instrument')
        self.image = self.information.find('Image')

        if self.version == '1.0':
            self.experiment = self.metadata.find('Experiment')
            self.acquisition_block = self.experiment.find('ExperimentBlocks').find('AcquisitionBlock')
            self.multi_track_setup = self.acquisition_block.find('MultiTrackSetup')
        else:
            self.experiment = None
            self.acquisition_block = None
            self.multi_track_setup = None

    def parse(self) -> None:
        self.get_experimenters()
        self.get_instruments()
        self.get_detectors()
        self.get_objectives()
        self.get_tubelenses()
        self.get_light_sources()
        self.get_filters()
        self.get_pixels()
        self.get_channels()
        self.get_planes()
        self.get_annotations()

    @staticmethod
    def text(item: Optional[Element], default: str = "") -> str:
        return default if item is None else item.text

    @staticmethod
    def def_list(item: Any) -> list[Any]:
        return [] if item is None else item

    @staticmethod
    def try_default(fun: Callable[[Any, ...], Any] | type, default: Any = None, *args: Any, **kwargs: Any) -> Any:
        try:
            return fun(*args, **kwargs)
        except Exception:  # noqa
            return default

    def get_experimenters(self) -> None:
        if self.version == '1.0':
            self.ome.experimenters = [
                model.Experimenter(id='Experimenter:0',
                                   user_name=self.information.find('User').find('DisplayName').text)]
        elif self.version in ('1.1', '1.2'):
            self.ome.experimenters = [
                model.Experimenter(id='Experimenter:0',
                                   user_name=self.information.find('Document').find('UserName').text)]

    def get_instruments(self) -> None:
        if self.version == '1.0':
            self.ome.instruments.append(model.Instrument(id=self.instrument.attrib['Id']))
        elif self.version in ('1.1', '1.2'):
            for _ in self.instrument.find('Microscopes'):
                self.ome.instruments.append(model.Instrument(id='Instrument:0'))

    def get_detectors(self) -> None:
        if self.version == '1.0':
            for detector in self.instrument.find('Detectors'):
                try:
                    detector_type = model.Detector_Type(self.text(detector.find('Type')).upper() or "")
                except ValueError:
                    detector_type = model.Detector_Type.OTHER

                self.ome.instruments[0].detectors.append(
                    model.Detector(
                        id=detector.attrib['Id'], model=self.text(detector.find('Manufacturer').find('Model')),
                        amplification_gain=float(self.text(detector.find('AmplificationGain'))),
                        gain=float(self.text(detector.find('Gain'))), zoom=float(self.text(detector.find('Zoom'))),
                        type=detector_type
                    ))
        elif self.version in ('1.1', '1.2'):
            for detector in self.instrument.find('Detectors'):
                try:
                    detector_type = model.Detector_Type(self.text(detector.find('Type')).upper() or "")
                except ValueError:
                    detector_type = model.Detector_Type.OTHER

                self.ome.instruments[0].detectors.append(
                    model.Detector(
                        id=detector.attrib['Id'].replace(' ', ''),
                        model=self.text(detector.find('Manufacturer').find('Model')),
                        type=detector_type
                    ))

    def get_objectives(self) -> None:
        for objective in self.instrument.find('Objectives'):
            self.ome.instruments[0].objectives.append(
                model.Objective(
                    id=objective.attrib['Id'],
                    model=self.text(objective.find('Manufacturer').find('Model')),
                    immersion=self.text(objective.find('Immersion')),  # type: ignore
                    lens_na=float(self.text(objective.find('LensNA'))),
                    nominal_magnification=float(self.text(objective.find('NominalMagnification')))))

    def get_tubelenses(self) -> None:
        if self.version == '1.0':
            for idx, tube_lens in enumerate({self.text(track_setup.find('TubeLensPosition'))
                                             for track_setup in self.multi_track_setup}):
                self.ome.instruments[0].objectives.append(
                    model.Objective(id=f'Objective:Tubelens:{idx}', model=tube_lens,
                                    nominal_magnification=float(
                                        re.findall(r'\d+[,.]\d*', tube_lens)[0].replace(',', '.'))
                                    ))
        elif self.version in ('1.1', '1.2'):
            for tubelens in self.instrument.find('TubeLenses'):
                try:
                    nominal_magnification = float(re.findall(r'\d+(?:[,.]\d*)?',
                                                             tubelens.attrib['Name'])[0].replace(',', '.'))
                except Exception:  # noqa
                    nominal_magnification = 1.0

                self.ome.instruments[0].objectives.append(
                    model.Objective(
                        id=f"Objective:{tubelens.attrib['Id']}",
                        model=tubelens.attrib['Name'],
                        nominal_magnification=nominal_magnification))

    def get_light_sources(self) -> None:
        if self.version == '1.0':
            for light_source in self.def_list(self.instrument.find('LightSources')):
                if light_source.find('LightSourceType').find('Laser') is not None:
                    self.ome.instruments[0].lasers.append(
                        model.Laser(
                            id=light_source.attrib['Id'],
                            model=self.text(light_source.find('Manufacturer').find('Model')),
                            power=float(self.text(light_source.find('Power'))),
                            wavelength=float(
                                self.text(light_source.find('LightSourceType').find('Laser').find('Wavelength')))))
        elif self.version in ('1.1', '1.2'):
            for light_source in self.def_list(self.instrument.find('LightSources')):
                if light_source.find('LightSourceType').find('Laser') is not None:
                    self.ome.instruments[0].lasers.append(
                        model.Laser(
                            id=f"LightSource:{light_source.attrib['Id']}",
                            power=float(self.text(light_source.find('Power'))),
                            wavelength=float(light_source.attrib['Id'][-3:])))

    def get_filters(self) -> None:
        if self.version == '1.0':
            for idx, filter_ in enumerate({self.text(beam_splitter.find('Filter'))
                                           for track_setup in self.multi_track_setup
                                           for beam_splitter in track_setup.find('BeamSplitters')}):
                self.ome.instruments[0].filter_sets.append(
                    model.FilterSet(id=f'FilterSet:{idx}', model=filter_)
                )

    def get_pixels(self) -> None:
        x_min = min([f.start[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_min = min([f.start[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        x_max = max([f.start[f.axes.index('X')] + f.shape[f.axes.index('X')] for f in self.filedict[0, 0, 0]])
        y_max = max([f.start[f.axes.index('Y')] + f.shape[f.axes.index('Y')] for f in self.filedict[0, 0, 0]])
        self.size_x = x_max - x_min
        self.size_y = y_max - y_min
        self.size_c, self.size_z, self.size_t = (self.reader.shape[self.reader.axes.index(directory_entry)]
                                                 for directory_entry in 'CZT')
        image = self.information.find('Image')
        pixel_type = self.text(image.find('PixelType'), 'Gray16')
        if pixel_type.startswith('Gray'):
            pixel_type = 'uint' + pixel_type[4:]
        objective_settings = image.find('ObjectiveSettings')

        self.ome.images.append(
            model.Image(
                id='Image:0',
                name=f"{self.text(self.information.find('Document').find('Name'))} #1",
                pixels=model.Pixels(
                    id='Pixels:0', size_x=self.size_x, size_y=self.size_y,
                    size_c=self.size_c, size_z=self.size_z, size_t=self.size_t,
                    dimension_order='XYCZT', type=pixel_type,  # type: ignore
                    significant_bits=int(self.text(image.find('ComponentBitCount'))),
                    big_endian=False, interleaved=False, metadata_only=True),  # type: ignore
                experimenter_ref=model.ExperimenterRef(id='Experimenter:0'),
                instrument_ref=model.InstrumentRef(id='Instrument:0'),
                objective_settings=model.ObjectiveSettings(
                    id=objective_settings.find('ObjectiveRef').attrib['Id'],
                    medium=self.text(objective_settings.find('Medium')),  # type: ignore
                    refractive_index=float(self.text(objective_settings.find('RefractiveIndex')))),
                stage_label=model.StageLabel(
                    name=f'Scene position #0',
                    x=self.positions[0], x_unit=self.um,
                    y=self.positions[1], y_unit=self.um,
                    z=self.positions[2], z_unit=self.um)))

        for distance in self.metadata.find('Scaling').find('Items'):
            if distance.attrib['Id'] == 'X':
                self.ome.images[0].pixels.physical_size_x = float(self.text(distance.find('Value'))) * 1e6
            elif distance.attrib['Id'] == 'Y':
                self.ome.images[0].pixels.physical_size_y = float(self.text(distance.find('Value'))) * 1e6
            elif self.size_z > 1 and distance.attrib['Id'] == 'Z':
                self.ome.images[0].pixels.physical_size_z = float(self.text(distance.find('Value'))) * 1e6

    @cached_property
    def positions(self) -> tuple[float, float, Optional[float]]:
        if self.version == '1.0':
            scenes = self.image.find('Dimensions').find('S').find('Scenes')
            positions = scenes[0].find('Positions')[0]
            return float(positions.attrib['X']), float(positions.attrib['Y']), float(positions.attrib['Z'])
        elif self.version in ('1.1', '1.2'):
            try:  # TODO
                scenes = self.image.find('Dimensions').find('S').find('Scenes')
                center_position = [float(pos) for pos in self.text(scenes[0].find('CenterPosition')).split(',')]
            except AttributeError:
                center_position = [0, 0]
            return center_position[0], center_position[1], None

    @cached_property
    def channels_im(self) -> dict:
        return {channel.attrib['Id']: channel for channel in self.image.find('Dimensions').find('Channels')}

    @cached_property
    def channels_ds(self) -> dict:
        return {channel.attrib['Id']: channel for channel in self.display_setting.find('Channels')}

    @cached_property
    def channels_ts(self) -> dict:
        return {detector.attrib['Id']: track_setup
                for track_setup in
                self.experiment.find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup')
                for detector in track_setup.find('Detectors')}

    def get_channels(self) -> None:
        if self.version == '1.0':
            for idx, (key, channel) in enumerate(self.channels_im.items()):
                detector_settings = channel.find('DetectorSettings')
                laser_scan_info = channel.find('LaserScanInfo')
                detector = detector_settings.find('Detector')
                try:
                    binning = model.Binning(self.text(detector_settings.find('Binning')))
                except ValueError:
                    binning = model.Binning.OTHER

                filterset = self.text(self.channels_ts[key].find('BeamSplitters')[0].find('Filter'))
                filterset_idx = [filterset.model for filterset in self.ome.instruments[0].filter_sets].index(filterset)

                light_sources_settings = channel.find('LightSourcesSettings')
                # no space in ome for multiple lightsources simultaneously
                if len(light_sources_settings) > idx:
                    light_source_settings = light_sources_settings[idx]
                else:
                    light_source_settings = light_sources_settings[0]
                light_source_settings = model.LightSourceSettings(
                    id=light_source_settings.find('LightSource').attrib['Id'],
                    attenuation=float(self.text(light_source_settings.find('Attenuation'))),
                    wavelength=float(self.text(light_source_settings.find('Wavelength'))),
                    wavelength_unit=self.nm)

                self.ome.images[0].pixels.channels.append(
                    model.Channel(
                        id=f'Channel:{idx}',
                        name=channel.attrib['Name'],
                        acquisition_mode=self.text(channel.find('AcquisitionMode')),  # type: ignore
                        color=model.Color(self.text(self.channels_ds[channel.attrib['Id']].find('Color'), 'white')),
                        detector_settings=model.DetectorSettings(id=detector.attrib['Id'], binning=binning),
                        # emission_wavelength=text(channel.find('EmissionWavelength')),  # TODO: fix
                        excitation_wavelength=light_source_settings.wavelength,
                        filter_set_ref=model.FilterSetRef(id=self.ome.instruments[0].filter_sets[filterset_idx].id),
                        illumination_type=self.text(channel.find('IlluminationType')),  # type: ignore
                        light_source_settings=light_source_settings,
                        samples_per_pixel=int(self.text(laser_scan_info.find('Averaging')))))
        elif self.version in ('1.1', '1.2'):
            for idx, (key, channel) in enumerate(self.channels_im.items()):
                detector_settings = channel.find('DetectorSettings')
                laser_scan_info = channel.find('LaserScanInfo')
                detector = detector_settings.find('Detector')
                try:
                    color = model.Color(self.text(self.channels_ds[channel.attrib['Id']].find('Color'), 'white'))
                except Exception:  # noqa
                    color = None
                try:
                    if (i := self.text(channel.find('EmissionWavelength'))) != '0':
                        emission_wavelength = float(i)
                    else:
                        emission_wavelength = None
                except Exception:  # noqa
                    emission_wavelength = None
                if laser_scan_info is not None:
                    samples_per_pixel = int(self.text(laser_scan_info.find('Averaging'), '1'))
                else:
                    samples_per_pixel = 1
                try:
                    binning = model.Binning(self.text(detector_settings.find('Binning')))
                except ValueError:
                    binning = model.Binning.OTHER

                light_sources_settings = channel.find('LightSourcesSettings')
                # no space in ome for multiple lightsources simultaneously
                if light_sources_settings is not None:
                    light_source_settings = light_sources_settings[0]
                    light_source_settings = model.LightSourceSettings(
                        id='LightSource:' + '_'.join([light_source_settings.find('LightSource').attrib['Id']
                                                      for light_source_settings in light_sources_settings]),
                        attenuation=self.try_default(float, None, self.text(light_source_settings.find('Attenuation'))),
                        wavelength=self.try_default(float, None, self.text(light_source_settings.find('Wavelength'))),
                        wavelength_unit=self.nm)
                else:
                    light_source_settings = None

                self.ome.images[0].pixels.channels.append(
                    model.Channel(
                        id=f'Channel:{idx}',
                        name=channel.attrib['Name'],
                        acquisition_mode=self.text(channel.find('AcquisitionMode')).replace(  # type: ignore
                            'SingleMoleculeLocalisation', 'SingleMoleculeImaging'),
                        color=color,
                        detector_settings=model.DetectorSettings(
                            id=detector.attrib['Id'].replace(' ', ""),
                            binning=binning),
                        emission_wavelength=emission_wavelength,
                        excitation_wavelength=self.try_default(float, None,
                                                               self.text(channel.find('ExcitationWavelength'))),
                        # filter_set_ref=model.FilterSetRef(id=ome.instruments[0].filter_sets[filterset_idx].id),
                        illumination_type=self.text(channel.find('IlluminationType')),  # type: ignore
                        light_source_settings=light_source_settings,
                        samples_per_pixel=samples_per_pixel))

    def get_planes(self) -> None:
        try:
            exposure_times = [float(self.text(channel.find('LaserScanInfo').find('FrameTime')))
                              for channel in self.channels_im.values()]
        except Exception:  # noqa
            exposure_times = [None] * len(self.channels_im)
        delta_ts = self.attachments['TimeStamps'].data()
        dt = np.diff(delta_ts)
        if len(dt) and np.std(dt) / np.mean(dt) > 0.02:
            dt = np.median(dt[dt > 0])
            delta_ts = dt * np.arange(len(delta_ts))
            warnings.warn(f'delta_t is inconsistent, using median value: {dt}')

        for t, z, c in product(range(self.size_t), range(self.size_z), range(self.size_c)):
            self.ome.images[0].pixels.planes.append(
                model.Plane(the_c=c, the_z=z, the_t=t, delta_t=delta_ts[t],
                            exposure_time=exposure_times[c],
                            position_x=self.positions[0], position_x_unit=self.um,
                            position_y=self.positions[1], position_y_unit=self.um,
                            position_z=self.positions[2], position_z_unit=self.um))

    def get_annotations(self) -> None:
        idx = 0
        for layer in [] if (ml := self.metadata.find('Layers')) is None else ml:
            rectangle = layer.find('Elements').find('Rectangle')
            if rectangle is not None:
                geometry = rectangle.find('Geometry')
                roi = model.ROI(id=f'ROI:{idx}', description=self.text(layer.find('Usage')))
                roi.union.append(
                    model.Rectangle(
                        id='Shape:0:0',
                        height=float(self.text(geometry.find('Height'))),
                        width=float(self.text(geometry.find('Width'))),
                        x=float(self.text(geometry.find('Left'))),
                        y=float(self.text(geometry.find('Top')))))
                self.ome.rois.append(roi)
                self.ome.images[0].roi_refs.append(model.ROIRef(id=f'ROI:{idx}'))
                idx += 1
