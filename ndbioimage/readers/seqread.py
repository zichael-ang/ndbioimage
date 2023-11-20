import re
from abc import ABC
from datetime import datetime
from functools import cached_property
from itertools import product
from pathlib import Path

import tifffile
import yaml
from ome_types import model
from ome_types.units import _quantity_property  # noqa

from .. import AbstractReader


def lazy_property(function, field, *arg_fields):
    def lazy(self):
        if self.__dict__.get(field) is None:
            self.__dict__[field] = function(*[getattr(self, arg_field) for arg_field in arg_fields])
            try:
                self.model_fields_set.add(field)
            except Exception:  # noqa
                pass
        return self.__dict__[field]
    return property(lazy)


class Plane(model.Plane):
    """ Lazily retrieve delta_t from metadata """
    def __init__(self, t0, file, **kwargs):
        super().__init__(**kwargs)
        # setting fields here because they would be removed by ome_types/pydantic after class definition
        setattr(self.__class__, 'delta_t', lazy_property(self.get_delta_t, 'delta_t', 't0', 'file'))
        setattr(self.__class__, 'delta_t_quantity', _quantity_property('delta_t'))
        self.__dict__['t0'] = t0
        self.__dict__['file'] = file

    @staticmethod
    def get_delta_t(t0, file):
        with tifffile.TiffFile(file) as tif:
            info = yaml.safe_load(tif.pages[0].tags[50839].value['Info'])
        return float((datetime.strptime(info["Time"], "%Y-%m-%d %H:%M:%S %z") - t0).seconds)


class Reader(AbstractReader, ABC):
    priority = 10

    @staticmethod
    def _can_open(path):
        return isinstance(path, Path) and path.is_dir()

    @cached_property
    def ome(self):
        ome = model.OME()
        with tifffile.TiffFile(self.filedict[0, 0, 0]) as tif:
            metadata = {key: yaml.safe_load(value) for key, value in tif.pages[0].tags[50839].value.items()}
        ome.experimenters.append(
            model.Experimenter(id="Experimenter:0", user_name=metadata["Info"]["Summary"]["UserName"]))
        objective_str = metadata["Info"]["ZeissObjectiveTurret-Label"]
        ome.instruments.append(model.Instrument())
        ome.instruments[0].objectives.append(
            model.Objective(
                id="Objective:0", manufacturer="Zeiss", model=objective_str,
                nominal_magnification=float(re.findall(r"(\d+)x", objective_str)[0]),
                lens_na=float(re.findall(r"/(\d\.\d+)", objective_str)[0]),
                immersion=model.Objective_Immersion.OIL if 'oil' in objective_str.lower() else None))
        tubelens_str = metadata["Info"]["ZeissOptovar-Label"]
        ome.instruments[0].objectives.append(
            model.Objective(
                id="Objective:Tubelens:0", manufacturer="Zeiss", model=tubelens_str,
                nominal_magnification=float(re.findall(r"\d?\d*[,.]?\d+(?=x$)", tubelens_str)[0].replace(",", "."))))
        ome.instruments[0].detectors.append(
            model.Detector(
                id="Detector:0", amplification_gain=100))
        ome.instruments[0].filter_sets.append(
            model.FilterSet(id='FilterSet:0', model=metadata["Info"]["ZeissReflectorTurret-Label"]))

        pxsize = metadata["Info"]["PixelSizeUm"]
        pxsize_cam = 6.5 if 'Hamamatsu' in metadata["Info"]["Core-Camera"] else None
        if pxsize == 0:
            pxsize = pxsize_cam / ome.instruments[0].objectives[0].nominal_magnification
        pixel_type = metadata["Info"]["PixelType"].lower()
        if pixel_type.startswith("gray"):
            pixel_type = "uint" + pixel_type[4:]
        else:
            pixel_type = "uint16"  # assume

        size_c, size_z, size_t = (max(i) + 1 for i in zip(*self.filedict.keys()))
        t0 = datetime.strptime(metadata["Info"]["Time"], "%Y-%m-%d %H:%M:%S %z")
        ome.images.append(
            model.Image(
                pixels=model.Pixels(
                    size_c=size_c, size_z=size_z, size_t=size_t,
                    size_x=metadata['Info']['Width'], size_y=metadata['Info']['Height'],
                    dimension_order="XYCZT", type=pixel_type, physical_size_x=pxsize, physical_size_y=pxsize,
                    physical_size_z=metadata["Info"]["Summary"]["z-step_um"]),
                objective_settings=model.ObjectiveSettings(id="Objective:0")))

        for c, z, t in product(range(size_c), range(size_z), range(size_t)):
            ome.images[0].pixels.planes.append(
                Plane(t0, self.filedict[c, z, t],
                      the_c=c, the_z=z, the_t=t, exposure_time=metadata["Info"]["Exposure-ms"] / 1000))

        # compare channel names from metadata with filenames
        pattern_c = re.compile(r"img_\d{3,}_(.*)_\d{3,}$")
        for c in range(size_c):
            ome.images[0].pixels.channels.append(
                model.Channel(
                    id=f"Channel:{c}", name=pattern_c.findall(self.filedict[c, 0, 0].stem)[0],
                    detector_settings=model.DetectorSettings(
                        id="Detector:0", binning=metadata["Info"]["Hamamatsu_sCMOS-Binning"]),
                    filter_set_ref=model.FilterSetRef(id='FilterSet:0')))
        return ome

    def open(self):
        if re.match(r'(?:\d+-)?Pos.*', self.path.name) is None:
            path = self.path / f"Pos{self.series}"
        else:
            path = self.path

        filelist = sorted([file for file in path.iterdir() if re.search(r'^img_\d{3,}.*\d{3,}.*\.tif$', file.name)])
        with tifffile.TiffFile(self.path / filelist[0]) as tif:
            metadata = {key: yaml.safe_load(value) for key, value in tif.pages[0].tags[50839].value.items()}

        # compare channel names from metadata with filenames
        cnamelist = metadata["Info"]["Summary"]["ChNames"]
        cnamelist = [c for c in cnamelist if any([c in f.name for f in filelist])]

        pattern_c = re.compile(r"img_\d{3,}_(.*)_\d{3,}$")
        pattern_z = re.compile(r"(\d{3,})$")
        pattern_t = re.compile(r"img_(\d{3,})")
        self.filedict = {(cnamelist.index(pattern_c.findall(file.stem)[0]),  # noqa
                          int(pattern_z.findall(file.stem)[0]),
                          int(pattern_t.findall(file.stem)[0])): file for file in filelist}

    def __frame__(self, c=0, z=0, t=0):
        return tifffile.imread(self.path / self.filedict[(c, z, t)])
