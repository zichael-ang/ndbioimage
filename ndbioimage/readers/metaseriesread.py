import re
from abc import ABC
from pathlib import Path
from typing import Optional

import tifffile
from ome_types import model
from ome_types.units import _quantity_property  # noqa

from .. import AbstractReader


class Reader(AbstractReader, ABC):
    priority = 20
    do_not_pickle = 'last_tif'

    @staticmethod
    def _can_open(path):
        return isinstance(path, Path) and (path.is_dir() or
                                           (path.parent.is_dir() and path.name.lower().startswith('pos')))

    @staticmethod
    def get_positions(path: str | Path) -> Optional[list[int]]:
        pat = re.compile(rf's(\d)_t\d+\.(tif|TIF)$')
        return sorted({int(m.group(1)) for file in Path(path).iterdir() if (m := pat.search(file.name))})

    def get_ome(self):
        ome = model.OME()
        tif = self.get_tif(0)
        metadata = tif.metaseries_metadata
        size_z = len(tif.pages)
        page = tif.pages[0]
        shape = {axis.lower(): size for axis, size in zip(page.axes, page.shape)}
        size_x, size_y = shape['x'], shape['y']

        ome.instruments.append(model.Instrument())

        size_c = 1
        size_t = max(self.filedict.keys()) + 1
        pixel_type = f"uint{metadata['PlaneInfo']['bits-per-pixel']}"
        ome.images.append(
            model.Image(
                pixels=model.Pixels(
                    size_c=size_c, size_z=size_z, size_t=size_t,
                    size_x=size_x, size_y=size_y,
                    dimension_order='XYCZT', type=pixel_type),
                objective_settings=model.ObjectiveSettings(id='Objective:0')))
        return ome

    def open(self):
        pat = re.compile(rf's{self.series}_t\d+\.(tif|TIF)$')
        filelist = sorted([file for file in self.path.iterdir() if pat.search(file.name)])
        pattern = re.compile(r't(\d+)$')
        self.filedict = {int(pattern.search(file.stem).group(1)) - 1: file for file in filelist}
        if len(self.filedict) == 0:
            raise FileNotFoundError
        self.last_tif = 0, tifffile.TiffFile(self.filedict[0])

    def close(self) -> None:
        self.last_tif[1].close()

    def get_tif(self, t: int = None):
        last_t, tif = self.last_tif
        if (t is None or t == last_t) and not tif.filehandle.closed:
            return tif
        else:
            tif.close()
            tif = tifffile.TiffFile(self.filedict[t])
            self.last_tif = t, tif
            return tif

    def __frame__(self, c=0, z=0, t=0):
        tif = self.get_tif(t)
        page = tif.pages[z]
        if page.axes.upper() == 'YX':
            return page.asarray()
        elif page.axes.upper() == 'XY':
            return page.asarray().T
        else:
            raise NotImplementedError(f'reading axes {page.axes} is not implemented')
