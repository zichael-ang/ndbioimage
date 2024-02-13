from abc import ABC
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
import yaml
from ome_types import model

from .. import AbstractReader


class Reader(AbstractReader, ABC):
    priority = 0
    do_not_pickle = 'reader'

    @staticmethod
    def _can_open(path):
        if isinstance(path, Path) and path.suffix in ('.tif', '.tiff'):
            with tifffile.TiffFile(path) as tif:
                return tif.is_imagej and tif.pages[-1]._nextifd() == 0  # noqa
        else:
            return False

    @cached_property
    def ome(self):
        metadata = {key: yaml.safe_load(value) if isinstance(value, str) else value
                    for key, value in self.reader.imagej_metadata.items()}

        page = self.reader.pages[0]
        self.p_ndim = page.ndim  # noqa
        size_x = page.imagelength
        size_y = page.imagewidth
        if self.p_ndim == 3:
            size_c = page.samplesperpixel
            self.p_transpose = [i for i in [page.axes.find(j) for j in 'SYX'] if i >= 0]  # noqa
            size_t = metadata.get('frames', 1)  # // C
        else:
            size_c = metadata.get('channels', 1)
            size_t = metadata.get('frames', 1)
        size_z = metadata.get('slices', 1)
        if 282 in page.tags and 296 in page.tags and page.tags[296].value == 1:
            f = page.tags[282].value
            pxsize = f[1] / f[0]
        else:
            pxsize = None

        dtype = page.dtype.name
        if dtype not in ('int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32',
                         'float', 'double', 'complex', 'double-complex', 'bit'):
            dtype = 'float'

        interval_t = metadata.get('interval', 0)

        ome = model.OME()
        ome.instruments.append(model.Instrument(id='Instrument:0'))
        ome.instruments[0].objectives.append(model.Objective(id='Objective:0'))
        ome.images.append(
            model.Image(
                id='Image:0',
                pixels=model.Pixels(
                    id='Pixels:0',
                    size_c=size_c, size_z=size_z, size_t=size_t, size_x=size_x, size_y=size_y,
                    dimension_order="XYCZT", type=dtype, physical_size_x=pxsize, physical_size_y=pxsize),
                objective_settings=model.ObjectiveSettings(id="Objective:0")))
        for c, z, t in product(range(size_c), range(size_z), range(size_t)):
            ome.images[0].pixels.planes.append(model.Plane(the_c=c, the_z=z, the_t=t, delta_t=interval_t * t))
        return ome

    def open(self):
        self.reader = tifffile.TiffFile(self.path)

    def close(self):
        self.reader.close()

    def __frame__(self, c, z, t):
        if self.p_ndim == 3:
            return np.transpose(self.reader.asarray(z + t * self.base.shape['z']), self.p_transpose)[c]
        else:
            return self.reader.asarray(c + z * self.base.shape['c'] + t * self.base.shape['c'] * self.base.shape['z'])
