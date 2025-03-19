import re
import warnings
from abc import ABC
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
import yaml
from ome_types import from_xml, model

from .. import AbstractReader, try_default


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
    def metadata(self):
        return {key: try_default(yaml.safe_load, value, value) if isinstance(value, str) else value
                for key, value in self.reader.imagej_metadata.items()}

    def get_ome(self):
        if self.reader.is_ome:
            match = re.match(r'^(.*)(pos.*)$', self.path.stem, flags=re.IGNORECASE)
            if match is not None and len(match.groups()) == 2:
                a, b = match.groups()
                with tifffile.TiffFile(self.path.with_stem(a + re.sub(r'\d', '0', b))) as file0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        ome = from_xml(file0.ome_metadata)
                        ome.images = [image for image in ome.images if self.path.stem[:len(image.name)] == image.name]
                        return ome

        page = self.reader.pages[0]
        size_y = page.imagelength
        size_x = page.imagewidth
        if self.p_ndim == 3:
            size_c = page.samplesperpixel
            size_t = self.metadata.get('frames', 1)  # // C
        else:
            size_c = self.metadata.get('channels', 1)
            size_t = self.metadata.get('frames', 1)
        size_z = self.metadata.get('slices', 1)
        if 282 in page.tags and 296 in page.tags and page.tags[296].value == 1:
            f = page.tags[282].value
            pxsize = f[1] / f[0]
        else:
            pxsize = None

        dtype = page.dtype.name
        if dtype not in ('int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32',
                         'float', 'double', 'complex', 'double-complex', 'bit'):
            dtype = 'float'

        interval_t = self.metadata.get('interval', 0)

        ome = model.OME()
        ome.instruments.append(model.Instrument(id='Instrument:0'))
        ome.instruments[0].objectives.append(model.Objective(id='Objective:0'))
        ome.images.append(
            model.Image(
                id='Image:0',
                pixels=model.Pixels(
                    id='Pixels:0',
                    size_c=size_c, size_z=size_z, size_t=size_t, size_x=size_x, size_y=size_y,
                    dimension_order='XYCZT', type=dtype,  # type: ignore
                    physical_size_x=pxsize, physical_size_y=pxsize),
                objective_settings=model.ObjectiveSettings(id='Objective:0')))
        for c, z, t in product(range(size_c), range(size_z), range(size_t)):
            ome.images[0].pixels.planes.append(model.Plane(the_c=c, the_z=z, the_t=t, delta_t=interval_t * t))
        return ome

    def open(self):
        self.reader = tifffile.TiffFile(self.path)
        page = self.reader.pages.first
        self.p_ndim = page.ndim  # noqa
        if self.p_ndim == 3:
            self.p_transpose = [i for i in [page.axes.find(j) for j in 'SYX'] if i >= 0]  # noqa
        else:
            self.p_transpose = [i for i in [page.axes.find(j) for j in 'YX'] if i >= 0]  # noqa

    def close(self):
        self.reader.close()

    def __frame__(self, c: int, z: int, t: int):
        dimension_order = self.ome.images[0].pixels.dimension_order.value
        if self.p_ndim == 3:
            axes = ''.join([ax.lower() for ax in dimension_order if ax.lower() in 'zt'])
            ct = {'z': z, 't': t}
            n = sum([ct[ax] * np.prod(self.base_shape[axes[:i]]) for i, ax in enumerate(axes)])
            return np.transpose(self.reader.asarray(int(n)), self.p_transpose)[int(c)]
        else:
            axes = ''.join([ax.lower() for ax in dimension_order if ax.lower() in 'czt'])
            czt = {'c': c, 'z': z, 't': t}
            n = sum([czt[ax] * np.prod(self.base_shape[axes[:i]]) for i, ax in enumerate(axes)])
        return np.transpose(self.reader.asarray(int(n)), self.p_transpose)
