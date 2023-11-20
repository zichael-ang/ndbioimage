from abc import ABC
from functools import cached_property
from itertools import product
from pathlib import Path
from struct import unpack
from warnings import warn

import numpy as np
from ome_types import model
from tifffile import TiffFile

from .. import AbstractReader


class Reader(AbstractReader, ABC):
    """ Can read some tif files written with Fiji which are broken because Fiji didn't finish writing. """
    priority = 90
    do_not_pickle = 'reader'

    @staticmethod
    def _can_open(path):
        if isinstance(path, Path) and path.suffix in ('.tif', '.tiff'):
            with TiffFile(path) as tif:
                return tif.is_imagej and not tif.is_bigtiff
        else:
            return False

    def __frame__(self, c, z, t):  # Override this, return the frame at c, z, t
        self.reader.filehandle.seek(self.offset + t * self.count)
        return np.reshape(unpack(self.fmt, self.reader.filehandle.read(self.count)), self.shape['yx'])

    def open(self):
        warn(f'File {self.path.name} is probably damaged, opening with fijiread.')
        self.reader = TiffFile(self.path)
        assert self.reader.pages[0].compression == 1, "Can only read uncompressed tiff files."
        assert self.reader.pages[0].samplesperpixel == 1, "Can only read 1 sample per pixel."
        self.offset = self.reader.pages[0].dataoffsets[0]  # noqa
        self.count = self.reader.pages[0].databytecounts[0]  # noqa
        self.bytes_per_sample = self.reader.pages[0].bitspersample // 8  # noqa
        self.fmt = self.reader.byteorder + self.count // self.bytes_per_sample * 'BHILQ'[self.bytes_per_sample - 1]  # noqa

    def close(self):
        self.reader.close()

    @cached_property
    def ome(self):
        size_y, size_x = self.reader.pages[0].shape
        size_c, size_z = 1, 1
        size_t = int(np.floor((self.reader.filehandle.size - self.reader.pages[0].dataoffsets[0]) / self.count))
        pixel_type = model.PixelType(self.reader.pages[0].dtype.name)
        ome = model.OME()
        ome.instruments.append(model.Instrument())
        ome.images.append(
            model.Image(
                pixels=model.Pixels(
                    size_c=size_c, size_z=size_z, size_t=size_t, size_x=size_x, size_y=size_y,
                    dimension_order="XYCZT", type=pixel_type),
                objective_settings=model.ObjectiveSettings(id="Objective:0")))
        for c, z, t in product(range(size_c), range(size_z), range(size_t)):
            ome.images[0].pixels.planes.append(model.Plane(the_c=c, the_z=z, the_t=t, delta_t=0))
        return ome
