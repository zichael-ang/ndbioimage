from ndbioimage import Imread, XmlData
import numpy as np
import tifffile
import yaml


class Reader(Imread):
    priority = 0
    do_not_pickle = 'reader'

    @staticmethod
    def _can_open(path):
        if isinstance(path, str) and (path.endswith('.tif') or path.endswith('.tiff')):
            with tifffile.TiffFile(path) as tif:
                return tif.is_imagej
        else:
            return False

    def open(self):
        self.reader = tifffile.TiffFile(self.path)

    def close(self):
        self.reader.close()

    def __metadata__(self):
        self.metadata = XmlData({key: yaml.safe_load(value) if isinstance(value, str) else value
                                 for key, value in self.reader.imagej_metadata.items()})
        P = self.reader.pages[0]
        self.pndim = P.ndim
        X = P.imagelength
        Y = P.imagewidth
        if self.pndim == 3:
            C = P.samplesperpixel
            self.transpose = [i for i in [P.axes.find(j) for j in 'SYX'] if i >= 0]
            T = self.metadata.get('frames', 1)  # // C
        else:
            C = self.metadata.get('channels', 1)
            T = self.metadata.get('frames', 1)
        Z = self.metadata.get('slices', 1)
        self.shape = (X, Y, C, Z, T)
        if 282 in P.tags and 296 in P.tags and P.tags[296].value == 1:
            f = P.tags[282].value
            self.pxsize = f[1] / f[0]
        # TODO: more metadata

    def __frame__(self, c, z, t):
        if self.pndim == 3:
            return np.transpose(self.reader.asarray(z + t * self.shape[3]), self.transpose)[c]
        else:
            return self.reader.asarray(c + z * self.shape[2] + t * self.shape[2] * self.shape[3])
