from ndbioimage import Imread
import numpy as np


class Reader(Imread):
    priority = 20

    @staticmethod
    def _can_open(path):
        return isinstance(path, np.ndarray) and 1 <= path.ndim <= 5

    def __metadata__(self):
        self.base = np.array(self.path, ndmin=5)
        self.title = self.path = 'numpy array'
        self.axes = self.axes[:self.base.ndim]
        self.shape = self.base.shape
        self.acquisitiondate = 'now'

    def __frame__(self, c, z, t):
        xyczt = (slice(None), slice(None), c, z, t)
        in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
        frame = self.base[in_idx]
        if self.axes.find('y') < self.axes.find('x'):
            return frame.T
        else:
            return frame

    def __str__(self):
        return self.path
