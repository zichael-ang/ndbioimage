import yaml
import re
import numpy as np
from copy import deepcopy
from pathlib import Path

try:
    # best if SimpleElastix is installed: https://simpleelastix.readthedocs.io/GettingStarted.html
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame, Series = None, None


if hasattr(yaml, 'full_load'):
    yamlload = yaml.full_load
else:
    yamlload = yaml.load


class Transforms(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args[1:], **kwargs)
        self.default = Transform()
        if len(args):
            self.load(args[0])

    def __mul__(self, other):
        new = Transforms()
        if isinstance(other, Transforms):
            for key0, value0 in self.items():
                for key1, value1 in other.items():
                    new[key0 + key1] = value0 * value1
            return new
        elif other is None:
            return self
        else:
            for key in self.keys():
                new[key] = self[key] * other
            return new

    def asdict(self):
        return {':'.join(str(i).replace('\\', '\\\\').replace(':', r'\:') for i in key): value.asdict()
                for key, value in self.items()}

    def load(self, file):
        if isinstance(file, dict):
            d = file
        else:
            with open(file.with_suffix(".yml"), 'r') as f:
                d = yamlload(f)
        pattern = re.compile(r'[^\\]:')
        for key, value in d.items():
            self[tuple(i.replace(r'\:', ':').replace('\\\\', '\\') for i in pattern.split(key))] = Transform(value)

    def __missing__(self, key):
        return self.default

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __hash__(self):
        return hash(frozenset((*self.__dict__.items(), *self.items())))

    def save(self, file):
        with open(file.with_suffix(".yml"), 'w') as f:
            yaml.safe_dump(self.asdict(), f, default_flow_style=None)

    def copy(self):
        return deepcopy(self)

    def adapt(self, origin, shape):
        for value in self.values():
            value.adapt(origin, shape)
        self.default.adapt(origin, shape)

    @property
    def inverse(self):
        inverse = self.copy()
        for key, value in self.items():
            inverse[key] = value.inverse
        return inverse

    @property
    def ndim(self):
        return len(list(self.keys())[0])


class Transform:
    def __init__(self, *args):
        if sitk is None:
            raise ImportError('SimpleElastix is not installed: '
                              'https://simpleelastix.readthedocs.io/GettingStarted.html')
        self.transform = sitk.ReadTransform(str(Path(__file__).parent / 'transform.txt'))
        self.dparameters = 0, 0, 0, 0, 0, 0
        self.shape = 512, 512
        self.origin = 255.5, 255.5
        if len(args) == 1:  # load from file or dict
            if isinstance(args[0], np.ndarray):
                self.matrix = args[0]
            else:
                self.load(*args)
        elif len(args) > 1:  # make new transform using fixed and moving image
            self.register(*args)
        self._last, self._inverse = None, None

    def __mul__(self, other):  # TODO: take care of dmatrix
        result = self.copy()
        if isinstance(other, Transform):
            result.matrix = self.matrix @ other.matrix
            result.dmatrix = self.dmatrix @ other.matrix + self.matrix @ other.dmatrix
        else:
            result.matrix = self.matrix @ other
            result.dmatrix = self.dmatrix @ other
        return result

    def __reduce__(self):
        return self.__class__, (self.asdict(),)

    def is_unity(self):
        return self.parameters == [1, 0, 0, 1, 0, 0]

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def cast_image(im):
        if not isinstance(im, sitk.Image):
            im = sitk.GetImageFromArray(im)
        return im

    @staticmethod
    def cast_array(im):
        if isinstance(im, sitk.Image):
            im = sitk.GetArrayFromImage(im)
        return im

    @property
    def matrix(self):
        return np.array(((*self.parameters[:2], self.parameters[4]),
                         (*self.parameters[2:4], self.parameters[5]),
                         (0, 0, 1)))

    @matrix.setter
    def matrix(self, value):
        value = np.asarray(value)
        self.parameters = [*value[0, :2], *value[1, :2], *value[:2, 2]]

    @property
    def dmatrix(self):
        return np.array(((*self.dparameters[:2], self.dparameters[4]),
                         (*self.dparameters[2:4], self.dparameters[5]),
                         (0, 0, 0)))

    @dmatrix.setter
    def dmatrix(self, value):
        value = np.asarray(value)
        self.dparameters = [*value[0, :2], *value[1, :2], *value[:2, 2]]

    @property
    def parameters(self):
        return self.transform.GetParameters()

    @parameters.setter
    def parameters(self, value):
        value = np.asarray(value)
        self.transform.SetParameters(value.tolist())

    @property
    def origin(self):
        return self.transform.GetFixedParameters()

    @origin.setter
    def origin(self, value):
        value = np.asarray(value)
        self.transform.SetFixedParameters(value.tolist())

    @property
    def inverse(self):
        if self._last is None or self._last != self.asdict():
            self._last = self.asdict()
            self._inverse = Transform(self.asdict())
            self._inverse.transform = self._inverse.transform.GetInverse()
            self._inverse._last = self._inverse.asdict()
            self._inverse._inverse = self
        return self._inverse

    def adapt(self, origin, shape):
        self.origin -= np.array(origin) + (self.shape - np.array(shape)[:2]) / 2
        self.shape = shape[:2]

    def asdict(self):
        return {'CenterOfRotationPoint': self.origin, 'Size': self.shape, 'TransformParameters': self.parameters,
                'dTransformParameters': np.nan_to_num(self.dparameters, nan=1e99).tolist()}

    def frame(self, im, default=0):
        if self.is_unity():
            return im
        else:
            dtype = im.dtype
            im = im.astype('float')
            intp = sitk.sitkBSpline if np.issubdtype(dtype, np.floating) else sitk.sitkNearestNeighbor
            return self.cast_array(sitk.Resample(self.cast_image(im), self.transform, intp, default)).astype(dtype)

    def coords(self, array, columns=None):
        """ Transform coordinates in 2 column numpy array,
            or in pandas DataFrame or Series objects in columns ['x', 'y']
        """
        if self.is_unity():
            return array.copy()
        elif DataFrame is not None and isinstance(array, (DataFrame, Series)):
            columns = columns or ['x', 'y']
            array = array.copy()
            if isinstance(array, DataFrame):
                array[columns] = self.coords(np.atleast_2d(array[columns].to_numpy()))
            elif isinstance(array, Series):
                array[columns] = self.coords(np.atleast_2d(array[columns].to_numpy()))[0]
            return array
        else:  # somehow we need to use the inverse here to get the same effect as when using self.frame
            return np.array([self.inverse.transform.TransformPoint(i.tolist()) for i in np.asarray(array)])

    def save(self, file):
        """ save the parameters of the transform calculated
            with affine_registration to a yaml file
        """
        if not file[-3:] == 'yml':
            file += '.yml'
        with open(file, 'w') as f:
            yaml.safe_dump(self.asdict(), f, default_flow_style=None)

    def load(self, file):
        """ load the parameters of a transform from a yaml file or a dict
        """
        if isinstance(file, dict):
            d = file
        else:
            if not file[-3:] == 'yml':
                file += '.yml'
            with open(file, 'r') as f:
                d = yamlload(f)
        self.origin = [float(i) for i in d['CenterOfRotationPoint']]
        self.parameters = [float(i) for i in d['TransformParameters']]
        self.dparameters = [float(i) for i in d['dTransformParameters']] \
            if 'dTransformParameters' in d else 6 * [np.nan]
        self.shape = [float(i) for i in d['Size']]

    def register(self, fix, mov, kind=None):
        """ kind: 'affine', 'translation', 'rigid'
        """
        kind = kind or 'affine'
        self.shape = fix.shape
        fix, mov = self.cast_image(fix), self.cast_image(mov)
        # TODO: implement RigidTransform
        tfilter = sitk.ElastixImageFilter()
        tfilter.LogToConsoleOff()
        tfilter.SetFixedImage(fix)
        tfilter.SetMovingImage(mov)
        tfilter.SetParameterMap(sitk.GetDefaultParameterMap(kind))
        tfilter.Execute()
        transform = tfilter.GetTransformParameterMap()[0]
        if kind == 'affine':
            self.parameters = [float(t) for t in transform['TransformParameters']]
            self.shape = [float(t) for t in transform['Size']]
            self.origin = [float(t) for t in transform['CenterOfRotationPoint']]
        elif kind == 'translation':
            self.parameters = [1.0, 0.0, 0.0, 1.0] + [float(t) for t in transform['TransformParameters']]
            self.shape = [float(t) for t in transform['Size']]
            self.origin = [(t - 1) / 2 for t in self.shape]
        else:
            raise NotImplementedError(f'{kind} tranforms not implemented (yet)')
        self.dparameters = 6 * [np.nan]
