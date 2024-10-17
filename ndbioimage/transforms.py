import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from parfor import Chunks, pmap
from skimage import filters
from tiffwrite import IJTiffFile
from tqdm.auto import tqdm

try:
    # best if SimpleElastix is installed: https://simpleelastix.readthedocs.io/GettingStarted.html
    import SimpleITK as sitk  # noqa
except ImportError:
    sitk = None

try:
    from pandas import DataFrame, Series, concat
except ImportError:
    DataFrame, Series, concat = None, None, None


if hasattr(yaml, 'full_load'):
    yamlload = yaml.full_load
else:
    yamlload = yaml.load


class Transforms(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = Transform()

    @classmethod
    def from_file(cls, file, C=True, T=True):
        with open(Path(file).with_suffix('.yml')) as f:
            return cls.from_dict(yamlload(f), C, T)

    @classmethod
    def from_dict(cls, d, C=True, T=True):
        new = cls()
        for key, value in d.items():
            if isinstance(key, str) and C:
                new[key.replace(r'\:', ':').replace('\\\\', '\\')] = Transform.from_dict(value)
            elif T:
                new[key] = Transform.from_dict(value)
        return new

    @classmethod
    def from_shifts(cls, shifts):
        new = cls()
        for key, shift in shifts.items():
            new[key] = Transform.from_shift(shift)
        return new

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
        return {key.replace('\\', '\\\\').replace(':', r'\:') if isinstance(key, str) else key: value.asdict()
                for key, value in self.items()}

    def __getitem__(self, item):
        return np.prod([self[i] for i in item[::-1]]) if isinstance(item, tuple) else super().__getitem__(item)

    def __missing__(self, key):
        return self.default

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __hash__(self):
        return hash(frozenset((*self.__dict__.items(), *self.items())))

    def save(self, file):
        with open(Path(file).with_suffix('.yml'), 'w') as f:
            yaml.safe_dump(self.asdict(), f, default_flow_style=None)

    def copy(self):
        return deepcopy(self)

    def adapt(self, origin, shape, channel_names):
        def key_map(a, b):
            def fun(b, key_a):
                for key_b in b:
                    if key_b in key_a or key_a in key_b:
                        return key_a, key_b

            return {n[0]: n[1] for key_a in a if (n := fun(b, key_a))}

        for value in self.values():
            value.adapt(origin, shape)
        self.default.adapt(origin, shape)
        transform_channels = {key for key in self.keys() if isinstance(key, str)}
        if set(channel_names) - transform_channels:
            mapping = key_map(channel_names, transform_channels)
            warnings.warn(f'The image file and the transform do not have the same channels,'
                          f' creating a mapping: {mapping}')
            for key_im, key_t in mapping.items():
                self[key_im] = self[key_t]

    @property
    def inverse(self):
        # TODO: check for C@T
        inverse = self.copy()
        for key, value in self.items():
            inverse[key] = value.inverse
        return inverse

    def coords_pandas(self, array, channel_names, columns=None):
        if isinstance(array, DataFrame):
            return concat([self.coords_pandas(row, channel_names, columns) for _, row in array.iterrows()], axis=1).T
        elif isinstance(array, Series):
            key = []
            if 'C' in array:
                key.append(channel_names[int(array['C'])])
            if 'T' in array:
                key.append(int(array['T']))
            return self[tuple(key)].coords(array, columns)
        else:
            raise TypeError('Not a pandas DataFrame or Series.')

    def with_beads(self, cyllens, bead_files):
        assert len(bead_files) > 0, 'At least one file is needed to calculate the registration.'
        transforms = [self.calculate_channel_transforms(file, cyllens) for file in bead_files]
        for key in {key for transform in transforms for key in transform.keys()}:
            new_transforms = [transform[key] for transform in transforms if key in transform]
            if len(new_transforms) == 1:
                self[key] = new_transforms[0]
            else:
                self[key] = Transform()
                self[key].parameters = np.mean([t.parameters for t in new_transforms], 0)
                self[key].dparameters = (np.std([t.parameters for t in new_transforms], 0) /
                                        np.sqrt(len(new_transforms))).tolist()
        return self

    @staticmethod
    def get_bead_files(path):
        from . import Imread
        files = []
        for file in path.iterdir():
            if file.name.lower().startswith('beads'):
                try:
                    with Imread(file):
                        files.append(file)
                except Exception:
                    pass
        files = sorted(files)
        if not files:
            raise Exception('No bead file found!')
        checked_files = []
        for file in files:
            try:
                if file.is_dir():
                    file /= 'Pos0'
                with Imread(file):  # check for errors opening the file
                    checked_files.append(file)
            except (Exception,):
                continue
        if not checked_files:
            raise Exception('No bead file found!')
        return checked_files

    @staticmethod
    def calculate_channel_transforms(bead_file, cyllens):
        """ When no channel is not transformed by a cylindrical lens, assume that the image is scaled by a factor 1.162
            in the horizontal direction """
        from . import Imread

        with Imread(bead_file, axes='zcyx') as im:  # noqa
            max_ims = im.max('z')
            goodch = [c for c, max_im in enumerate(max_ims) if not im.is_noise(max_im)]
            if not goodch:
                goodch = list(range(len(max_ims)))
            untransformed = [c for c in range(im.shape['c']) if cyllens[im.detector[c]].lower() == 'none']

            good_and_untrans = sorted(set(goodch) & set(untransformed))
            if good_and_untrans:
                masterch = good_and_untrans[0]
            else:
                masterch = goodch[0]
            transform = Transform()
            if not good_and_untrans:
                matrix = transform.matrix
                matrix[0, 0] = 0.86
                transform.matrix = matrix
            transforms = Transforms()
            for c in tqdm(goodch, desc='Calculating channel transforms'):  # noqa
                if c == masterch:
                    transforms[im.channel_names[c]] = transform
                else:
                    transforms[im.channel_names[c]] = Transform.register(max_ims[masterch], max_ims[c]) * transform
        return transforms

    @staticmethod
    def save_channel_transform_tiff(bead_files, tiffile):
        from . import Imread
        n_channels = 0
        for file in bead_files:
            with Imread(file) as im:
                n_channels = max(n_channels, im.shape['c'])
        with IJTiffFile(tiffile) as tif:
            for t, file in enumerate(bead_files):
                with Imread(file) as im:
                    with Imread(file).with_transform() as jm:
                        for c in range(im.shape['c']):
                            tif.save(np.hstack((im(c=c, t=0).max('z'), jm(c=c, t=0).max('z'))), c, 0, t)

    def with_drift(self, im):
        """ Calculate shifts relative to the first frame
            divide the sequence into groups,
            compare each frame to the frame in the middle of the group and compare these middle frames to each other
        """
        im = im.transpose('tzycx')
        t_groups = [list(chunk) for chunk in Chunks(range(im.shape['t']), size=round(np.sqrt(im.shape['t'])))]
        t_keys = [int(np.round(np.mean(t_group))) for t_group in t_groups]
        t_pairs = [(int(np.round(np.mean(t_group))), frame) for t_group in t_groups for frame in t_group]
        t_pairs.extend(zip(t_keys, t_keys[1:]))
        fmaxz_keys = {t_key: filters.gaussian(im[t_key].max('z'), 5) for t_key in t_keys}

        def fun(t_key_t, im, fmaxz_keys):
            t_key, t = t_key_t
            if t_key == t:
                return 0, 0
            else:
                fmaxz = filters.gaussian(im[t].max('z'), 5)
                return Transform.register(fmaxz_keys[t_key], fmaxz, 'translation').parameters[4:]

        shifts = np.array(pmap(fun, t_pairs, (im, fmaxz_keys), desc='Calculating image shifts.'))
        shift_keys_cum = np.zeros(2)
        for shift_keys, t_group in zip(np.vstack((-shifts[0], shifts[im.shape['t']:])), t_groups):
            shift_keys_cum += shift_keys
            shifts[t_group] += shift_keys_cum

        for i, shift in enumerate(shifts[:im.shape['t']]):
            self[i] = Transform.from_shift(shift)
        return self


class Transform:
    def __init__(self):
        if sitk is None:
            self.transform = None
        else:
            self.transform = sitk.ReadTransform(str(Path(__file__).parent / 'transform.txt'))
        self.dparameters = [0., 0., 0., 0., 0., 0.]
        self.shape = [512., 512.]
        self.origin = [255.5, 255.5]
        self._last, self._inverse = None, None

    def __reduce__(self):
        return self.from_dict, (self.asdict(),)

    def __repr__(self):
        return self.asdict().__repr__()

    def __str__(self):
        return self.asdict().__str__()

    @classmethod
    def register(cls, fix, mov, kind=None):
        """ kind: 'affine', 'translation', 'rigid' """
        if sitk is None:
            raise ImportError('SimpleElastix is not installed: '
                              'https://simpleelastix.readthedocs.io/GettingStarted.html')
        new = cls()
        kind = kind or 'affine'
        new.shape = fix.shape
        fix, mov = new.cast_image(fix), new.cast_image(mov)
        # TODO: implement RigidTransform
        tfilter = sitk.ElastixImageFilter()
        tfilter.LogToConsoleOff()
        tfilter.SetFixedImage(fix)
        tfilter.SetMovingImage(mov)
        tfilter.SetParameterMap(sitk.GetDefaultParameterMap(kind))
        tfilter.Execute()
        transform = tfilter.GetTransformParameterMap()[0]
        if kind == 'affine':
            new.parameters = [float(t) for t in transform['TransformParameters']]
            new.shape = [float(t) for t in transform['Size']]
            new.origin = [float(t) for t in transform['CenterOfRotationPoint']]
        elif kind == 'translation':
            new.parameters = [1.0, 0.0, 0.0, 1.0] + [float(t) for t in transform['TransformParameters']]
            new.shape = [float(t) for t in transform['Size']]
            new.origin = [(t - 1) / 2 for t in new.shape]
        else:
            raise NotImplementedError(f'{kind} tranforms not implemented (yet)')
        new.dparameters = 6 * [np.nan]
        return new

    @classmethod
    def from_shift(cls, shift):
        return cls.from_array(np.array(((1, 0, shift[0]), (0, 1, shift[1]), (0, 0, 1))))

    @classmethod
    def from_array(cls, array):
        new = cls()
        new.matrix = array
        return new

    @classmethod
    def from_file(cls, file):
        with open(Path(file).with_suffix('.yml')) as f:
            return cls.from_dict(yamlload(f))

    @classmethod
    def from_dict(cls, d):
        new = cls()
        new.origin = [float(i) for i in d['CenterOfRotationPoint']]
        new.parameters = [float(i) for i in d['TransformParameters']]
        new.dparameters = [float(i) for i in d['dTransformParameters']] if 'dTransformParameters' in d else 6 * [np.nan]
        new.shape = [float(i) for i in d['Size']]
        return new

    def __mul__(self, other):  # TODO: take care of dmatrix
        result = self.copy()
        if isinstance(other, Transform):
            result.matrix = self.matrix @ other.matrix
            result.dmatrix = self.dmatrix @ other.matrix + self.matrix @ other.dmatrix
        else:
            result.matrix = self.matrix @ other
            result.dmatrix = self.dmatrix @ other
        return result

    def is_unity(self):
        return self.parameters == [1, 0, 0, 1, 0, 0]

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def cast_image(im):
        if not isinstance(im, sitk.Image):
            im = sitk.GetImageFromArray(np.asarray(im))
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
        if self.transform is not None:
            return list(self.transform.GetParameters())

    @parameters.setter
    def parameters(self, value):
        if self.transform is not None:
            value = np.asarray(value)
            self.transform.SetParameters(value.tolist())

    @property
    def origin(self):
        if self.transform is not None:
            return self.transform.GetFixedParameters()

    @origin.setter
    def origin(self, value):
        if self.transform is not None:
            value = np.asarray(value)
            self.transform.SetFixedParameters(value.tolist())

    @property
    def inverse(self):
        if self.is_unity():
            return self
        if self._last is None or self._last != self.asdict():
            self._last = self.asdict()
            self._inverse = Transform.from_dict(self.asdict())
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
            if sitk is None:
                raise ImportError('SimpleElastix is not installed: '
                                  'https://simpleelastix.readthedocs.io/GettingStarted.html')
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
