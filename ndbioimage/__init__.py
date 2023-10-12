import re
import warnings

import pandas
import yaml
import numpy as np
import multiprocessing
import ome_types
from ome_types import ureg, model
from pint import set_application_registry
from datetime import datetime
from tqdm.auto import tqdm
from itertools import product
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from functools import cached_property, wraps
from parfor import parfor
from tiffwrite import IJTiffFile
from numbers import Number
from argparse import ArgumentParser
from pathlib import Path
from importlib.metadata import version
from traceback import print_exc
from operator import truediv
from .transforms import Transform, Transforms
from .jvm import JVM

try:
    __version__ = version(Path(__file__).parent.name)
except (Exception,):
    __version__ = 'unknown'

try:
    with open(Path(__file__).parent.parent / '.git' / 'HEAD') as g:
        head = g.read().split(':')[1].strip()
    with open(Path(__file__).parent.parent / '.git' / head) as h:
        __git_commit_hash__ = h.read().rstrip('\n')
except (Exception,):
    __git_commit_hash__ = 'unknown'

ureg.default_format = '~P'
set_application_registry(ureg)
warnings.filterwarnings('ignore', 'Reference to unknown ID')


class ReaderNotFoundError(Exception):
    pass


class ImTransforms(Transforms):
    """ Transforms class with methods to calculate channel transforms from bead files etc. """

    def __init__(self, path, cyllens, file=None, transforms=None):
        super().__init__()
        self.cyllens = tuple(cyllens)
        if transforms is None:
            # TODO: check this
            if re.search(r'^Pos\d+', path.name):
                self.path = path.parent.parent
            else:
                self.path = path.parent
            if file is not None:
                if isinstance(file, str) and file.lower().endswith('.yml'):
                    self.ymlpath = file
                    self.beadfile = None
                else:
                    self.ymlpath = self.path / 'transform.yml'
                    self.beadfile = file
            else:
                self.ymlpath = self.path / 'transform.yml'
                self.beadfile = None
            self.tifpath = self.ymlpath.with_suffix('.tif')
            try:
                self.load(self.ymlpath)
            except (Exception,):
                print('No transform file found, trying to generate one.')
                if not self.files:
                    raise FileNotFoundError('No bead files found to calculate the transform from.')
                self.calculate_transforms()
                self.save(self.ymlpath)
                self.save_transform_tiff()
                print(f'Saving transform in {self.ymlpath}.')
                print(f'Please check the transform in {self.tifpath}.')
        else:  # load from dict transforms
            self.path = path
            self.beadfile = file
            for i, (key, value) in enumerate(transforms.items()):
                self[key] = Transform(value)

    def coords_pandas(self, array, cnamelist, colums=None):
        if isinstance(array, pandas.DataFrame):
            return pandas.concat([self[(cnamelist[int(row['C'])],)].coords(row, colums)
                                  for _, row in array.iterrows()], axis=1).T
        elif isinstance(array, pandas.Series):
            return self[(cnamelist[int(array['C'])],)].coords(array, colums)
        else:
            raise TypeError('Not a pandas DataFrame or Series.')

    @cached_property
    def files(self):
        try:
            if self.beadfile is None:
                files = self.get_bead_files()
            else:
                files = self.beadfile
            if isinstance(files, str):
                files = (Path(files),)
            elif isinstance(files, Path):
                files = (files,)
            return tuple(files)
        except (Exception,):
            return ()

    def get_bead_files(self):
        files = sorted([f for f in self.path.iterdir() if f.name.lower().startswith('beads')
                        and not f.suffix.lower() == '.pdf' and not f.suffix.lower() == 'pkl'])
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

    def calculate_transform(self, file):
        """ When no channel is not transformed by a cylindrical lens, assume that the image is scaled by a factor 1.162
            in the horizontal direction """
        with Imread(file, axes='zcxy') as im:
            max_ims = im.max('z')
            goodch = [c for c, max_im in enumerate(max_ims) if not im.is_noise(max_im)]
            if not goodch:
                goodch = list(range(len(max_ims)))
            untransformed = [c for c in range(im.shape['c']) if self.cyllens[im.detector[c]].lower() == 'none']

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
            for c in tqdm(goodch):
                if c == masterch:
                    transforms[(im.cnamelist[c],)] = transform
                else:
                    transforms[(im.cnamelist[c],)] = Transform(max_ims[masterch], max_ims[c]) * transform
        return transforms

    def calculate_transforms(self):
        transforms = [self.calculate_transform(file) for file in self.files]
        for key in set([key for transform in transforms for key in transform.keys()]):
            new_transforms = [transform[key] for transform in transforms if key in transform]
            if len(new_transforms) == 1:
                self[key] = new_transforms[0]
            else:
                self[key] = Transform()
                self[key].parameters = np.mean([t.parameters for t in new_transforms], 0)
                self[key].dparameters = (np.std([t.parameters for t in new_transforms], 0) /
                                         np.sqrt(len(new_transforms))).tolist()

    def save_transform_tiff(self):
        n_channels = 0
        for file in self.files:
            with Imread(file) as im:
                n_channels = max(n_channels, im.shape['c'])
        with IJTiffFile(self.tifpath, (n_channels, 1, len(self.files))) as tif:
            for t, file in enumerate(self.files):
                with Imread(file) as im:
                    with Imread(file, transform=True) as jm:
                        for c in range(im.shape['c']):
                            tif.save(np.hstack((im(c=c, t=0).max('z'), jm(c=c, t=0).max('z'))), c, 0, t)


class ImShiftTransforms(Transforms):
    """ Class to handle drift in xy. The image this is applied to must have a channel transform already, which is then
        replaced by this class. """

    def __init__(self, im, shifts=None):
        """ im:                     Calculate shifts from channel-transformed images
            im, t x 2 array         Sets shifts from array, one row per frame
            im, dict {frame: shift} Sets shifts from dict, each key is a frame number from where a new shift is applied
            im, file                Loads shifts from a saved file """
        super().__init__()
        with (Imread(im, transform=True, drift=False) if isinstance(im, str)
                                                      else im.new(transform=True, drift=False)) as im:
            self.impath = im.path
            self.path = self.impath.parent / self.impath.stem + '_shifts.txt'
            self.tracks, self.detectors, self.files = im.track, im.detector, im.beadfile
            if shifts is not None:
                if isinstance(shifts, np.ndarray):
                    self.shifts = shifts
                    self.shifts2transforms(im)
                elif isinstance(shifts, dict):
                    self.shifts = np.zeros((im.shape['t'], 2))
                    for k in sorted(shifts.keys()):
                        self.shifts[k:] = shifts[k]
                    self.shifts2transforms(im)
                elif isinstance(shifts, str):
                    self.load(im, shifts)
            elif self.path.exists():
                self.load(im, self.path)
            else:
                self.calulate_shifts(im)
                self.save()

    def __call__(self, channel, time, tracks=None, detectors=None):
        tracks = tracks or self.tracks
        detectors = detectors or self.detectors
        track, detector = tracks[channel], detectors[channel]
        if (track, detector, time) in self:
            return self[track, detector, time]
        elif (0, detector, time) in self:
            return self[0, detector, time]
        else:
            return Transform()

    def load(self, im, file):
        self.shifts = np.loadtxt(file)
        self.shifts2transforms(im)

    def save(self, file=None):
        self.path = file or self.path
        np.savetxt(self.path, self.shifts)

    def coords(self, array, colums=None):
        if isinstance(array, pandas.DataFrame):
            return pandas.concat([self(int(row['C']), int(row['T'])).coords(row, colums)
                                  for _, row in array.iterrows()], axis=1).T
        elif isinstance(array, pandas.Series):
            return self(int(array['C']), int(array['T'])).coords(array, colums)
        else:
            raise TypeError('Not a pandas DataFrame or Series.')

    def calulate_shifts0(self, im):
        """ Calculate shifts relative to the first frame """
        im0 = im[:, 0, 0].squeeze().transpose(2, 0, 1)

        @parfor(range(1, im.shape['t']), (im, im0), desc='Calculating image shifts.')
        def fun(t, im, im0):
            return Transform(im0, im[:, 0, t].squeeze().transpose(2, 0, 1), 'translation')

        transforms = [Transform()] + fun
        self.shifts = np.array([t.parameters[4:] for t in transforms])
        self.set_transforms(transforms, im.transform)

    def calulate_shifts(self, im):
        """ Calculate shifts relative to the previous frame """

        @parfor(range(1, im.shape['t']), (im,), desc='Calculating image shifts.')
        def fun(t, im):
            return Transform(im[:, 0, t - 1].squeeze().transpose(2, 0, 1), im[:, 0, t].squeeze().transpose(2, 0, 1),
                             'translation')

        transforms = [Transform()] + fun
        self.shifts = np.cumsum([t.parameters[4:] for t in transforms])
        self.set_transforms(transforms, im.transform)

    def shifts2transforms(self, im):
        self.set_transforms([Transform(np.array(((1, 0, s[0]), (0, 1, s[1]), (0, 0, 1))))
                             for s in self.shifts], im.transform)

    def set_transforms(self, shift_transforms, channel_transforms):
        for key, value in channel_transforms.items():
            for t, T in enumerate(shift_transforms):
                self[key[0], key[1], t] = T * channel_transforms[key]


class DequeDict(OrderedDict):
    def __init__(self, maxlen=None, *args, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __truncate__(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(False)

    def __setitem__(self, *args, **kwargs):
        super().__setitem__(*args, **kwargs)
        self.__truncate__()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.__truncate__()


def find(obj, **kwargs):
    for item in obj:
        if all([getattr(item, key) == value for key, value in kwargs.items()]):
            return item


def try_default(fun, default, *args, **kwargs):
    try:
        return fun(*args, **kwargs)
    except (Exception,):
        return default


def get_ome(path):
    from .readers.bfread import jars
    try:
        jvm = JVM(jars)
        ome_meta = jvm.metadata_tools.createOMEXMLMetadata()
        reader = jvm.image_reader()
        reader.setMetadataStore(ome_meta)
        reader.setId(str(path))
        ome = ome_types.from_xml(str(ome_meta.dumpXML()), parser='lxml')
    except (Exception,):
        print_exc()
        ome = model.OME()
    finally:
        jvm.kill_vm()
    return ome


class Shape(tuple):
    def __new__(cls, shape, axes='xyczt'):
        if isinstance(shape, Shape):
            axes = shape.axes
        instance = super().__new__(cls, shape)
        instance.axes = axes.lower()
        return instance

    def __getitem__(self, n):
        if isinstance(n, str):
            if len(n) == 1:
                return self[self.axes.find(n.lower())] if n.lower() in self.axes else 1
            else:
                return tuple(self[i] for i in n)
        return super().__getitem__(n)

    @cached_property
    def xyczt(self):
        return tuple(self[i] for i in 'xyczt')


class Imread(np.lib.mixins.NDArrayOperatorsMixin):
    def __new__(cls, path=None, *args, **kwargs):
        if cls is not Imread:
            return super().__new__(cls)
        if len(AbstractReader.__subclasses__()) == 0:
            raise Exception('Restart python kernel please!')
        if isinstance(path, Imread):
            return path
        path, _ = AbstractReader.split_path_series(path)
        for subclass in sorted(AbstractReader.__subclasses__(), key=lambda subclass_: subclass_.priority):
            if subclass._can_open(path):
                do_not_pickle = (AbstractReader.do_not_pickle,) if isinstance(AbstractReader.do_not_pickle, str) \
                    else AbstractReader.do_not_pickle
                subclass_do_not_pickle = (subclass.do_not_pickle,) if isinstance(subclass.do_not_pickle, str) \
                    else subclass.do_not_pickle if hasattr(subclass, 'do_not_pickle') else ()
                subclass.do_not_pickle = set(do_not_pickle).union(set(subclass_do_not_pickle))

                return super().__new__(subclass)
        raise ReaderNotFoundError(f'No reader found for {path}.')

    def __init__(self, base=None, slice=None, shape=(0, 0, 0, 0, 0), dtype=None,
                 transform=False, drift=False, beadfile=None, frame_decorator=None):
        self.base = base
        self.slice = slice
        self._shape = Shape(shape)
        self.dtype = dtype
        self.frame_decorator = frame_decorator

        self.transform = transform
        self.drift = drift
        self.beadfile = beadfile

        self.flags = dict(C_CONTIGUOUS=False, F_CONTIGUOUS=False, OWNDATA=False, WRITEABLE=False,
                          ALIGNED=False, WRITEBACKIFCOPY=False, UPDATEIFCOPY=False)

    def __call__(self, c=None, z=None, t=None, x=None, y=None):
        """ same as im[] but allowing keyword axes, but slices need to made with slice() or np.s_ """
        return self[{k: slice(v) if v is None else v for k, v in dict(c=c, z=z, t=t, x=x, y=y).items()}]

    def __copy__(self):
        return self.copy()

    def __contains__(self, item):
        def unique_yield(a, b):
            for k in a:
                yield k
            for k in b:
                if k not in a:
                    yield k

        for idx in unique_yield([key[:3] for key in self.cache.keys()],
                                product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t']))):
            xyczt = (slice(None), slice(None)) + idx
            in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
            if item in np.asarray(self[in_idx]):
                return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if not self.isclosed:
            self.isclosed = True
            if hasattr(self, 'close'):
                self.close()

    def __getitem__(self, n):
        """ slice like a numpy array but return an Imread instance """
        if self.isclosed:
            raise IOError("file is closed")
        if isinstance(n, (slice, Number)):  # None = :
            n = (n,)
        elif isinstance(n, type(Ellipsis)):
            n = (None,) * len(self.axes)
        elif isinstance(n, dict):  # allow im[dict(z=0)] etc.
            n = [n.get(i, slice(None)) for i in self.axes]
        n = list(n)

        # deal with ...
        ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
        if len(ell) > 1:
            raise IndexError('an index can only have a single ellipsis (...)')
        if len(ell):
            if len(n) > self.ndim:
                n.remove(Ellipsis)
            else:
                n[ell[0]] = None
                while len(n) < self.ndim:
                    n.insert(ell[0], None)
        while len(n) < self.ndim:
            n.append(None)

        axes_idx = [self.shape.axes.find(i) for i in 'xyczt']
        n = [n[j] if 0 <= j < len(n) else None for j in axes_idx]  # reorder n

        new_slice = []
        for s, e in zip(self.slice, n):
            if e is None:
                new_slice.append(s)
            else:
                new_slice.append(s[e])

        # TODO: check output dimensionality when requested shape in some dimension is 1
        if all([isinstance(s, Number) or s.size == 1 for s in new_slice]):
            return self.block(*new_slice).item()
        else:
            new = View(self)
            new.slice = new_slice
            new._shape = Shape([1 if isinstance(s, Number) else len(s) for s in new_slice])
            new.axes = ''.join(j for j in self.axes if j in [i for i, s in zip('xyczt', new_slice)
                                                             if not isinstance(s, Number)])
            return new

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in self.do_not_pickle}

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return self.summary

    def __setstate__(self, state):
        """ What happens during unpickling """
        self.__dict__.update(state)
        if isinstance(self, AbstractReader):
            self.open()
        self.cache = DequeDict(16)

    def __str__(self):
        return str(self.path)

    def __array__(self, dtype=None):
        block = self.block(*self.slice)
        axes_idx = [self.shape.axes.find(i) for i in 'xyczt']
        axes_squeeze = tuple({i for i, j in enumerate(axes_idx) if j == -1}.union(
            {i for i, j in enumerate(self.slice) if isinstance(j, Number)}))
        block = block.squeeze(axes_squeeze)
        if dtype is not None:
            block = block.astype(dtype)
        if block.ndim == 0:
            return block.item()
        axes = ''.join(j for i, j in enumerate('xyczt') if i not in axes_squeeze)
        return block.transpose([axes.find(i) for i in self.shape.axes if i in axes])

    def __array_arg_fun__(self, fun, axis=None, out=None):
        """ frame-wise application of np.argmin and np.argmax """
        if axis is None:
            value = arg = None
            for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                xyczt = (slice(None), slice(None)) + idx
                in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                new = np.asarray(self[in_idx])
                new_arg = np.unravel_index(fun(new), new.shape)
                new_value = new[new_arg]
                if value is None:
                    arg = new_arg + idx
                    value = new_value
                else:
                    i = fun((value, new_value))
                    arg = (arg, new_arg + idx)[i]
                    value = (value, new_value)[i]
            axes = ''.join(i for i in self.axes if i in 'xy') + 'czt'
            arg = np.ravel_multi_index([arg[axes.find(i)] for i in self.axes], self.shape)
            if out is None:
                return arg
            else:
                out.itemset(arg)
                return out
        else:
            if isinstance(axis, str):
                axis_str, axis_idx = axis, self.axes.index(axis)
            else:
                axis_str, axis_idx = self.axes[axis], axis
            if axis_str not in self.axes:
                raise IndexError(f'Axis {axis_str} not in {self.axes}.')
            out_shape = list(self.shape)
            out_axes = list(self.axes)
            out_shape.pop(axis_idx)
            out_axes.pop(axis_idx)
            if out is None:
                out = np.zeros(out_shape, int)
            if axis_str in 'xy':
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                    new = self[in_idx]
                    out[out_idx] = fun(np.asarray(new), new.axes.find(axis_str))
            else:
                value = np.zeros(out.shape, self.dtype)
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                    new_value = self[in_idx]
                    new_arg = np.full_like(new_value, idx['czt'.find(axis_str)])
                    if idx['czt'.find(axis_str)] == 0:
                        value[out_idx] = new_value
                        out[out_idx] = new_arg
                    else:
                        old_value = value[out_idx]
                        i = fun((old_value, new_value), 0)
                        value[out_idx] = np.where(i, new_value, old_value)
                        out[out_idx] = np.where(i, new_arg, out[out_idx])
            return out

    def __array_fun__(self, funs, axis=None, dtype=None, out=None, keepdims=False, initials=None, where=True,
                      ffuns=None, cfun=None):
        """ frame-wise application of np.min, np.max, np.sum, np.mean and their nan equivalents """
        p = re.compile(r'\d')
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        if initials is None:
            initials = [None for _ in funs]
        if ffuns is None:
            ffuns = [None for _ in funs]

        def ffun_(frame):
            return np.asarray(frame)
        ffuns = [ffun_ if ffun is None else ffun for ffun in ffuns]
        if cfun is None:
            def cfun(*res):
                return res[0]

        # TODO: smarter transforms
        if axis is None:
            for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                xyczt = (slice(None), slice(None)) + idx
                in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                w = where if where is None or isinstance(where, bool) else where[in_idx]
                initials = [fun(np.asarray(ffun(self[in_idx])), initial=initial, where=w)
                            for fun, ffun, initial in zip(funs, ffuns, initials)]
            res = cfun(*initials)
            res = (np.round(res) if dtype.kind in 'ui' else res).astype(p.sub('', dtype.name))
            if keepdims:
                res = np.array(res, dtype, ndmin=self.ndim)
            if out is None:
                return res
            else:
                out.itemset(res)
                return out
        else:
            if isinstance(axis, str):
                axis_str, axis_idx = axis, self.axes.index(axis)
            else:
                axis_idx = axis % self.ndim
                axis_str = self.axes[axis_idx]
            if axis_str not in self.axes:
                raise IndexError(f'Axis {axis_str} not in {self.axes}.')
            out_shape = list(self.shape)
            out_axes = list(self.axes)
            if not keepdims:
                out_shape.pop(axis_idx)
                out_axes.pop(axis_idx)
            if out is None:
                out = np.zeros(out_shape, dtype)
            if axis_str in 'xy':
                xy = 'xy' if self.axes.find('x') < self.axes.find('y') else 'yx'
                frame_ax = xy.find(axis_str)
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                    w = where if where is None or isinstance(where, bool) else where[in_idx]
                    res = cfun(*[fun(ffun(self[in_idx]), frame_ax, initial=initial, where=w)
                                 for fun, ffun, initial in zip(funs, ffuns, initials)])
                    out[out_idx] = (np.round(res) if out.dtype.kind in 'ui' else res).astype(p.sub('', dtype.name))
            else:
                tmps = [np.zeros(out_shape) for _ in ffuns]
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)

                    if idx['czt'.find(axis_str)] == 0:
                        w = where if where is None or isinstance(where, bool) else (where[in_idx],)
                        for tmp, fun, ffun, initial in zip(tmps, funs, ffuns, initials):
                            tmp[out_idx] = fun((ffun(self[in_idx]),), 0, initial=initial, where=w)
                    else:
                        w = where if where is None or isinstance(where, bool) else \
                            (np.ones_like(where[in_idx]), where[in_idx])
                        for tmp, fun, ffun in zip(tmps, funs, ffuns):
                            tmp[out_idx] = fun((tmp[out_idx], ffun(self[in_idx])), 0, where=w)
                out[...] = (np.round(cfun(*tmps)) if out.dtype.kind in 'ui' else
                            cfun(*tmps)).astype(p.sub('', dtype.name))
            return out

    def __framet__(self, c, z, t):
        return self.transform_frame(self.__frame__(c, z, t), c, t)

    @property
    def axes(self):
        return self.shape.axes

    @axes.setter
    def axes(self, value):
        shape = self.shape[value]
        if isinstance(shape, Number):
            shape = (shape,)
        self._shape = Shape(shape, value)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    @cached_property
    def extrametadata(self):
        if isinstance(self.path, Path):
            if self.path.with_suffix('.pzl2').exists():
                pname = self.path.with_suffix('.pzl2')
            elif self.path.with_suffix('.pzl').exists():
                pname = self.path.with_suffix('.pzl')
            else:
                return
            try:
                return self.get_config(pname)
            except (Exception,):
                return
        return

    @property
    def ndim(self):
        return len(self.shape)

    @cached_property
    def piezoval(self):
        """ gives the height of the piezo and focus motor, only available when CylLensGUI was used """

        def upack(idx):
            time = list()
            val = list()
            if len(idx) == 0:
                return time, val
            for i in idx:
                time.append(int(re.search(r'\d+', n[i]).group(0)))
                val.append(w[i])
            return zip(*sorted(zip(time, val)))

        # Maybe the values are stored in the metadata
        n = self.metadata.search('LsmTag|Name')[0]
        w = self.metadata.search('LsmTag')[0]
        if n is not None:
            # n = self.metadata['LsmTag|Name'][1:-1].split(', ')
            # w = str2float(self.metadata['LsmTag'][1:-1].split(', '))

            pidx = np.where([re.search(r'^Piezo\s\d+$', x) is not None for x in n])[0]
            sidx = np.where([re.search(r'^Zstage\s\d+$', x) is not None for x in n])[0]

            ptime, pval = upack(pidx)
            stime, sval = upack(sidx)

        # Or maybe in an extra '.pzl' file
        else:
            m = self.extrametadata
            if m is not None and 'p' in m:
                q = np.array(m['p'])
                if not len(q.shape):
                    q = np.zeros((1, 3))

                ptime = [int(i) for i in q[:, 0]]
                pval = [float(i) for i in q[:, 1]]
                sval = [float(i) for i in q[:, 2]]

            else:
                ptime = []
                pval = []
                sval = []

        df = pandas.DataFrame(columns=['frame', 'piezoZ', 'stageZ'])
        df['frame'] = ptime
        df['piezoZ'] = pval
        df['stageZ'] = np.array(sval) - np.array(pval) - \
                       self.metadata.re_search(r'AcquisitionModeSetup\|ReferenceZ', 0)[0] * 1e6

        # remove duplicates
        df = df[~df.duplicated('frame', 'last')]
        return df

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if isinstance(value, Shape):
            self._shape = value
        else:
            self._shape = Shape((value['xyczt'.find(i.lower())] for i in self.axes), self.axes)

    @property
    def summary(self):
        """ gives a helpful summary of the recorded experiment """
        s = [f"path/filename: {self.path}",
             f"series/pos:    {self.series}"]
        if isinstance(self, View):
            s.append(f"reader:        {self.base.__class__.__module__.split('.')[-1]} view")
        else:
            s.append(f"reader:        {self.__class__.__module__.split('.')[-1]} base")
        s.extend((f"dtype:         {self.dtype}",
                  f"shape ({self.axes}):".ljust(15) + f"{' x '.join(str(i) for i in self.shape)}"))
        if self.pxsize_um:
            s.append(f'pixel size:    {1000 * self.pxsize_um:.2f} nm')
        if self.zstack and self.deltaz_um:
            s.append(f'z-interval:    {1000 * self.deltaz_um:.2f} nm')
        if self.exposuretime_s and not all(e is None for e in self.exposuretime_s):
            s.append(f'exposuretime:  {self.exposuretime_s[0]:.2f} s')
        if self.timeseries and self.timeinterval:
            s.append(f'time interval: {self.timeinterval:.3f} s')
        if self.binning:
            s.append('binning:       {}x{}'.format(*self.binning))
        if self.laserwavelengths:
            s.append('laser colors:  ' + ' | '.join([' & '.join(len(w) * ('{:.0f}',)).format(*w)
                                                     for w in self.laserwavelengths]) + ' nm')
        if self.laserpowers:
            s.append('laser powers:  ' + ' | '.join([' & '.join(len(p) * ('{:.3g}',)).format(*[100 * i for i in p])
                                                     for p in self.laserpowers]) + ' %')
        if self.objective:
            s.append('objective:     {}'.format(self.objective.model))
        if self.magnification:
            s.append('magnification: {}x'.format(self.magnification))
        if self.tubelens:
            s.append('tubelens:      {}'.format(self.tubelens.model))
        if self.filter:
            s.append('filterset:     {}'.format(self.filter))
        if self.powermode:
            s.append('powermode:     {}'.format(self.powermode))
        if self.collimator:
            s.append('collimator:   ' + (' {}' * len(self.collimator)).format(*self.collimator))
        if self.tirfangle:
            s.append('TIRF angle:   ' + (' {:.2f}°' * len(self.tirfangle)).format(*self.tirfangle))
        if self.gain:
            s.append('gain:         ' + (' {:.0f}' * len(self.gain)).format(*self.gain))
        if self.pcf:
            s.append('pcf:          ' + (' {:.2f}' * len(self.pcf)).format(*self.pcf))
        return '\n'.join(s)

    @property
    def T(self):
        return self.transpose()

    @cached_property
    def timeseries(self):
        return self.shape['t'] > 1

    @cached_property
    def zstack(self):
        return self.shape['z'] > 1

    @wraps(np.argmax)
    def argmax(self, *args, **kwargs):
        return self.__array_arg_fun__(np.argmax, *args, **kwargs)

    @wraps(np.argmin)
    def argmin(self, *args, **kwargs):
        return self.__array_arg_fun__(np.argmin, *args, **kwargs)

    @wraps(np.max)
    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.max], axis, None, out, keepdims, [initial], where)

    @wraps(np.mean)
    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **kwargs):
        dtype = dtype or float
        n = np.prod(self.shape) if axis is None else self.shape[axis]

        def sfun(frame):
            return np.asarray(frame).astype(float)

        def cfun(res):
            return res / n

        return self.__array_fun__([np.sum], axis, dtype, out, keepdims, None, where, [sfun], cfun)

    @wraps(np.min)
    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.min], axis, None, out, keepdims, [initial], where)

    @wraps(np.moveaxis)
    def moveaxis(self, source, destination):
        raise NotImplementedError('moveaxis is not implemented')

    @wraps(np.nanmax)
    def nanmax(self, axis=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.nanmax], axis, None, out, keepdims, [initial], where)

    @wraps(np.nanmean)
    def nanmean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **kwargs):
        dtype = dtype or float

        def sfun(frame):
            return np.asarray(frame).astype(float)

        def nfun(frame):
            return np.invert(np.isnan(frame))

        return self.__array_fun__([np.nansum, np.sum], axis, dtype, out, keepdims, None, where, (sfun, nfun), truediv)

    @wraps(np.nanmin)
    def nanmin(self, axis=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.nanmin], axis, None, out, keepdims, [initial], where)

    @wraps(np.nansum)
    def nansum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.nansum], axis, dtype, out, keepdims, [initial], where)

    @wraps(np.nanstd)
    def nanstd(self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=None):
        return self.nanvar(axis, dtype, out, ddof, keepdims, where=where, std=True)

    @wraps(np.nanvar)
    def nanvar(self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=True, std=False):
        dtype = dtype or float

        def sfun(frame):
            return np.asarray(frame).astype(float)

        def s2fun(frame):
            return np.asarray(frame).astype(float) ** 2

        def nfun(frame):
            return np.invert(np.isnan(frame))

        if std:
            def cfun(s, s2, n):
                return np.sqrt((s2 - s ** 2 / n) / (n - ddof))
        else:
            def cfun(s, s2, n):
                return (s2 - s ** 2 / n) / (n - ddof)
        return self.__array_fun__([np.nansum, np.nansum, np.sum], axis, dtype, out, keepdims, None, where,
                                  (sfun, s2fun, nfun), cfun)

    @wraps(np.reshape)
    def reshape(self, *args, **kwargs):
        return np.asarray(self).reshape(*args, **kwargs)

    @wraps(np.squeeze)
    def squeeze(self, axes=None):
        new = self.copy()
        if axes is None:
            axes = tuple(i for i, j in enumerate(new.shape) if j == 1)
        elif isinstance(axes, Number):
            axes = (axes,)
        else:
            axes = tuple(new.axes.find(ax) if isinstance(ax, str) else ax for ax in axes)
        if any([new.shape[ax] != 1 for ax in axes]):
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        new.axes = ''.join(j for i, j in enumerate(new.axes) if i not in axes)
        return new

    @wraps(np.std)
    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=True):
        return self.var(axis, dtype, out, ddof, keepdims, where=where, std=True)

    @wraps(np.sum)
    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, **kwargs):
        return self.__array_fun__([np.sum], axis, dtype, out, keepdims, [initial], where)

    @wraps(np.swapaxes)
    def swapaxes(self, axis1, axis2):
        new = self.copy()
        axes = new.axes
        if isinstance(axis1, str):
            axis1 = axes.find(axis1)
        if isinstance(axis2, str):
            axis2 = axes.find(axis2)
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        new.axes = axes
        return new

    @wraps(np.transpose)
    def transpose(self, axes=None):
        new = self.copy()
        if axes is None:
            new.axes = new.axes[::-1]
        else:
            new.axes = ''.join(ax if isinstance(ax, str) else new.axes[ax] for ax in axes)
        return new

    @wraps(np.var)
    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=True, std=False):
        dtype = dtype or float
        n = np.prod(self.shape) if axis is None else self.shape[axis]

        def sfun(frame):
            return np.asarray(frame).astype(float)

        def s2fun(frame):
            return np.asarray(frame).astype(float) ** 2

        if std:
            def cfun(s, s2):
                return np.sqrt((s2 - s ** 2 / n) / (n - ddof))
        else:
            def cfun(s, s2):
                return (s2 - s ** 2 / n) / (n - ddof)
        return self.__array_fun__([np.sum, np.sum], axis, dtype, out, keepdims, None, where, (sfun, s2fun), cfun)

    def asarray(self):
        return self.__array__()

    def astype(self, dtype, *args, **kwargs):
        new = self.copy()
        new.dtype = dtype
        return new

    def block(self, x=None, y=None, c=None, z=None, t=None):
        """ returns 5D block of frames """
        x, y, c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1)
                         for i, e in zip('xyczt', (x, y, c, z, t))]
        d = np.empty((len(x), len(y), len(c), len(z), len(t)), self.dtype)
        for (ci, cj), (zi, zj), (ti, tj) in product(enumerate(c), enumerate(z), enumerate(t)):
            d[:, :, ci, zi, ti] = self.frame(cj, zj, tj)[x][:, y]
        return d

    def copy(self):
        return View(self)

    def data(self, c=0, z=0, t=0):
        """ returns 3D stack of frames """
        c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1) for i, e in zip('czt', (c, z, t))]
        return np.dstack([self.frame(ci, zi, ti) for ci, zi, ti in product(c, z, t)])

    def frame(self, c=0, z=0, t=0):
        """ returns single 2D frame """
        c = self.get_channel(c)
        c %= self.base.shape['c']
        z %= self.base.shape['z']
        t %= self.base.shape['t']

        # cache last n (default 16) frames in memory for speed (~250x faster)
        key = (c, z, t, self.transform, self.frame_decorator)
        if key in self.cache:
            self.cache.move_to_end(key)
            f = self.cache[key]
        else:
            f = self.__framet__(c, z, t)
            if self.frame_decorator is not None:
                f = self.frame_decorator(self, f, c, z, t)
            self.cache[key] = f
        if self.dtype is not None:
            return f.copy().astype(self.dtype)
        else:
            return f.copy()

    def get_channel(self, channel_name):
        if not isinstance(channel_name, str):
            return channel_name
        else:
            c = [i for i, c in enumerate(self.cnamelist) if c.lower().startswith(channel_name.lower())]
            assert len(c) > 0, 'Channel {} not found in {}'.format(c, self.cnamelist)
            assert len(c) < 2, 'Channel {} not unique in {}'.format(c, self.cnamelist)
            return c[0]

    @staticmethod
    def get_config(file):
        """ Open a yml config file """
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            r'tag:yaml.org,2002:float',
            re.compile(r'''^(?:
                     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$''', re.X),
            list(r'-+0123456789.'))
        with open(file, 'r') as f:
            return yaml.load(f, loader)

    def get_czt(self, c, z, t):
        czt = []
        for i, n in zip('czt', (c, z, t)):
            if n is None:
                czt.append(list(range(self.shape[i])))
            elif isinstance(n, range):
                if n.stop < 0:
                    stop = n.stop % self.shape[i]
                elif n.stop > self.shape[i]:
                    stop = self.shape[i]
                else:
                    stop = n.stop
                czt.append(list(range(n.start % self.shape[i], stop, n.step)))
            elif isinstance(n, Number):
                czt.append([n % self.shape[i]])
            else:
                czt.append([k % self.shape[i] for k in n])
        return [self.get_channel(c) for c in czt[0]], *czt[1:]

    @staticmethod
    def get_ome(path):
        """ Use java BioFormats to make an ome metadata structure. """
        with multiprocessing.get_context('spawn').Pool(1) as pool:
            ome = pool.map(get_ome, (path,))[0]
            return ome

    def is_noise(self, volume=None):
        """ True if volume only has noise """
        if volume is None:
            volume = self
        fft = np.fft.fftn(volume)
        corr = np.fft.fftshift(np.fft.ifftn(fft * fft.conj()).real / np.sum(volume ** 2))
        return -np.log(1 - corr[tuple([0] * corr.ndim)]) > 5

    @staticmethod
    def kill_vm():
        JVM().kill_vm()

    def new(self, *args, **kwargs):
        warnings.warn('Imread.new has been deprecated, use Imread.view instead.', DeprecationWarning, 2)
        return self.view(*args, **kwargs)

    def save_as_tiff(self, fname=None, c=None, z=None, t=None, split=False, bar=True, pixel_type='uint16', **kwargs):
        """ saves the image as a tif file
            split: split channels into different files """
        if fname is None:
            fname = self.path.with_suffix('.tif')
            if fname == self.path:
                raise FileExistsError(f'File {fname} exists already.')
        if not isinstance(fname, Path):
            fname = Path(fname)
        if split:
            for i in range(self.shape['c']):
                if self.timeseries:
                    self.save_as_tiff(fname.with_name(f'{fname.stem}_C{i:01d}').with_suffix('.tif'), i, 0, None, False,
                                      bar, pixel_type)
                else:
                    self.save_as_tiff(fname.with_name(f'{fname.stem}_C{i:01d}').with_suffix('.tif'), i, None, 0, False,
                                      bar, pixel_type)
        else:
            n = [c, z, t]
            for i, ax in enumerate('czt'):
                if n[i] is None:
                    n[i] = range(self.shape[ax])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            shape = [len(i) for i in n]
            at_least_one = False
            with IJTiffFile(fname.with_suffix('.tif'), shape, pixel_type,
                            pxsize=self.pxsize_um, deltaz=self.deltaz_um, **kwargs) as tif:
                for i, m in tqdm(zip(product(*[range(s) for s in shape]), product(*n)),
                                 total=np.prod(shape), desc='Saving tiff', disable=not bar):
                    if np.any(self(*m)) or not at_least_one:
                        tif.save(self(*m), *i)
                        at_least_one = True

    def set_transform(self):
        # handle transforms
        if self.transform is False or self.transform is None:
            self.transform = None
        else:
            if isinstance(self.transform, Transforms):
                self.transform = self.transform
            else:
                if isinstance(self.transform, str):
                    self.transform = ImTransforms(self.path, self.cyllens, self.transform)
                else:
                    self.transform = ImTransforms(self.path, self.cyllens, self.beadfile)
                if self.drift is True:
                    self.transform = ImShiftTransforms(self)
                elif not (self.drift is False or self.drift is None):
                    self.transform = ImShiftTransforms(self, self.drift)
            self.transform.adapt(self.frameoffset, self.shape.xyczt)
            self.beadfile = self.transform.files

    @staticmethod
    def split_path_series(path):
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, Path) and path.name.startswith('Pos'):
            return path.parent, int(path.name.lstrip('Pos'))
        return path, 0

    def transform_frame(self, frame, c, t=0):
        if self.transform is None:
            return frame
        else:
            return self.transform[(self.cnamelist[c],)].frame(frame)

    def view(self, *args, **kwargs):
        return View(self, *args, **kwargs)


class View(Imread):
    def __init__(self, base, dtype=None, transform=None, drift=None, beadfile=None):
        super().__init__(base.base, base.slice, base.shape, dtype or base.dtype, transform or base.transform,
                         drift or base.drift, beadfile or base.beadfile, base.frame_decorator)
        self.set_transform()

    def __getattr__(self, item):
        if not hasattr(self.base, item):
            raise AttributeError(f'{self.__class__} object has no attribute {item}')
        return self.base.__getattribute__(item)


class AbstractReader(Imread, metaclass=ABCMeta):
    """ class to read image files, while taking good care of important metadata,
            currently optimized for .czi files, but can open anything that bioformats can handle
        path: path to the image file
        optional:
        series: in case multiple experiments are saved in one file, like in .lif files
        transform: automatically correct warping between channels, need transforms.py among others
        drift: automatically correct for drift, only works if transform is not None or False
        beadfile: image file(s) with beads which can be used for correcting warp
        dtype: datatype to be used when returning frames
        meta: define metadata, used for pickle-ing

        NOTE: run imread.kill_vm() at the end of your script/program, otherwise python might not terminate

        modify images on the fly with a decorator function:
            define a function which takes an instance of this object, one image frame,
            and the coordinates c, z, t as arguments, and one image frame as return
            >> imread.frame_decorator = fun
            then use imread as usually

        Examples:
            >> im = imread('/DATA/lenstra_lab/w.pomp/data/20190913/01_YTL639_JF646_DefiniteFocus.czi')
            >> im
             << shows summary
            >> im.shape
             << (256, 256, 2, 1, 600)
            >> plt.imshow(im(1, 0, 100))
             << plots frame at position c=1, z=0, t=100 (python type indexing), note: round brackets; always 2d array
                with 1 frame
            >> data = im[:,:,0,0,:25]
             << retrieves 5d numpy array containing first 25 frames at c=0, z=0, note: square brackets; always 5d array
            >> plt.imshow(im.max(0, None, 0))
             << plots max-z projection at c=0, t=0
            >> len(im)
             << total number of frames
            >> im.pxsize
             << 0.09708737864077668 image-plane pixel size in um
            >> im.laserwavelengths
             << [642, 488]
            >> im.laserpowers
             << [0.02, 0.0005] in %

            See __init__ and other functions for more ideas.

        Subclassing:
            Subclass this class to add more file types. A subclass should always have at least the following methods:
                staticmethod _can_open(path): returns True when the subclass can open the image in path
                __metadata__(self): pulls some metadata from the file and do other format specific things, it needs to
                                    define a few properties, like shape, etc.
                __frame__(self, c, z, t): this should return a single frame at channel c, slice z and time t
                optional close(self): close the file in a proper way
                optional field priority: subclasses with lower priority will be tried first, default = 99
                Any other method can be overridden as needed
        wp@tl2019-2023 """

    priority = 99
    do_not_pickle = 'cache'
    ureg = ureg

    @staticmethod
    @abstractmethod
    def _can_open(path):  # Override this method, and return true when the subclass can open the file
        return False

    @abstractmethod
    def __frame__(self, c, z, t):  # Override this, return the frame at c, z, t
        return np.random.randint(0, 255, self.shape['xy'])

    @cached_property
    def ome(self):
        return self.get_ome(self.path)

    def open(self):  # Optionally override this, open file handles etc.
        """ filehandles cannot be pickled and should be marked such by setting do_not_pickle = 'file_handle_name' """
        return

    def close(self):  # Optionally override this, close file handles etc.
        return

    def __init__(self, path, transform=False, drift=False, beadfile=None, dtype=None, axes=None):
        if isinstance(path, Imread):
            return
        super().__init__(self, transform=transform, drift=drift, beadfile=beadfile)
        self.isclosed = False
        if isinstance(path, str):
            path = Path(path)
        self.path, self.series = self.split_path_series(path)
        if isinstance(path, Path):
            self.title = self.path.name
            self.acquisitiondate = datetime.fromtimestamp(self.path.stat().st_mtime).strftime('%y-%m-%dT%H:%M:%S')
        else:  # ndarray
            self.title = 'ndarray'
            self.acquisitiondate = 'now'

        self.reader = None
        self.pcf = None
        self.powermode = None
        self.collimator = None
        self.tirfangle = None
        self.cyllens = ['None', 'None']
        self.duolink = 'None'
        self.detector = [0, 1]
        self.track = [0]
        self.cache = DequeDict(16)
        self.frameoffset = 0, 0  # how far apart the centers of frame and sensor are

        self.open()
        # extract some metadata from ome
        instrument = self.ome.instruments[0] if self.ome.instruments else None
        image = self.ome.images[0]
        pixels = image.pixels
        self.shape = pixels.size_x, pixels.size_y, pixels.size_c, pixels.size_z, pixels.size_t
        self.dtype = pixels.type.value if dtype is None else dtype
        self.pxsize = pixels.physical_size_x_quantity
        try:
            self.exposuretime = tuple(find(image.pixels.planes, the_c=c).exposure_time_quantity
                                      for c in range(self.shape['c']))
        except AttributeError:
            self.exposuretime = ()

        if self.zstack:
            self.deltaz = image.pixels.physical_size_z_quantity
            self.deltaz_um = None if self.deltaz is None else self.deltaz.to(self.ureg.um).m
        else:
            self.deltaz = self.deltaz_um = None
        if self.ome.images[0].objective_settings:
            self.objective = find(instrument.objectives, id=self.ome.images[0].objective_settings.id)
        else:
            self.objective = None
        try:
            t0 = find(image.pixels.planes, the_c=0, the_t=0, the_z=0).delta_t
            t1 = find(image.pixels.planes, the_c=0, the_t=self.shape['t'] - 1, the_z=0).delta_t
            self.timeinterval = (t1 - t0) / (self.shape['t'] - 1) if self.shape['t'] > 1 else None
        except AttributeError:
            self.timeinterval = None
        try:
            self.binning = [int(i) for i in image.pixels.channels[0].detector_settings.binning.value.split('x')]
            self.pxsize *= self.binning[0]
        except (AttributeError, IndexError, ValueError):
            self.binning = None
        self.cnamelist = [channel.name for channel in image.pixels.channels]
        try:
            optovars = [objective for objective in instrument.objectives if 'tubelens' in objective.id.lower()]
        except AttributeError:
            optovars = []
        if len(optovars) == 0:
            self.tubelens = None
        else:
            self.tubelens = optovars[0]
        if self.objective:
            if self.tubelens:
                self.magnification = self.objective.nominal_magnification * self.tubelens.nominal_magnification
            else:
                self.magnification = self.objective.nominal_magnification
            self.NA = self.objective.lens_na
        else:
            self.magnification = None
            self.NA = None

        self.gain = [find(instrument.detectors, id=channel.detector_settings.id).amplification_gain
                     for channel in image.pixels.channels
                     if channel.detector_settings
                     and find(instrument.detectors, id=channel.detector_settings.id).amplification_gain]
        self.laserwavelengths = [(channel.excitation_wavelength_quantity.to(self.ureg.nm).m,)
                                 for channel in pixels.channels if channel.excitation_wavelength_quantity]
        self.laserpowers = try_default(lambda: [(1 - channel.light_source_settings.attenuation,)
                                                for channel in pixels.channels], [])
        self.filter = try_default(lambda: [find(instrument.filter_sets, id=channel.filter_set_ref.id).model
                                           for channel in image.pixels.channels], None)
        self.pxsize_um = None if self.pxsize is None else self.pxsize.to(self.ureg.um).m
        self.exposuretime_s = [None if i is None else i.to(self.ureg.s).m for i in self.exposuretime]

        if axes is None:
            self.axes = ''.join(i for i in 'cztxy' if self.shape[i] > 1)
        elif axes.lower() == 'full':
            self.axes = 'cztxy'
        else:
            self.axes = axes
        self.slice = [np.arange(s, dtype=int) for s in self.shape.xyczt]

        m = self.extrametadata
        if m is not None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                if 'FeedbackChannels' in m:
                    self.feedback = m['FeedbackChannels']
                else:
                    self.feedback = m['FeedbackChannel']
            except (Exception,):
                self.cyllens = ['None', 'None']
                self.duolink = 'None'
                self.feedback = []
        try:
            self.cyllenschannels = np.where([self.cyllens[self.detector[c]].lower() != 'none'
                                             for c in range(self.shape['c'])])[0].tolist()
        except (Exception,):
            pass
        self.set_transform()
        try:
            s = int(re.findall(r'_(\d{3})_', self.duolink)[0]) * ureg.nm
        except (Exception,):
            s = 561 * ureg.nm
        try:
            sigma = []
            for c, d in enumerate(self.detector):
                emission = (np.hstack(self.laserwavelengths[c]) + 22) * ureg.nm
                sigma.append([emission[emission > s].max(initial=0), emission[emission < s].max(initial=0)][d])
            sigma = np.hstack(sigma)
            sigma[sigma == 0] = 600 * ureg.nm
            sigma /= 2 * self.NA * self.pxsize
            self.sigma = sigma.magnitude.tolist()
        except (Exception,):
            self.sigma = [2] * self.shape['c']
        if not self.NA:
            self.immersionN = 1
        elif 1.5 < self.NA:
            self.immersionN = 1.661
        elif 1.3 < self.NA < 1.5:
            self.immersionN = 1.518
        elif 1 < self.NA < 1.3:
            self.immersionN = 1.33
        else:
            self.immersionN = 1

        p = re.compile(r'(\d+):(\d+)$')
        try:
            self.track, self.detector = zip(*[[int(i) for i in p.findall(find(
                self.ome.images[0].pixels.channels, id=f'Channel:{c}').detector_settings.id)[0]]
                                              for c in range(self.shape['c'])])
        except Exception:
            pass

def main():
    parser = ArgumentParser(description='Display info and save as tif')
    parser.add_argument('file', help='image_file')
    parser.add_argument('out', help='path to tif out', type=str, default=None, nargs='?')
    parser.add_argument('-r', '--register', help='register channels', action='store_true')
    parser.add_argument('-c', '--channel', help='channel', type=int, default=None)
    parser.add_argument('-z', '--zslice', help='z-slice', type=int, default=None)
    parser.add_argument('-t', '--time', help='time', type=int, default=None)
    parser.add_argument('-s', '--split', help='split channels', action='store_true')
    parser.add_argument('-f', '--force', help='force overwrite', action='store_true')
    args = parser.parse_args()

    with Imread(args.file, transform=args.register) as im:
        print(im.summary)
        if args.out:
            out = Path(args.out).absolute()
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not args.force:
                print('File {} exists already, add the -f flag if you want to overwrite it.'.format(args.out))
            else:
                im.save_as_tiff(out, args.channel, args.zslice, args.time, args.split)


from .readers import *
