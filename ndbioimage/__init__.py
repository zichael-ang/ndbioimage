from __future__ import annotations

import multiprocessing
import os
import re
import warnings
from abc import ABC, ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from functools import cached_property, wraps
from importlib.metadata import version
from itertools import product
from operator import truediv
from pathlib import Path
from traceback import print_exc
from typing import Any, Callable, Generator, Iterable, Optional, Sequence, TypeVar

import numpy as np
import yaml
from numpy.typing import ArrayLike, DTypeLike
from ome_types import OME, from_xml, model, ureg
from pint import set_application_registry
from tiffwrite import FrameInfo, IJTiffParallel
from tqdm.auto import tqdm

from .jvm import JVM, JVMException
from .transforms import Transform, Transforms

try:
    __version__ = version(Path(__file__).parent.name)
except Exception:  # noqa
    __version__ = 'unknown'

try:
    with open(Path(__file__).parent.parent / '.git' / 'HEAD') as g:
        head = g.read().split(':')[1].strip()
    with open(Path(__file__).parent.parent / '.git' / head) as h:
        __git_commit_hash__ = h.read().rstrip('\n')
except Exception:  # noqa
    __git_commit_hash__ = 'unknown'

ureg.default_format = '~P'
set_application_registry(ureg)
warnings.filterwarnings('ignore', 'Reference to unknown ID')
Number = int | float | np.integer | np.floating


class ReaderNotFoundError(Exception):
    pass


class TransformTiff(IJTiffParallel):
    """ transform frames in a parallel process to speed up saving """
    def __init__(self, image: Imread, *args: Any, **kwargs: Any) -> None:
        self.image = image
        super().__init__(*args, **kwargs)

    def parallel(self, frame: tuple[int, int, int]) -> tuple[FrameInfo]:
        return (np.asarray(self.image(*frame)), 0, 0, 0),


class DequeDict(OrderedDict):
    def __init__(self, maxlen: int = None, *args: Any, **kwargs: Any) -> None:
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        super().__setitem__(*args, **kwargs)
        self.truncate()

    def truncate(self) -> None:
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(False)

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)  # type: ignore
        self.truncate()


def find(obj: Sequence[Any], **kwargs: Any) -> Any:
    for item in obj:
        try:
            if all([getattr(item, key) == value for key, value in kwargs.items()]):
                return item
        except AttributeError:
            pass


R = TypeVar('R')


def try_default(fun: Callable[..., R], default: Any, *args: Any, **kwargs: Any) -> R:
    try:
        return fun(*args, **kwargs)
    except Exception:  # noqa
        return default


def bioformats_ome(path: str | Path) -> OME:
    from .readers.bfread import jars
    try:
        jvm = JVM(jars)  # noqa
        ome_meta = jvm.metadata_tools.createOMEXMLMetadata()
        reader = jvm.image_reader()
        reader.setMetadataStore(ome_meta)
        reader.setId(str(path))
        ome = from_xml(str(ome_meta.dumpXML()), parser='lxml')
    except Exception:  # noqa
        print_exc()
        ome = model.OME()
    finally:
        jvm.kill_vm()  # noqa
    return ome


class Shape(tuple):
    def __new__(cls, shape: Sequence[int] | Shape, axes: str = 'yxczt') -> Shape:
        if isinstance(shape, Shape):
            axes = shape.axes  # type: ignore
        new = super().__new__(cls, shape)
        new.axes = axes.lower()
        return new  # type: ignore

    def __getitem__(self, n: int | str) -> int | tuple[int]:
        if isinstance(n, str):
            if len(n) == 1:
                return self[self.axes.find(n.lower())] if n.lower() in self.axes else 1
            else:
                return tuple(self[i] for i in n)  # type: ignore
        return super().__getitem__(n)

    @cached_property
    def yxczt(self) -> tuple[int, int, int, int, int]:
        return tuple(self[i] for i in 'yxczt')  # type: ignore


class OmeCache(DequeDict):
    """ prevent (potentially expensive) rereading of ome data by caching """

    instance = None

    def __new__(cls) -> OmeCache:
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        super().__init__(64)

    def __reduce__(self) -> tuple[type, tuple]:
        return self.__class__, ()

    def __getitem__(self, path: Path | str | tuple) -> OME:
        if isinstance(path, tuple):
            return super().__getitem__(path)
        else:
            return super().__getitem__(self.path_and_lstat(path))

    def __setitem__(self, path: Path | str | tuple, value: OME) -> None:
        if isinstance(path, tuple):
            super().__setitem__(path, value)
        else:
            super().__setitem__(self.path_and_lstat(path), value)

    def __contains__(self, path: Path | str | tuple) -> bool:
        if isinstance(path, tuple):
            return super().__contains__(path)
        else:
            return super().__contains__(self.path_and_lstat(path))

    @staticmethod
    def path_and_lstat(path: str | Path) -> tuple[Path, Optional[os.stat_result], Optional[os.stat_result]]:
        path = Path(path)
        return (path, (path.lstat() if path.exists() else None),
                (path.with_suffix('.ome.xml').lstat() if path.with_suffix('.ome.xml').exists() else None))


def get_positions(path: str | Path) -> Optional[list[int]]:
    subclass = AbstractReader.get_subclass(path)
    return subclass.get_positions(AbstractReader.split_path_series(path)[0])


class Imread(np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """ class to read image files, while taking good care of important metadata,
        currently optimized for .czi files, but can open anything that bioformats can handle
            path: path to the image file
            optional:
            axes: order of axes, default: cztyx, but omitting any axes with lenght 1
            dtype: datatype to be used when returning frames

            modify images on the fly with a decorator function:
                define a function which takes an instance of this object, one image frame,
                and the coordinates c, z, t as arguments, and one image frame as return
                >> imread.frame_decorator = fun
                then use imread as usually

            Examples:
                >> im = Imread('/path/to/file.image', axes='czt)
                >> im
                 << shows summary
                >> im.shape
                 << (15, 26, 1000, 1000)
                >> im.axes
                 << 'ztyx'
                >> plt.imshow(im[1, 0])
                 << plots frame at position z=1, t=0 (python type indexing)
                >> plt.imshow(im[:, 0].max('z'))
                 << plots max-z projection at t=0
                >> im.pxsize_um
                 << 0.09708737864077668 image-plane pixel size in um
                >> im.laserwavelengths
                 << [642, 488]
                >> im.laserpowers
                 << [0.02, 0.0005] in %

                See __init__ and other functions for more ideas.

            Subclassing:
                Subclass AbstractReader to add more file types. A subclass should always have at least the following
                methods:
                    staticmethod _can_open(path): returns True when the subclass can open the image in path
                    __frame__(self, c, z, t): this should return a single frame at channel c, slice z and time t
                    optional open(self): code to be run during initialization, e.g. to open a file handle
                    optional close(self): close the file in a proper way
                    optional class field priority: subclasses with lower priority will be tried first, default = 99
                    optional get_ome(self) -> OME: return an OME structure with metadata,
                        if not present bioformats will be used to generate an OME
                    Any other method can be overridden as needed
    """

    isclosed: Optional[bool]
    channel_names: Optional[list[str]]
    series: Optional[int]
    pxsize_um: Optional[float]
    deltaz_um: Optional[float]
    exposuretime_s: Optional[list[float]]
    timeinterval: Optional[float]
    binning: Optional[list[int]]
    laserwavelengths: Optional[list[tuple[float]]]
    laserpowers: Optional[list[tuple[float]]]
    objective: Optional[model.Objective]
    magnification: Optional[float]
    tubelens: Optional[model.Objective]
    filter: Optional[str]
    powermode: Optional[str]
    collimator: Optional[str]
    tirfangle: Optional[list[float]]
    gain: Optional[list[float]]
    pcf: Optional[list[float]]
    path: Path
    __frame__: Callable[[int, int, int], np.ndarray]

    @staticmethod
    def get_subclass(path: Path | str | Any):
        if len(AbstractReader.__subclasses__()) == 0:
            raise Exception('Restart python kernel please!')
        path, _ = AbstractReader.split_path_series(path)
        for subclass in sorted(AbstractReader.__subclasses__(), key=lambda subclass_: subclass_.priority):
            if subclass._can_open(path):  # noqa
                do_not_pickle = (AbstractReader.do_not_pickle,) if isinstance(AbstractReader.do_not_pickle, str) \
                    else AbstractReader.do_not_pickle
                subclass_do_not_pickle = (subclass.do_not_pickle,) if isinstance(subclass.do_not_pickle, str) \
                    else subclass.do_not_pickle if hasattr(subclass, 'do_not_pickle') else ()
                subclass.do_not_pickle = set(do_not_pickle).union(set(subclass_do_not_pickle))
                return subclass
        raise ReaderNotFoundError(f'No reader found for {path}.')


    def __new__(cls, path: Path | str | Imread | Any = None, dtype: DTypeLike = None, axes: str = None) -> Imread:
        if cls is not Imread:
            return super().__new__(cls)
        if isinstance(path, Imread):
            return path
        subclass = cls.get_subclass(path)
        do_not_pickle = (AbstractReader.do_not_pickle,) if isinstance(AbstractReader.do_not_pickle, str) \
            else AbstractReader.do_not_pickle
        subclass_do_not_pickle = (subclass.do_not_pickle,) if isinstance(subclass.do_not_pickle, str) \
            else subclass.do_not_pickle if hasattr(subclass, 'do_not_pickle') else ()
        subclass.do_not_pickle = set(do_not_pickle).union(set(subclass_do_not_pickle))
        return super().__new__(subclass)

    def __init__(self, *args: Any, **kwargs: Any):
        def parse(base: Imread = None,  # noqa
                  slice: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,  # noqa
                  shape: tuple[int, ...] = (0, 0, 0, 0, 0),  # noqa
                  dtype: DTypeLike = None,  # noqa
                  frame_decorator: Callable[[Imread, np.ndarray, int, int, int], np.ndarray] = None  # noqa
                  ) -> tuple[Any, ...]:
            return base, slice, shape, dtype, frame_decorator

        base, slice, shape, dtype, frame_decorator = parse(*args, **kwargs)  # noqa
        self.base = base or self
        self.slice = slice
        self._shape = Shape(shape)
        self.dtype = dtype
        self.frame_decorator = frame_decorator
        self.transform = Transforms()
        self.flags = dict(C_CONTIGUOUS=False, F_CONTIGUOUS=False, OWNDATA=False, WRITEABLE=False,
                          ALIGNED=False, WRITEBACKIFCOPY=False, UPDATEIFCOPY=False)

    def __call__(self, c: int = None, z: int = None, t: int = None, x: int = None, y: int = None) -> np.ndarray:
        """ same as im[] but allowing keyword axes, but slices need to made with slice() or np.s_ """
        return self[{k: slice(v) if v is None else v for k, v in dict(c=c, z=z, t=t, x=x, y=y).items()}]

    def __copy__(self) -> Imread:
        return self.copy()

    def __contains__(self, item: Number) -> bool:
        def unique_yield(a: Iterable[Any], b: Iterable[Any]) -> Generator[Any, None, None]:
            for k in a:
                yield k
            for k in b:
                if k not in a:
                    yield k

        for idx in unique_yield([key[:3] for key in self.cache.keys()],
                                product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t']))):
            yxczt = (slice(None), slice(None)) + idx
            in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
            if item in np.asarray(self[in_idx]):
                return True
        return False

    def __enter__(self) -> Imread:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if not self.isclosed:
            self.isclosed = True
            if hasattr(self, 'close'):
                self.close()

    def __getitem__(self, n: int | Sequence[int] | Sequence[slice] | slice | type(Ellipsis) |
                    dict[str, int | Sequence[int] | Sequence[slice] | slice | type(Ellipsis)]
                    ) -> Number | Imread | np.ndarray:
        """ slice like a numpy array but return an Imread instance """
        if self.isclosed:
            raise OSError('file is closed')
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

        axes_idx = [self.shape.axes.find(i) for i in 'yxczt']
        n = [n[j] if 0 <= j < len(n) else None for j in axes_idx]  # reorder n

        new_slice = []
        for s, e in zip(self.slice, n):
            if e is None:
                new_slice.append(s)
            else:
                new_slice.append(s[e])

        # TODO: check output dimensionality when requested shape in some dimension is 1
        if all([isinstance(s, Number) or a < 0 and s.size == 1 for s, a in zip(new_slice, axes_idx)]):
            return self.block(*new_slice).item()
        else:
            new = View(self)
            new.slice = new_slice
            new._shape = Shape([1 if isinstance(s, Number) else len(s) for s in new_slice])
            new.axes = ''.join(j for j in self.axes if j in [i for i, s in zip('yxczt', new_slice)
                                                             if not isinstance(s, Number)])
            return new

    def __getstate__(self) -> dict[str: Any]:
        return ({key: value for key, value in self.__dict__.items() if key not in self.do_not_pickle} |
                {'cache_size': self.cache.maxlen})

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return self.summary

    def __setstate__(self, state: dict[str, Any]) -> None:
        """ What happens during unpickling """
        self.__dict__.update({key: value for key, value in state.items() if key != 'cache_size'})
        if isinstance(self, AbstractReader):
            self.open()
        self.cache = DequeDict(state.get('cache_size', 16))

    def __str__(self) -> str:
        return str(self.path)

    # @property
    # def __array_interface__(self) -> dict[str, Any]:
    #     return dict(shape=tuple(self.shape), typestr=self.dtype.str, version=3, data=self.tobytes())

    def tobytes(self) -> bytes:
        return self.flatten().tobytes()

    def __array__(self, dtype: DTypeLike = None, copy: bool = None) -> np.ndarray:
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        block = self.block(*self.slice)
        axes_idx = [self.shape.axes.find(i) for i in 'yxczt']
        axes_squeeze = tuple({i for i, j in enumerate(axes_idx) if j == -1}.union(
            {i for i, j in enumerate(self.slice) if isinstance(j, Number)}))
        block = block.squeeze(axes_squeeze)
        if dtype is not None:
            block = block.astype(dtype)
        if block.ndim == 0:
            return block.item()
        axes = ''.join(j for i, j in enumerate('yxczt') if i not in axes_squeeze)
        return block.transpose([axes.find(i) for i in self.shape.axes if i in axes])

    def __array_arg_fun__(self, fun: Callable[[ArrayLike, Optional[int]], Number | np.ndarray],
                          axis: int | str = None, out: np.ndarray = None) -> Number | np.ndarray:
        """ frame-wise application of np.argmin and np.argmax """
        if axis is None:
            value = arg = None
            for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                yxczt = (slice(None), slice(None)) + idx
                in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
                new = np.asarray(self[in_idx])
                new_arg = np.unravel_index(fun(new), new.shape)  # type: ignore
                new_value = new[new_arg]
                if value is None:
                    arg = new_arg + idx
                    value = new_value
                else:
                    i = fun((value, new_value))  # type: ignore
                    arg = (arg, new_arg + idx)[i]
                    value = (value, new_value)[i]
            axes = ''.join(i for i in self.axes if i in 'yx') + 'czt'
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
            if axis_str in 'yx':
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    yxczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(yxczt['yxczt'.find(i)] for i in out_axes)
                    in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
                    new = self[in_idx]
                    out[out_idx] = fun(np.asarray(new), new.axes.find(axis_str))
            else:
                value = np.zeros(out.shape, self.dtype)
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    yxczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(yxczt['yxczt'.find(i)] for i in out_axes)
                    in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
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

    def __array_fun__(self, funs: Sequence[Callable[[ArrayLike], Number | np.ndarray]], axis: int | str = None,
                      dtype: DTypeLike = None, out: np.ndarray = None, keepdims: bool = False,
                      initials: list[Number | np.ndarray] = None, where: bool | int | np.ndarray = True,
                      ffuns: Sequence[Callable[[ArrayLike], np.ndarray]] = None,
                      cfun: Callable[..., np.ndarray] = None) -> Number | np.ndarray:
        """ frame-wise application of np.min, np.max, np.sum, np.mean and their nan equivalents """
        p = re.compile(r'\d')
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        if initials is None:
            initials = [None for _ in funs]
        if ffuns is None:
            ffuns = [None for _ in funs]

        def ffun_(frame: ArrayLike) -> np.ndarray:
            return np.asarray(frame)
        ffuns = [ffun_ if ffun is None else ffun for ffun in ffuns]
        if cfun is None:
            def cfun(*res):  # noqa
                return res[0]

        # TODO: smarter transforms
        if axis is None:
            for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                yxczt = (slice(None), slice(None)) + idx
                in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
                w = where if where is None or isinstance(where, bool) else where[in_idx]
                initials = [fun(np.asarray(ffun(self[in_idx])), initial=initial, where=w)  # type: ignore
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
            if axis_str in 'yx':
                yx = 'yx' if self.axes.find('x') > self.axes.find('y') else 'yx'
                frame_ax = yx.find(axis_str)
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    yxczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(yxczt['yxczt'.find(i)] for i in out_axes)
                    in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)
                    w = where if where is None or isinstance(where, bool) else where[in_idx]
                    res = cfun(*[fun(ffun(self[in_idx]), frame_ax, initial=initial, where=w)  # type: ignore
                                 for fun, ffun, initial in zip(funs, ffuns, initials)])
                    out[out_idx] = (np.round(res) if out.dtype.kind in 'ui' else res).astype(p.sub('', dtype.name))
            else:
                tmps = [np.zeros(out_shape) for _ in ffuns]
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    yxczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(yxczt['yxczt'.find(i)] for i in out_axes)
                    in_idx = tuple(yxczt['yxczt'.find(i)] for i in self.axes)

                    if idx['czt'.find(axis_str)] == 0:
                        w = where if where is None or isinstance(where, bool) else (where[in_idx],)
                        for tmp, fun, ffun, initial in zip(tmps, funs, ffuns, initials):
                            tmp[out_idx] = fun((ffun(self[in_idx]),), 0, initial=initial, where=w)  # type: ignore
                    else:
                        w = where if where is None or isinstance(where, bool) else \
                            (np.ones_like(where[in_idx]), where[in_idx])
                        for tmp, fun, ffun in zip(tmps, funs, ffuns):
                            tmp[out_idx] = fun((tmp[out_idx], ffun(self[in_idx])), 0, where=w)  # type: ignore
                out[...] = (np.round(cfun(*tmps)) if out.dtype.kind in 'ui' else
                            cfun(*tmps)).astype(p.sub('', dtype.name))
            return out

    @property
    def axes(self) -> str:
        return self.shape.axes

    @axes.setter
    def axes(self, value: str) -> None:
        shape = self.shape[value]
        if isinstance(shape, Number):
            shape = (shape,)
        self._shape = Shape(shape, value)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, value: DTypeLike) -> None:
        self._dtype = np.dtype(value)

    @cached_property
    def extrametadata(self) -> Optional[Any]:
        if isinstance(self.path, Path):
            if self.path.with_suffix('.pzl2').exists():
                pname = self.path.with_suffix('.pzl2')
            elif self.path.with_suffix('.pzl').exists():
                pname = self.path.with_suffix('.pzl')
            else:
                return
            try:
                return self.get_config(pname)
            except Exception:  # noqa
                return
        return

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)  # type: ignore

    @property
    def shape(self) -> Shape:
        return self._shape

    @shape.setter
    def shape(self, value: Shape | tuple[int, ...]) -> None:
        if isinstance(value, Shape):
            self._shape = value
        else:
            self._shape = Shape([value['yxczt'.find(i.lower())] for i in self.axes], self.axes)

    @property
    def summary(self) -> str:
        """ gives a helpful summary of the recorded experiment """
        s = [f'path/filename: {self.path}',
             f'series/pos:    {self.series}',
             f"reader:        {self.base.__class__.__module__.split('.')[-1]}"]
        s.extend((f'dtype:         {self.dtype}',
                  f'shape ({self.axes}):'.ljust(15) + f"{' x '.join(str(i) for i in self.shape)}"))
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
        if self.objective and self.objective.model:
            s.append(f'objective:     {self.objective.model}')
        if self.magnification:
            s.append(f'magnification: {self.magnification}x')
        if self.tubelens and self.tubelens.model:
            s.append(f'tubelens:      {self.tubelens.model}')
        if self.filter:
            s.append(f'filterset:     {self.filter}')
        if self.powermode:
            s.append(f'powermode:     {self.powermode}')
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
    def T(self) -> Imread:  # noqa
        return self.transpose()  # type: ignore

    @cached_property
    def timeseries(self) -> bool:
        return self.shape['t'] > 1

    @cached_property
    def zstack(self) -> bool:
        return self.shape['z'] > 1

    @wraps(np.ndarray.argmax)
    def argmax(self, *args, **kwargs):
        return self.__array_arg_fun__(np.argmax, *args, **kwargs)

    @wraps(np.ndarray.argmin)
    def argmin(self, *args, **kwargs):
        return self.__array_arg_fun__(np.argmin, *args, **kwargs)

    @wraps(np.ndarray.max)
    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True, **_):
        return self.__array_fun__([np.max], axis, None, out, keepdims, [initial], where)

    @wraps(np.ndarray.mean)
    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **_):
        dtype = dtype or float
        n = np.prod(self.shape) if axis is None else self.shape[axis]

        def sfun(frame: ArrayLike) -> np.ndarray:
            return np.asarray(frame).astype(float)

        def cfun(res: np.ndarray) -> np.ndarray:
            return res / n

        return self.__array_fun__([np.sum], axis, dtype, out, keepdims, None, where, [sfun], cfun)

    @wraps(np.ndarray.min)
    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True, **_):
        return self.__array_fun__([np.min], axis, None, out, keepdims, [initial], where)

    @wraps(np.moveaxis)
    def moveaxis(self, source, destination):
        raise NotImplementedError('moveaxis is not implemented')

    @wraps(np.nanmax)
    def nanmax(self, axis=None, out=None, keepdims=False, initial=None, where=True, **_):
        return self.__array_fun__([np.nanmax], axis, None, out, keepdims, [initial], where)

    @wraps(np.nanmean)
    def nanmean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **_):
        dtype = dtype or float

        def sfun(frame):
            return np.asarray(frame).astype(float)

        def nfun(frame):
            return np.invert(np.isnan(frame))

        return self.__array_fun__([np.nansum, np.sum], axis, dtype, out, keepdims, None, where, (sfun, nfun), truediv)

    @wraps(np.nanmin)
    def nanmin(self, axis=None, out=None, keepdims=False, initial=None, where=True, **_):
        return self.__array_fun__([np.nanmin], axis, None, out, keepdims, [initial], where)

    @wraps(np.nansum)
    def nansum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, **_):
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

    @wraps(np.ndarray.flatten)
    def flatten(self, *args, **kwargs):
        return np.asarray(self).flatten(*args, **kwargs)

    @wraps(np.ndarray.reshape)
    def reshape(self, *args, **kwargs):
        return np.asarray(self).reshape(*args, **kwargs)

    @wraps(np.ndarray.squeeze)
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

    @wraps(np.ndarray.std)
    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=True):
        return self.var(axis, dtype, out, ddof, keepdims, where=where, std=True)  # type: ignore

    @wraps(np.ndarray.sum)
    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, **_):
        return self.__array_fun__([np.sum], axis, dtype, out, keepdims, [initial], where)

    @wraps(np.ndarray.swapaxes)
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

    @wraps(np.ndarray.transpose)
    def transpose(self, *axes):
        new = self.copy()
        if not axes:
            new.axes = new.axes[::-1]
        else:
            new.axes = ''.join(ax if isinstance(ax, str) else new.axes[ax] for ax in axes)
        return new

    @wraps(np.ndarray.var)
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

    def asarray(self) -> np.ndarray:
        return self.__array__()

    @wraps(np.ndarray.astype)
    def astype(self, dtype, *_, **__):
        new = self.copy()
        new.dtype = dtype
        return new

    def block(self, y: int | Sequence[int] = None, x: int | Sequence[int] = None,
              c: int | Sequence[int] = None, z: int | Sequence[int] = None,
              t: int | Sequence[int] = None) -> np.ndarray:
        """ returns 5D block of frames """
        y, x, c, z, t = (np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1)
                         for i, e in zip('yxczt', (y, x, c, z, t)))
        d = np.empty((len(y), len(x), len(c), len(z), len(t)), self.dtype)
        for (ci, cj), (zi, zj), (ti, tj) in product(enumerate(c), enumerate(z), enumerate(t)):
            d[:, :, ci, zi, ti] = self.frame(cj, zj, tj)[y][:, x]  # type: ignore
        return d

    def copy(self) -> View:
        return View(self)

    def data(self, c: int | Sequence[int] = 0, z: int | Sequence[int] = 0, t: int | Sequence[int] = 0) -> np.ndarray:
        """ returns 3D stack of frames """
        c, z, t = (np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1) for i, e in zip('czt', (c, z, t)))
        return np.dstack([self.frame(ci, zi, ti) for ci, zi, ti in product(c, z, t)])

    def frame(self, c: int = 0, z: int = 0, t: int = 0) -> np.ndarray:
        """ returns single 2D frame """
        c = self.get_channel(c)
        c %= self.base.shape['c']
        z %= self.base.shape['z']
        t %= self.base.shape['t']

        # cache last n (default 16) frames in memory for speed (~250x faster)
        key = (c, z, t, self.transform, self.frame_decorator)
        if self.cache.maxlen and key in self.cache:
            self.cache.move_to_end(key)
            f = self.cache[key]
        else:
            f = self.transform[self.channel_names[c], t].frame(self.__frame__(c, z, t))
            if self.frame_decorator is not None:
                f = self.frame_decorator(self, f, c, z, t)
            if self.cache.maxlen:
                self.cache[key] = f
        if self.dtype is not None:
            return f.copy().astype(self.dtype)
        else:
            return f.copy()

    def get_channel(self, channel_name: str | int) -> int:
        if not isinstance(channel_name, str):
            return channel_name
        else:
            c = [i for i, c in enumerate(self.channel_names) if c.lower().startswith(channel_name.lower())]
            assert len(c) > 0, f'Channel {c} not found in {self.channel_names}'
            assert len(c) < 2, f'Channel {c} not unique in {self.channel_names}'
            return c[0]

    @staticmethod
    def get_config(file: Path | str) -> Any:
        """ Open a yml config file """
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            r'tag:yaml.org,2002:float',
            re.compile(r'''^(?:
                     [-+]?([0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?([0-9][0-9_]*)([eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$''', re.X),
            list(r'-+0123456789.'))
        with open(file) as f:
            return yaml.load(f, loader)

    def get_czt(self, c: int | Sequence[int], z: int | Sequence[int],
                t: int | Sequence[int]) -> tuple[list[int], list[int], list[int]]:
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
        return [self.get_channel(c) for c in czt[0]], *czt[1:3]  # type: ignore

    @staticmethod
    def bioformats_ome(path: [str, Path]) -> OME:
        """ Use java BioFormats to make an ome metadata structure. """
        with multiprocessing.get_context('spawn').Pool(1) as pool:
            return pool.map(bioformats_ome, (path,))[0]

    @staticmethod
    def fix_ome(ome: OME) -> OME:
        # fix ome if necessary
        for image in ome.images:
            try:
                if image.pixels.physical_size_z is None and len(set([plane.the_z
                                                                     for plane in image.pixels.planes])) > 1:
                    z = np.array([(plane.position_z * ureg.Quantity(plane.position_z_unit.value).to(ureg.m).magnitude,
                                   plane.the_z)
                                  for plane in image.pixels.planes if plane.the_c == 0 and plane.the_t == 0])
                    i = np.argsort(z[:, 1])
                    image.pixels.physical_size_z = np.nanmean(np.true_divide(*np.diff(z[i], axis=0).T)) * 1e6
                    image.pixels.physical_size_z_unit = 'µm'  # type: ignore
            except Exception:   # noqa
                pass
        return ome

    @staticmethod
    def read_ome(path: [str, Path]) -> Optional[OME]:
        path = Path(path)
        if path.with_suffix('.ome.xml').exists():
            return OME.from_xml(path.with_suffix('.ome.xml'))

    def get_ome(self) -> OME:
        """ overload this """
        return self.bioformats_ome(self.path)

    @cached_property
    def ome(self) -> OME:
        cache = OmeCache()
        if self.path not in cache:
            ome = self.read_ome(self.path)
            if ome is None:
                ome = self.get_ome()
            cache[self.path] = self.fix_ome(ome)
        return cache[self.path]

    def is_noise(self, volume: ArrayLike = None) -> bool:
        """ True if volume only has noise """
        if volume is None:
            volume = self
        fft = np.fft.fftn(volume)
        corr = np.fft.fftshift(np.fft.ifftn(fft * fft.conj()).real / np.sum(volume ** 2))
        return 1 - corr[tuple([0] * corr.ndim)] < 0.0067

    @staticmethod
    def kill_vm() -> None:
        JVM().kill_vm()

    def new(self, *args: Any, **kwargs: Any) -> View:
        warnings.warn('Imread.new has been deprecated, use Imread.view instead.', DeprecationWarning, 2)
        return self.view(*args, **kwargs)

    def save_as_movie(self, fname: Path | str = None,
                      c: int | Sequence[int] = None, z: int | Sequence[int] = None,  # noqa
                      t: str | int | Sequence[int] = None,  # noqa
                      colors: tuple[str] = None, brightnesses: tuple[float] = None,
                      scale: int = None, bar: bool = True) -> None:
        """ saves the image as a mp4 or mkv file """
        from matplotlib.colors import to_rgb
        from skvideo.io import FFmpegWriter

        if t is None:
            t = np.arange(self.shape['t'])
        elif isinstance(t, str):
            t = eval(f"np.arange(self.shape['t'])[{t}]")
        elif np.isscalar(t):
            t = (t,)

        def get_ab(tyx: Imread, p: tuple[float, float] = (1, 99)) -> tuple[float, float]:
            s = tyx.flatten()
            s = s[s > 0]
            a, b = np.percentile(s, p)
            if a == b:
                a, b = np.min(s), np.max(s)
            if a == b:
                a, b = 0, 1
            return a, b

        def cframe(frame: ArrayLike, color: str, a: float, b: float, scale: float = 1) -> np.ndarray:  # noqa
            color = to_rgb(color)
            frame = (frame - a) / (b - a)
            frame = np.dstack([255 * frame * i for i in color])
            return np.clip(np.round(frame), 0, 255).astype('uint8')

        ab = list(zip(*[get_ab(i) for i in self.transpose('cztyx')]))  # type: ignore
        colors = colors or ('r', 'g', 'b')[:self.shape['c']] + max(0, self.shape['c'] - 3) * ('w',)
        brightnesses = brightnesses or (1,) * self.shape['c']
        scale = scale or 1
        shape_x = 2 * ((self.shape['x'] * scale + 1) // 2)
        shape_y = 2 * ((self.shape['y'] * scale + 1) // 2)

        with FFmpegWriter(
                str(fname).format(name=self.path.stem, path=str(self.path.parent)),
                outputdict={'-vcodec': 'libx264', '-preset': 'veryslow', '-pix_fmt': 'yuv420p', '-r': '7',
                            '-vf': f'setpts={25 / 7}*PTS,scale={shape_x}:{shape_y}:flags=neighbor'}
        ) as movie:
            im = self.transpose('tzcyx')  # type: ignore
            for ti in tqdm(t, desc='Saving movie', disable=not bar):
                movie.writeFrame(np.max([cframe(yx, c, a, b / s, scale)
                                         for yx, a, b, c, s in zip(im[ti].max('z'), *ab, colors, brightnesses)], 0))

    def save_as_tiff(self, fname: Path | str = None, c: int | Sequence[int] = None, z: int | Sequence[int] = None,
                     t: int | Sequence[int] = None, split: bool = False, bar: bool = True, pixel_type: str = 'uint16',
                     **kwargs: Any) -> None:
        """ saves the image as a tif file
            split: split channels into different files """
        fname = Path(str(fname).format(name=self.path.stem, path=str(self.path.parent)))
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
            with TransformTiff(self, fname.with_suffix('.tif'), dtype=pixel_type,
                               pxsize=self.pxsize_um, deltaz=self.deltaz_um, **kwargs) as tif:
                for i, m in tqdm(zip(product(*[range(s) for s in shape]), product(*n)),  # noqa
                                 total=np.prod(shape), desc='Saving tiff', disable=not bar):
                    tif.save(m, *i)

    def with_transform(self, channels: bool = True, drift: bool = False, file: Path | str = None,
                       bead_files: Sequence[Path | str] = ()) -> View:
        """ returns a view where channels and/or frames are registered with an affine transformation
            channels: True/False register channels using bead_files
            drift: True/False register frames to correct drift
            file: load registration from file with name file, default: transform.yml in self.path.parent
            bead_files: files used to register channels, default: files in self.path.parent,
                with names starting with 'beads'
            """
        view = self.view()
        if file is None:
            file = Path(view.path.parent) / 'transform.yml'
        else:
            file = Path(file)
        if not bead_files:
            try:
                bead_files = Transforms.get_bead_files(view.path.parent)
            except Exception:  # noqa
                if not file.exists():
                    raise Exception('No transform file and no bead file found.')
                bead_files = ()

        if channels:
            try:
                view.transform = Transforms.from_file(file, T=drift)
            except Exception:  # noqa
                view.transform = Transforms().with_beads(view.cyllens, bead_files)
                if drift:
                    view.transform = view.transform.with_drift(view)
                view.transform.save(file.with_suffix('.yml'))
                view.transform.save_channel_transform_tiff(bead_files, file.with_suffix('.tif'))
        elif drift:
            try:
                view.transform = Transforms.from_file(file, C=False)
            except Exception:  # noqa
                view.transform = Transforms().with_drift(self)
        view.transform.adapt(view.frameoffset, view.shape.yxczt, view.channel_names)
        return view

    def set_cache_size(self, cache_size: int) -> None:
        assert isinstance(cache_size, int) and cache_size >= 0
        self.cache.maxlen = cache_size
        self.cache.truncate()

    @staticmethod
    def split_path_series(path: Path | str) -> tuple[Path, int]:
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, Path) and path.name.startswith('Pos') and path.name.lstrip('Pos').isdigit():
            return path.parent, int(path.name.lstrip('Pos'))
        return path, 0

    def view(self, *args: Any, **kwargs: Any) -> View:
        return View(self, *args, **kwargs)


class View(Imread, ABC):
    def __init__(self, base: Imread, dtype: DTypeLike = None) -> None:
        super().__init__(base.base, base.slice, base.shape, dtype or base.dtype, base.frame_decorator)
        self.transform = base.transform

    def __getattr__(self, item: str) -> Any:
        if not hasattr(self.base, item):
            raise AttributeError(f'{self.__class__} object has no attribute {item}')
        return self.base.__getattribute__(item)


class AbstractReader(Imread, metaclass=ABCMeta):
    priority = 99
    do_not_pickle = 'cache'
    ureg = ureg

    @staticmethod
    @abstractmethod
    def _can_open(path: Path | str) -> bool:
        """ Override this method, and return true when the subclass can open the file """
        return False

    @staticmethod
    def get_positions(path: str | Path) -> Optional[list[int]]:
        return None

    @abstractmethod
    def __frame__(self, c: int, z: int, t: int) -> np.ndarray:
        """ Override this, return the frame at c, z, t """
        return np.random.randint(0, 255, self.shape['yx'])

    def open(self) -> None:
        """ Optionally override this, open file handles etc.
            filehandles cannot be pickled and should be marked such by setting do_not_pickle = 'file_handle_name' """
        return

    def close(self) -> None:
        """ Optionally override this, close file handles etc. """
        return

    def __init__(self, path: Path | str | Imread | Any = None, dtype: DTypeLike = None, axes: str = None) -> None:
        if isinstance(path, Imread):
            return
        super().__init__()
        self.isclosed = False
        if isinstance(path, str):
            path = Path(path)
        self.path, self.series = self.split_path_series(path)
        if isinstance(path, Path) and path.exists():
            self.title = self.path.name
            self.acquisitiondate = datetime.fromtimestamp(self.path.stat().st_mtime).strftime('%y-%m-%dT%H:%M:%S')
        else:  # ndarray
            self.title = self.__class__.__name__
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
        image = self.ome.images[self.series if len(self.ome.images) > 1 else 0]
        pixels = image.pixels
        self.shape = pixels.size_y, pixels.size_x, pixels.size_c, pixels.size_z, pixels.size_t
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
        if image.objective_settings:
            self.objective = find(instrument.objectives, id=image.objective_settings.id)
        else:
            self.objective = None
        try:
            t0 = find(image.pixels.planes, the_c=0, the_t=0, the_z=0).delta_t
            t1 = find(image.pixels.planes, the_c=0, the_t=self.shape['t'] - 1, the_z=0).delta_t
            self.timeinterval = (t1 - t0) / (self.shape['t'] - 1) if self.shape['t'] > 1 and t1 > t0 else None
        except AttributeError:
            self.timeinterval = None
        try:
            self.binning = [int(i) for i in image.pixels.channels[0].detector_settings.binning.value.split('x')]
            if self.pxsize is not None:
                self.pxsize *= self.binning[0]
        except (AttributeError, IndexError, ValueError):
            self.binning = None
        self.channel_names = [channel.name for channel in image.pixels.channels]
        self.channel_names += [chr(97 + i) for i in range(len(self.channel_names), self.shape['c'])]
        self.cnamelist = self.channel_names
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
        self.filter = try_default(  # type: ignore
            lambda: [find(instrument.filter_sets, id=channel.filter_set_ref.id).model
                     for channel in image.pixels.channels], None)
        self.pxsize_um = None if self.pxsize is None else self.pxsize.to(self.ureg.um).m
        self.exposuretime_s = [None if i is None else i.to(self.ureg.s).m for i in self.exposuretime]

        if axes is None:
            self.axes = ''.join(i for i in 'cztyx' if self.shape[i] > 1)
        elif axes.lower() == 'full':
            self.axes = 'cztyx'
        else:
            self.axes = axes
        self.slice = [np.arange(s, dtype=int) for s in self.shape.yxczt]

        m = self.extrametadata
        if m is not None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                if 'FeedbackChannels' in m:
                    self.feedback = m['FeedbackChannels']
                else:
                    self.feedback = m['FeedbackChannel']
            except Exception:  # noqa
                self.cyllens = ['None', 'None']
                self.duolink = 'None'
                self.feedback = []
        try:
            self.cyllenschannels = np.where([self.cyllens[self.detector[c]].lower() != 'none'
                                             for c in range(self.shape['c'])])[0].tolist()
        except Exception:  # noqa
            pass
        try:
            s = int(re.findall(r'_(\d{3})_', self.duolink)[0]) * ureg.nm
        except Exception:  # noqa
            s = 561 * ureg.nm
        try:
            sigma = []
            for c, d in enumerate(self.detector):
                emission = (np.hstack(self.laserwavelengths[c]) + 22) * ureg.nm
                sigma.append([emission[emission > s].max(initial=0), emission[emission < s].max(initial=0)][d])
            sigma = np.hstack(sigma)
            sigma[sigma == 0] = 600 * ureg.nm
            sigma /= 2 * self.NA * self.pxsize
            self.sigma = sigma.magnitude.tolist()  # type: ignore
        except Exception:  # noqa
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
                self.ome.images[self.series].pixels.channels, id=f'Channel:{c}').detector_settings.id)[0]]
                                              for c in range(self.shape['c'])])
        except Exception:  # noqa
            pass


def main() -> None:
    parser = ArgumentParser(description='Display info and save as tif')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('file', help='image_file', type=str, nargs='*')
    parser.add_argument('-w', '--write', help='path to tif/movie out, {folder}, {name} and {ext} take this from file in',
                        type=str, default=None)
    parser.add_argument('-o', '--extract_ome', help='extract ome to xml file', action='store_true')
    parser.add_argument('-r', '--register', help='register channels', action='store_true')
    parser.add_argument('-c', '--channel', help='channel', type=int, default=None)
    parser.add_argument('-z', '--zslice', help='z-slice', type=int, default=None)
    parser.add_argument('-t', '--time', help='time (frames) in python slicing notation', type=str, default=None)
    parser.add_argument('-s', '--split', help='split channels', action='store_true')
    parser.add_argument('-f', '--force', help='force overwrite', action='store_true')
    parser.add_argument('-C', '--movie-colors', help='colors for channels in movie', type=str, nargs='*')
    parser.add_argument('-B', '--movie-brightnesses', help='scale brightness of each channel',
                        type=float, nargs='*')
    parser.add_argument('-S', '--movie-scale', help='upscale movie xy size, int', type=float)
    args = parser.parse_args()

    for file in tqdm(args.file, desc='operating on files', disable=len(args.file) == 1):
        file = Path(file)
        with Imread(file) as im:  # noqa
            if args.register:
                im = im.with_transform()  # noqa
            if args.write:
                write = Path(args.write.format(folder=str(file.parent), name=file.stem, ext=file.suffix)).absolute()  # noqa
                write.parent.mkdir(parents=True, exist_ok=True)
                if write.exists() and not args.force:
                    print(f'File {args.write} exists already, add the -f flag if you want to overwrite it.')
                elif write.suffix in ('.mkv', '.mp4'):
                    im.save_as_movie(write, args.channel, args.zslice, args.time, args.movie_colors,
                                     args.movie_brightnesses, args.movie_scale, bar=len(args.file) == 1)
                else:
                    im.save_as_tiff(write, args.channel, args.zslice, args.time, args.split, bar=len(args.file) == 1)
            if args.extract_ome:
                with open(im.path.with_suffix('.ome.xml'), 'w') as f:
                    f.write(im.ome.to_xml())
            if len(args.file) == 1:
                print(im.summary)


from .readers import *  # noqa
