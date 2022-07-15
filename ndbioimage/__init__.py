import os
import re
import inspect
import pandas
import yaml
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
from itertools import product
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from functools import cached_property
from parfor import parfor
from tiffwrite import IJTiffFile
from numbers import Number
from argparse import ArgumentParser
from warnings import warn
from ndbioimage.transforms import Transform, Transforms
from ndbioimage.jvm import JVM
from ndbioimage._version import __version__, __git_commit_hash__


class ImTransformsBase(Transforms):
    def coords(self, array, colums=None):
        if isinstance(array, pandas.DataFrame):
            return pandas.concat([self(int(row['C']), int(row['T'])).coords(row, colums)
                                  for _, row in array.iterrows()], axis=1).T
        elif isinstance(array, pandas.Series):
            return self(int(array['C']), int(array['T'])).coords(array, colums)
        else:
            raise TypeError('Not a pandas DataFrame or Series.')


class ImTransforms(ImTransformsBase):
    """ Transforms class with methods to calculate channel transforms from bead files etc.
    """
    def __init__(self, path, cyllens, tracks=None, detectors=None, file=None, transforms=None):
        super().__init__()
        self.cyllens = cyllens
        self.tracks = tracks
        self.detectors = detectors
        if transforms is None:
            # TODO: check this
            if re.search(r'^Pos\d+', os.path.basename(path.rstrip(os.path.sep))):
                self.path = os.path.dirname(os.path.dirname(path))
            else:
                self.path = os.path.dirname(path)
            if file is not None:
                if isinstance(file, str) and file.lower().endswith('.yml'):
                    self.ymlpath = file
                    self.beadfile = None
                else:
                    self.ymlpath = os.path.join(self.path, 'transform.yml')
                    self.beadfile = file
            else:
                self.ymlpath = os.path.join(self.path, 'transform.yml')
                self.beadfile = None
            self.tifpath = self.ymlpath[:-3] + 'tif'
            try:
                self.load(self.ymlpath)
            except Exception:
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
            for key, value in transforms.items():
                self[tuple([int(i) for i in key.split(':')])] = Transform(value)

    @cached_property
    def files(self):
        try:
            if self.beadfile is None:
                files = self.get_bead_files()
            else:
                files = self.beadfile
            if isinstance(files, str):
                files = (files,)
            return files
        except Exception:
            return ()

    def __reduce__(self):
        return self.__class__, (self.path, self.cyllens, self.tracks, self.detectors, self.files, self.asdict())

    def __call__(self, channel, time=None, tracks=None, detectors=None):
        tracks = tracks or self.tracks
        detectors = detectors or self.detectors
        return super().__call__(channel, time, tracks, detectors)

    def get_bead_files(self):
        files = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().startswith('beads')
                        and not f.lower().endswith('.pdf')])
        if not files:
            raise Exception('No bead file found!')
        Files = []
        for file in files:
            try:
                if os.path.isdir(file):
                    file = os.path.join(file, 'Pos0')
                with Imread(file) as im:  # check for errors opening the file
                    pass
                Files.append(file)
            except Exception:
                continue
        if not Files:
            raise Exception('No bead file found!')
        return Files

    def calculate_transform(self, file):
        """ When no channel is not transformed by a cylindrical lens, assume that the image is scaled by a factor 1.162
            in the horizontal direction
        """
        with Imread(file) as im:
            ims = [im.max(c) for c in range(im.shape[2])]
            goodch = [c for c in range(im.shape[2]) if not im.isnoise(im.max(c))]
            untransformed = [c for c in range(im.shape[2]) if self.cyllens[im.detector[c]].lower() == 'none']

            good_and_untrans = sorted(set(goodch) & set(untransformed))
            if good_and_untrans:
                masterch = good_and_untrans[0]
            else:
                masterch = goodch[0]
            print(f'{untransformed = }, {masterch = }, {goodch = }')
            C = Transform()
            if not np.any(good_and_untrans):
                M = C.matrix
                M[0, 0] = 0.86
                C.matrix = M
            Tr = Transforms()
            for c in tqdm(goodch):
                if c == masterch:
                    Tr[im.track[c], im.detector[c]] = C
                else:
                    Tr[im.track[c], im.detector[c]] = Transform(ims[masterch], ims[c]) * C
        return Tr

    def calculate_transforms(self):
        Tq = [self.calculate_transform(file) for file in self.files]
        for key in set([key for t in Tq for key in t.keys()]):
            T = [t[key] for t in Tq if key in t]
            if len(T) == 1:
                self[key] = T[0]
            else:
                self[key] = Transform()
                self[key].parameters = np.mean([t.parameters for t in T], 0)
                self[key].dparameters = (np.std([t.parameters for t in T], 0) / np.sqrt(len(T))).tolist()

    def save_transform_tiff(self):
        C = 0
        for file in self.files:
            with Imread(file) as im:
                C = max(C, im.shape[2])
        with IJTiffFile(self.tifpath, (C, 1, len(self.files))) as tif:
            for t, file in enumerate(self.files):
                with Imread(file) as im:
                    with Imread(file, transform=True) as jm:
                        for c in range(im.shape[2]):
                            tif.save(np.hstack((im.max(c), jm.max(c))), c, 0, t)


class ImShiftTransforms(ImTransformsBase):
    """ Class to handle drift in xy. The image this is applied to must have a channeltransform already, which is then
        replaced by this class.
    """
    def __init__(self, im, shifts=None):
        """ im:                     Calculate shifts from channel-transformed images
            im, t x 2 array         Sets shifts from array, one row per frame
            im, dict {frame: shift} Sets shifts from dict, each key is a frame number from where a new shift is applied
            im, file                Loads shifts from a saved file
        """
        super().__init__()
        with (Imread(im, transform=True, drift=False) if isinstance(im, str)
                                                      else im.new(transform=True, drift=False)) as im:
            self.impath = im.path
            self.path = os.path.splitext(self.impath)[0] + '_shifts.txt'
            self.tracks, self.detectors, self.files = im.track, im.detector, im.beadfile
            if shifts is not None:
                if isinstance(shifts, np.ndarray):
                    self.shifts = shifts
                    self.shifts2transforms(im)
                elif isinstance(shifts, dict):
                    self.shifts = np.zeros((im.shape[4], 2))
                    for k in sorted(shifts.keys()):
                        self.shifts[k:] = shifts[k]
                    self.shifts2transforms(im)
                elif isinstance(shifts, str):
                    self.load(im, shifts)
            elif os.path.exists(self.path):
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

    def __reduce__(self):
        return self.__class__, (self.impath, self.shifts)

    def load(self, im, file):
        self.shifts = np.loadtxt(file)
        self.shifts2transforms(im)

    def save(self, file=None):
        self.path = file or self.path
        np.savetxt(self.path, self.shifts)

    def calulate_shifts0(self, im):
        """ Calculate shifts relative to the first frame """
        im0 = im[:, 0, 0].squeeze().transpose(2, 0, 1)

        @parfor(range(1, im.shape[4]), (im, im0), desc='Calculating image shifts.')
        def fun(t, im, im0):
            return Transform(im0, im[:, 0, t].squeeze().transpose(2, 0, 1), 'translation')
        transforms = [Transform()] + fun
        self.shifts = np.array([t.parameters[4:] for t in transforms])
        self.setTransforms(transforms, im.transform)

    def calulate_shifts(self, im):
        """ Calculate shifts relative to the previous frame """
        @parfor(range(1, im.shape[4]), (im,), desc='Calculating image shifts.')
        def fun(t, im):
            return Transform(im[:, 0, t-1].squeeze().transpose(2, 0, 1), im[:, 0, t].squeeze().transpose(2, 0, 1),
                             'translation')
        transforms = [Transform()] + fun
        self.shifts = np.cumsum([t.parameters[4:] for t in transforms])
        self.setTransforms(transforms, im.transform)

    def shifts2transforms(self, im):
        self.setTransforms([Transform(np.array(((1, 0, s[0]), (0, 1, s[1]), (0, 0, 1))))
                            for s in self.shifts], im.transform)

    def setTransforms(self, shift_transforms, channel_transforms):
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


def tolist(item):
    if isinstance(item, XmlData):
        return [item]
    elif hasattr(item, 'items'):
        return item
    elif isinstance(item, str):
        return [item]
    try:
        iter(item)
        return list(item)
    except TypeError:
        return list((item,))


class Shape(tuple):
    def __new__(cls, shape, axes='xyczt'):
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


class XmlData(OrderedDict):
    def __init__(self, elem=None):
        super(XmlData, self).__init__()
        if elem:
            if isinstance(elem, dict):
                self.update(elem)
            else:
                self.update(XmlData._todict(elem)[1])

    def re_search(self, reg, default=None, *args, **kwargs):
        return tolist(XmlData._output(XmlData._search(self, reg, True, default, *args, **kwargs)[1]))

    def search(self, key, default=None):
        return tolist(XmlData._output(XmlData._search(self, key, False, default)[1]))

    def re_search_all(self, reg, *args, **kwargs):
        K, V = XmlData._search_all(self, reg, True, *args, **kwargs)
        return {k: XmlData._output(v) for k, v in zip(K, V)}

    def search_all(self, key):
        K, V = XmlData._search_all(self, key, False)
        return {k: XmlData._output(v) for k, v in zip(K, V)}

    @staticmethod
    def _search(d, key, regex=False, default=None, *args, **kwargs):
        if isinstance(key, (list, tuple)):
            if len(key) == 1:
                key = key[0]
            else:
                for v in XmlData._search_all(d, key[0], regex, *args, **kwargs)[1]:
                    found, value = XmlData._search(v, key[1:], regex, default, *args, **kwargs)
                    if found:
                        return True, value
                return False, default

        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        return True, v
                    elif isinstance(v, dict):
                        found, value = XmlData._search(v, key, regex, default, *args, **kwargs)
                        if found:
                            return True, value
                    elif isinstance(v, (list, tuple)):
                        for w in v:
                            found, value = XmlData._search(w, key, regex, default, *args, **kwargs)
                            if found:
                                return True, value
                else:
                    found, value = XmlData._search(v, key, regex, default, *args, **kwargs)
                    if found:
                        return True, value
        return False, default

    @staticmethod
    def _search_all(d, key, regex=False, *args, **kwargs):
        K = []
        V = []
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        K.append(k)
                        V.append(v)
                    elif isinstance(v, dict):
                        q, w = XmlData._search_all(v, key, regex, *args, **kwargs)
                        K.extend([str(k) + '|' + i for i in q])
                        V.extend(w)
                    elif isinstance(v, (list, tuple)):
                        for j, val in enumerate(v):
                            q, w = XmlData._search_all(val, key, regex, *args, **kwargs)
                            K.extend([str(k) + '|' + str(j) + '|' + i for i in q])
                            V.extend(w)
                else:
                    q, w = XmlData._search_all(v, key, regex, *args, **kwargs)
                    K.extend([str(k) + '|' + i for i in q])
                    V.extend(w)
        return K, V

    @staticmethod
    def _enumdict(d):
        d2 = {}
        for k, v in d.items():
            idx = [int(i) for i in re.findall(r'(?<=:)\d+$', k)]
            if idx:
                key = re.findall(r'^.*(?=:\d+$)', k)[0]
                if key not in d2:
                    d2[key] = {}
                d2[key][idx[0]] = d['{}:{}'.format(key, idx[0])]
            else:
                d2[k] = v
        rec = False
        for k, v in d2.items():
            if [int(i) for i in re.findall(r'(?<=:)\d+$', k)]:
                rec = True
                break
        if rec:
            return XmlData._enumdict(d2)
        else:
            return d2

    @staticmethod
    def _unique_children(l):
        if l:
            keys, values = zip(*l)
            d = {}
            for k in set(keys):
                value = [v for m, v in zip(keys, values) if k == m]
                if len(value) == 1:
                    d[k] = value[0]
                else:
                    d[k] = value
            return d
        else:
            return {}

    @staticmethod
    def _todict(elem):
        d = {}
        if hasattr(elem, 'Key') and hasattr(elem, 'Value'):
            name = elem.Key.cdata
            d = elem.Value.cdata
            return name, d

        if hasattr(elem, '_attributes') and elem._attributes is not None and 'ID' in elem._attributes:
            name = elem._attributes['ID']
            elem._attributes.pop('ID')
        elif hasattr(elem, '_name'):
            name = elem._name
        else:
            name = 'none'

        if name == 'Value':
            if hasattr(elem, 'children') and len(elem.children):
                return XmlData._todict(elem.children[0])

        if hasattr(elem, 'children'):
            children = [XmlData._todict(child) for child in elem.children]
            children = XmlData._unique_children(children)
            if children:
                d = OrderedDict(d, **children)
        if hasattr(elem, '_attributes'):
            children = elem._attributes
            if children:
                d = OrderedDict(d, **children)
        if not len(d.keys()) and hasattr(elem, 'cdata'):
            return name, elem.cdata

        return name, XmlData._enumdict(d)

    @staticmethod
    def _output(s):
        if isinstance(s, dict):
            return XmlData(s)
        elif isinstance(s, (tuple, list)):
            return [XmlData._output(i) for i in s]
        elif not isinstance(s, str):
            return s
        elif len(s) > 1 and s[0] == '[' and s[-1] == ']':
            return [XmlData._output(i) for i in s[1:-1].split(', ')]
        elif re.search(r'^[-+]?\d+$', s):
            return int(s)
        elif re.search(r'^[-+]?\d?\d*\.?\d+([eE][-+]?\d+)?$', s):
            return float(s)
        elif s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return XmlData(value) if isinstance(value, dict) else value


class Imread(np.lib.mixins.NDArrayOperatorsMixin, metaclass=ABCMeta):
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
        wp@tl2019-2021
    """

    # TODO: more numpy.ndarray methods as needed
    # TODO: imread.std
    # TODO: metadata in OME format tree

    priority = 99
    do_not_pickle = 'base', 'copies', 'cache'

    @staticmethod
    @abstractmethod
    def _can_open(path):  # Override this method, and return true when the subclass can open the file
        return False

    @abstractmethod
    def __metadata__(self):
        return

    @abstractmethod
    def __frame__(self, c, z, t):  # Override this, return the frame at c, z, t
        return np.random.randint(0, 255, self.shape['xy'])

    def open(self):  # Optionally override this, open file handles etc.
        """ filehandles cannot be pickled and should be marked such by setting do_not_pickle = 'file_handle_name' """
        return

    def close(self):  # Optionally override this, close file handles etc.
        return

    def __new__(cls, path=None, *args, **kwargs):
        if cls is not Imread:
            return super().__new__(cls)
        if len(cls.__subclasses__()) == 0:
            raise Exception('Restart python kernel please!')
        if isinstance(path, Imread):
            return path
        for subclass in sorted(cls.__subclasses__(), key=lambda subclass: subclass.priority):
            if subclass._can_open(path):
                do_not_pickle = (cls.do_not_pickle,) if isinstance(cls.do_not_pickle, str) else cls.do_not_pickle
                subclass_do_not_pickle = (subclass.do_not_pickle,) if isinstance(subclass.do_not_pickle, str) \
                    else subclass.do_not_pickle if hasattr(subclass, 'do_not_pickle') else ()
                subclass.do_not_pickle = set(do_not_pickle).union(set(subclass_do_not_pickle))
                return super().__new__(subclass)

    def __init__(self, path, series=0, transform=False, drift=False, beadfile=None, sigma=None, dtype=None,
                 axes='cztxy'):
        if isinstance(path, Imread):
            return
        self._shape = Shape((0, 0, 0, 0, 0))
        self.base = None
        self.copies = []
        if isinstance(path, str):
            self.path = os.path.abspath(path)
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.acquisitiondate = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime('%y-%m-%dT%H:%M:%S')
        else:
            self.path = path  # ndarray
        self.transform = transform
        self.drift = drift
        self.beadfile = beadfile
        self.dtype = dtype
        self.series = series
        self.pxsize = 1e-1
        self.settimeinterval = 0
        self.pxsizecam = 0
        self.magnification = 0
        self.exposuretime = (0,)
        self.deltaz = 1
        self.pcf = (1, 1)
        self.laserwavelengths = [[]]
        self.laserpowers = [[]]
        self.powermode = 'normal'
        self.optovar = (1,)
        self.binning = 1
        self.collimator = (1,)
        self.tirfangle = (0,)
        self.gain = (100, 100)
        self.objective = 'unknown'
        self.filter = 'unknown'
        self.NA = 1
        self.cyllens = ['None', 'None']
        self.duolink = 'None'
        self.detector = [0, 1]
        self.track = [0]
        self.metadata = {}
        self.cache = DequeDict(16)
        self._frame_decorator = None

        self.open()
        self.__metadata__()
        self.file_shape = self.shape.xyczt

        if axes.lower() == 'squeeze':
            self.axes = ''.join(i for i in 'cztxy' if self.shape[i] > 1)
        elif axes.lower() == 'full':
            self.axes = 'cztxy'
        else:
            self.axes = axes
        self.slice = [np.arange(s, dtype=int) for s in self.shape.xyczt]

        # how far apart the centers of frame and sensor are
        self.frameoffset = self.shape['x'] / 2, self.shape['y'] / 2
        if not hasattr(self, 'cnamelist'):
            self.cnamelist = 'abcdefghijklmnopqrstuvwxyz'[:self.shape['c']]

        m = self.extrametadata
        if m is not None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                if 'FeedbackChannels' in m:
                    self.feedback = m['FeedbackChannels']
                else:
                    self.feedback = m['FeedbackChannel']
            except Exception:
                self.cyllens = ['None', 'None']
                self.duolink = 'None'
                self.feedback = []
        try:
            self.cyllenschannels = np.where([self.cyllens[self.detector[c]].lower() != 'none'
                                             for c in range(self.shape['c'])])[0].tolist()
        except Exception:
            pass
        self.set_transform()
        try:
            s = int(re.findall(r'_(\d{3})_', self.duolink)[0])
        except Exception:
            s = 561
        if sigma is None:
            try:
                sigma = []
                for t, d in zip(self.track, self.detector):
                    l = np.array(self.laserwavelengths[t]) + 22
                    sigma.append([l[l > s].max(initial=0), l[l < s].max(initial=0)][d])
                sigma = np.array(sigma)
                sigma[sigma == 0] = 600
                sigma /= 2 * self.NA * self.pxsize * 1000
                self.sigma = sigma.tolist()
            except Exception:
                self.sigma = [2] * self.shape['c']
        else:
            self.sigma = sigma
        if 1.5 < self.NA:
            self.immersionN = 1.661
        elif 1.3 < self.NA < 1.5:
            self.immersionN = 1.518
        elif 1 < self.NA < 1.3:
            self.immersionN = 1.33
        else:
            self.immersionN = 1

    @cached_property
    def timeseries(self):
        return self.shape['t'] > 1

    @cached_property
    def zstack(self):
        return self.shape['z'] > 1

    def set_transform(self):
        # handle transforms
        if self.transform is False or self.transform is None:
            self.transform = None
        else:
            if isinstance(self.transform, Transforms):
                self.transform = self.transform
            else:
                if isinstance(self.transform, str):
                    self.transform = ImTransforms(self.path, self.cyllens, self.track, self.detector, self.transform)
                else:
                    self.transform = ImTransforms(self.path, self.cyllens, self.track, self.detector, self.beadfile)
                if self.drift is True:
                    self.transform = ImShiftTransforms(self)
                elif not (self.drift is False or self.drift is None):
                    self.transform = ImShiftTransforms(self, self.drift)
            self.transform.adapt(self.frameoffset, self.shape.xyczt)
            self.beadfile = self.transform.files

    def __framet__(self, c, z, t):
        return self.transform_frame(self.__frame__(c, z, t), c, t)

    def new(self, **kwargs):
        # TODO: fix this function
        c, a = self.__reduce__()
        new_kwargs = {key: value for key, value in zip(inspect.getfullargspec(c).args[1:], a)}
        for key, value in kwargs.items():
            new_kwargs[key] = value
        return c(**new_kwargs)

    @staticmethod
    def get_config(file):
        """ Open a yml parameter file
        """
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

    @staticmethod
    def kill_vm():
        JVM().kill_vm()

    @property
    def frame_decorator(self):
        return self._frame_decorator

    @frame_decorator.setter
    def frame_decorator(self, decorator):
        self._frame_decorator = decorator
        self.cache = DequeDict(self.cache.maxlen)

    def __repr__(self):
        """ gives a helpful summary of the recorded experiment
        """
        s = [100 * '#']
        s.append('path/filename: {}'.format(self.path))
        s.append('shape ({}): '.format(self.axes) + ' ' * (5 - self.ndim) +
                 ' x '.join(('{}',) * self.ndim).format(*self.shape))
        s.append('pixelsize:     {:.2f} nm'.format(self.pxsize * 1000))
        if self.zstack:
            s.append('z-interval:    {:.2f} nm'.format(self.deltaz * 1000))
        s.append('Exposuretime:  ' + ('{:.2f} ' * len(self.exposuretime)).format(
            *(np.array(self.exposuretime) * 1000)) + 'ms')
        if self.timeseries:
            if self.timeval and np.diff(self.timeval).shape[0]:
                s.append('t-interval:    {:.3f} ± {:.3f} s'.format(
                    np.diff(self.timeval).mean(), np.diff(self.timeval).std()))
            else:
                s.append('t-interval:    {:.2f} s'.format(self.settimeinterval))
        s.append('binning:       {}x{}'.format(self.binning, self.binning))
        s.append('laser colors:  ' + ' | '.join([' & '.join(len(l)*('{:.0f}',)).format(*l)
                                                 for l in self.laserwavelengths]) + ' nm')
        s.append('laser powers:  ' + ' | '.join([' & '.join(len(l)*('{}',)).format(*[100 * i for i in l])
                                                 for l in self.laserpowers]) + ' %')
        s.append('objective:     {}'.format(self.objective))
        s.append('magnification: {}x'.format(self.magnification))
        s.append('optovar:      ' + (' {}' * len(self.optovar)).format(*self.optovar) + 'x')
        s.append('filterset:     {}'.format(self.filter))
        s.append('powermode:     {}'.format(self.powermode))
        s.append('collimator:   ' + (' {}' * len(self.collimator)).format(*self.collimator))
        s.append('TIRF angle:   ' + (' {:.2f}°' * len(self.tirfangle)).format(*self.tirfangle))
        s.append('gain:         ' + (' {:.0f}' * len(self.gain)).format(*self.gain))
        s.append('pcf:          ' + (' {:.2f}' * len(self.pcf)).format(*self.pcf))
        return '\n'.join(s)

    def __str__(self):
        return self.path

    def __len__(self):
        return self.shape[0]

    def __call__(self, *n):
        """ returns single 2D frame
            im(n):     index linearly in czt order
            im(c,z):   return im(c,z,t=0)
            im(c,z,t): return im(c,z,t)
        """
        if len(n) == 1:
            n = self.get_channel(n[0])
            c = int(n % self.shape['c'])
            z = int((n // self.shape['c']) % self.shape['z'])
            t = int((n // (self.shape['c'] * self.shape['z'])) % self.shape['t'])
            return self.frame(c, z, t)
        else:
            return self.frame(*[int(i) for i in n])

    def __getitem__(self, n):
        # None = :
        if isinstance(n, (slice, Number)):
            n = (n,)
        elif isinstance(n, type(Ellipsis)):
            n = (None,) * len(self.axes)
        n = list(n)

        # deal with ...
        ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
        if len(ell) > 1:
            raise IndexError("an index can only have a single ellipsis (...)")
        if len(ell):
            if len(n) > self.ndim:
                n.remove(Ellipsis)
            else:
                n[ell[0]] = None
                while len(n) < self.ndim:
                    n.insert(ell[0], None)
        while len(n) < self.ndim:
            n.append(None)

        T = [self.shape.axes.find(i) for i in 'xyczt']
        n = [n[j] if 0 <= j < len(n) else None for j in T]  # reorder n

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
            new = self.copy()
            new.slice = new_slice
            new._shape = Shape([1 if isinstance(s, Number) else len(s) for s in new_slice])
            new.axes = ''.join(j for j in self.axes if j in [i for i, s in zip('xyczt', new_slice)
                                                             if not isinstance(s, Number)])
            return new

    def __contains__(self, item):
        def unique_yield(l, i):
            for k in l:
                print(f'{k} from cache')
                yield k
            for k in i:
                if k not in l:
                    print(k)
                    yield k
        for idx in unique_yield(list(self.cache.keys()),
                                product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t']))):
            xyczt = (slice(None), slice(None)) + idx
            in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
            if item in np.asarray(self[in_idx]):
                return True
        return False

    def __array__(self, dtype=None):
        block = self.block(*self.slice)
        T = [self.shape.axes.find(i) for i in 'xyczt']
        S = tuple({i for i, j in enumerate(T) if j == -1}.union(
            {i for i, j in enumerate(self.slice) if isinstance(j, Number)}))
        block = block.squeeze(S)
        if dtype is not None:
            block = block.astype(dtype)
        if block.ndim == 0:
            return block.item()
        axes = ''.join(j for i, j in enumerate('xyczt') if i not in S)
        return block.transpose([axes.find(i) for i in self.shape.axes if i in axes])

    def asarray(self):
        return self.__array__()

    def astype(self, dtype):
        new = self.copy()
        new.dtype = dtype
        return new

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        exception = None
        self.base = None
        for copy in self.copies:
            try:
                copy.__exit__()
            except Exception as e:
                exception = e
        self.copies = []
        if hasattr(self, 'close'):
            self.close()
        if exception:
            raise exception

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in self.do_not_pickle}

    def __setstate__(self, state):
        """ What happens during unpickling """
        self.__dict__.update(state)
        self.open()
        self.copies = []
        self.cache = DequeDict(16)

    def __del__(self):
        print('delete')
        if not self.copies:
            if self.base is None:
                self.close()
            else:
                self.base.copies.remove(self)

    def __copy__(self):
        return self.copy()

    def copy(self):
        new = Imread.__new__(self.__class__)
        new.copies = []
        new.base = self
        for key, value in self.__dict__.items():
            if not hasattr(new, key):
                new.__dict__[key] = value
        self.copies.append(new)
        return new

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = Shape((value['xyczt'.find(i.lower())] for i in self.axes), self.axes)

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
    def ndim(self):
        return len(self.shape)

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

    def transpose(self, axes=None):
        new = self.copy()
        if axes is None:
            new.axes = new.axes[::-1]
        else:
            new.axes = ''.join(ax if isinstance(ax, str) else new.axes[ax] for ax in axes)
        return new

    @property
    def T(self):
        return self.transpose()

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

    def moveaxis(self, source, destination):
        raise NotImplementedError('moveaxis it not implemented')

    def czt(self, n):
        """ returns indices c, z, t used when calling im(n)
        """
        if not isinstance(n, tuple):
            c = n % self.shape['c']
            z = (n // self.shape['c']) % self.shape['z']
            t = (n // (self.shape['c'] * self.shape['z'])) % self.shape['t']
            return c, z, t
        n = list(n)
        if len(n) == 2 or len(n) == 4:
            n.append(slice(0, -1, 1))
        if len(n) == 3:
            n = list(n)
            for i, (ax, e) in enumerate(zip('czt', n)):
                if isinstance(e, slice):
                    a = [e.start, e.stop, e.step]
                    if a[0] is None:
                        a[0] = 0
                    if a[1] is None:
                        a[1] = -1
                    if a[2] is None:
                        a[2] = 1
                    for j in range(2):
                        if a[j] < 0:
                            a[j] %= self.shape[ax]
                            a[j] += 1
                    n[i] = np.arange(*a)
            n = [np.array(i) for i in n]
            return tuple(n)
        if len(n) == 5:
            return tuple(n[2:5])

    def czt2n(self, c, z, t):
        return c + z * self.shape['c'] + t * self.shape['c'] * self.shape['z']

    def transform_frame(self, frame, c, t=0):
        if self.transform is None:
            return frame
        else:
            return self.transform(c, t, self.track, self.detector).frame(frame)

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

    def _stats(self, fun, c=None, z=None, t=None, ffun=None):
        """ fun = np.min, np.max, np.sum or their nan varieties """
        warn('Warning: _stats is deprecated.')
        c, z, t = self.get_czt(c, z, t)
        if fun in (np.min, np.nanmin):
            val = np.inf
        elif fun in (np.max, np.nanmax):
            val = -np.inf
        else:
            val = 0
        if ffun is None:
            ffun = lambda im: im
        T = np.full(self.shape['xy'], val, self.dtype)
        for ic in c:
            m = np.full(self.shape['xy'], val, self.dtype)
            if isinstance(self.transform, ImShiftTransforms):
                for it in t:
                    n = np.full(self.shape['xy'], val, self.dtype)
                    for iz in z:
                        n = fun((n, ffun(self.__frame__(ic, iz, it))), 0)
                    m = self.transform_frame(n, ic, it)
            else:
                for it, iz in product(t, z):
                    m = fun((m, ffun(self.__frame__(ic, iz, it))), 0)
                if isinstance(self.transform, ImTransforms):
                    m = self.transform_frame(m, ic, 0)
            T = fun((T, m), 0)
        return T

    @staticmethod
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

    def argmin(self, *args, **kwargs):
        return Imread.__array_arg_fun__(self, np.argmin, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        return Imread.__array_arg_fun__(self, np.argmax, *args, **kwargs)

    def __array_fun__(self, fun, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True, ffun=None):
        """ frame-wise application of np.min, np.max, np.sum, np.mean and theis nan equivalents """
        if ffun is None:
            ffun = lambda im: im
        if dtype is None:
            dtype = self.dtype if out is None else out.dtype

        # TODO: smarter transforms
        if where is not True:
            raise NotImplementedError('Argument where != True is not implemented.')
        if axis is None:
            for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                xyczt = (slice(None), slice(None)) + idx
                in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                initial = fun(np.asarray(ffun(self[in_idx])), initial=initial)
            if out is None:
                return np.array(initial, dtype, ndmin=self.ndim) if keepdims else initial
            else:
                out.itemset(initial)
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
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                    new = ffun(self[in_idx])
                    out[out_idx] = fun(np.asarray(new), new.axes.find(axis_str), initial=initial)
            else:
                for idx in product(range(self.shape['c']), range(self.shape['z']), range(self.shape['t'])):
                    xyczt = (slice(None), slice(None)) + idx
                    out_idx = tuple(xyczt['xyczt'.find(i)] for i in out_axes)
                    in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
                    if idx['czt'.find(axis_str)] == 0:
                        out[out_idx] = fun((ffun(self[in_idx]),), 0, initial=initial)
                    else:
                        out[out_idx] = fun((out[out_idx], self[in_idx]), 0)
            return out

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.sum, axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def nansum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.nansum, axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def min(self, axis=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.min, axis, out=out, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def nanmin(self, axis=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.nanmin, axis, out=out, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def max(self, axis=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.max, axis, out=out, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def nanmax(self, axis=None, out=None, keepdims=False, initial=0, where=True, ffun=None, **kwargs):
        return Imread.__array_fun__(self, np.nanmax, axis, out=out, keepdims=keepdims, initial=initial,
                                    where=where, ffun=ffun)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **kwargs):
        res = self.sum(axis, out=out, keepdims=keepdims, where=where)
        shape = np.prod(self.shape) if axis is None else self.shape[axis]
        if out is None:
            res = res / shape
            if dtype is not None:
                res = res.astype(dtype)
            return res
        else:
            if out.dtype.kind in 'ui':
                res //= shape
            else:
                res /= shape
            return res.astype(out.dtype, copy=False)

    def nanmean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, **kwargs):
        res = self.nansum(axis, out=out, keepdims=keepdims, where=where)
        if out is None:
            res = res / self.sum(axis, None, keepdims=keepdims, where=where, ffun=lambda x: np.invert(np.isnan(x)))
            if dtype is None:
                res = res.astype(dtype)
            return res
        else:
            if out.dtype.kind in 'ui':
                res //= self.sum(axis, None, keepdims=keepdims, where=where, ffun=lambda x: np.invert(np.isnan(x)))
            else:
                res /= self.sum(axis, None, keepdims=keepdims, where=where, ffun=lambda x: np.invert(np.isnan(x)))
            return res.astype(out.dtype, copy=False)

    def reshape(self, *args, **kwargs):
        return np.asarray(self).reshape(*args, **kwargs)

    @property
    def is_noise(self):
        """ True if volume only has noise """
        F = np.fft.fftn(self)
        S = np.fft.fftshift(np.fft.ifftn(F * F.conj()).real / np.sum(self ** 2))
        return -np.log(1 - S[tuple([0] * S.ndim)]) > 5

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    def get_channel(self, channel_name):
        if not isinstance(channel_name, str):
            return channel_name
        else:
            c = [i for i, c in enumerate(self.cnamelist) if c.lower().startswith(channel_name.lower())]
            assert len(c) > 0, 'Channel {} not found in {}'.format(c, self.cnamelist)
            assert len(c) < 2, 'Channel {} not unique in {}'.format(c, self.cnamelist)
            return c[0]

    def frame(self, c=0, z=0, t=0):
        """ returns single 2D frame
        """
        c = self.get_channel(c)
        c %= self.file_shape[2]
        z %= self.file_shape[3]
        t %= self.file_shape[4]

        # cache last n (default 16) frames in memory for speed (~250x faster)
        if (c, z, t) in self.cache:
            self.cache.move_to_end((c, z, t))
            f = self.cache[(c, z, t)]
        else:
            f = self.__framet__(c, z, t)
            if self.frame_decorator is not None:
                f = self.frame_decorator(self, f, c, z, t)
            self.cache[(c, z, t)] = f
        if self.dtype is not None:
            return f.copy().astype(self.dtype)
        else:
            return f.copy()

    def data(self, c=0, z=0, t=0):
        """ returns 3D stack of frames
        """
        c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1) for i, e in zip('czt', (c, z, t))]
        return np.dstack([self.frame(ci, zi, ti) for ci, zi, ti in product(c, z, t)])

    def block(self, x=None, y=None, c=None, z=None, t=None):
        """ returns 5D block of frames
        """
        x, y, c, z, t = [np.arange(self.shape[i]) if e is None else np.array(e, ndmin=1)
                         for i, e in zip('xyczt', (x, y, c, z, t))]
        d = np.full((len(x), len(y), len(c), len(z), len(t)), np.nan, self.dtype)
        for (ci, cj), (zi, zj), (ti, tj) in product(enumerate(c), enumerate(z), enumerate(t)):
            d[:, :, ci, zi, ti] = self.frame(cj, zj, tj)[x][:, y]
        return d

    @cached_property
    def timeval(self):
        if hasattr(self, 'metadata') and isinstance(self.metadata, XmlData):
            image = self.metadata.search('Image')
            if (isinstance(image, dict) and self.series in image) or (isinstance(image, list) and len(image)):
                image = XmlData(image[0])
            return sorted(np.unique(image.search_all('DeltaT').values()))[:self.shape['t']]
        else:
            return (np.arange(self.shape['t']) * self.settimeinterval).tolist()

    @cached_property
    def timeinterval(self):
        return float(np.diff(self.timeval).mean()) if len(self.timeval) > 1 else 1

    @cached_property
    def piezoval(self):
        """ gives the height of the piezo and focus motor, only available when CylLensGUI was used
        """
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

    @cached_property
    def extrametadata(self):
        if isinstance(self.path, str) and len(self.path) > 3:
            if os.path.isfile(self.path[:-3] + 'pzl2'):
                pname = self.path[:-3] + 'pzl2'
            elif os.path.isfile(self.path[:-3] + 'pzl'):
                pname = self.path[:-3] + 'pzl'
            else:
                return
            try:
                return self.get_config(pname)
            except Exception:
                return
        return

    def save_as_tiff(self, fname=None, c=None, z=None, t=None, split=False, bar=True, pixel_type='uint16'):
        """ saves the image as a tiff-file
            split: split channels into different files
        """
        if fname is None:
            if isinstance(self.path, str):
                fname = self.path[:-3] + 'tif'
            else:
                raise Exception('No filename given.')
        elif not fname[-3:] == 'tif':
            fname += '.tif'
        if split:
            for i in range(self.shape['c']):
                if self.timeseries:
                    self.save_as_tiff(fname[:-3] + '_C{:01d}.tif'.format(i), i, 0, None, False, bar, pixel_type)
                else:
                    self.save_as_tiff(fname[:-3] + '_C{:01d}.tif'.format(i), i, None, 0, False, bar, pixel_type)
        else:
            n = [c, z, t]
            for i, ax in enumerate('czt'):
                if n[i] is None:
                    n[i] = range(self.shape[ax])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            shape = [len(i) for i in n]
            at_least_one = False
            with IJTiffFile(fname, shape, pixel_type, pxsize=self.pxsize, deltaz=self.deltaz) as tif:
                for i, m in tqdm(zip(product(*[range(s) for s in shape]), product(*n)),
                                 total=np.prod(shape), desc='Saving tiff', disable=not bar):
                    if np.any(self(*m)) or not at_least_one:
                        tif.save(self(*m), *i)
                        at_least_one = True

    @cached_property
    def summary(self):
        return self.__repr__()


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

    if os.path.exists(args.file):
        with Imread(args.file, transform=args.register) as im:
            print(im.summary)
            if args.out:
                out = os.path.abspath(args.out)
                if not os.path.exists(os.path.dirname(out)):
                    os.makedirs(os.path.dirname(out))
                if os.path.exists(out) and not args.force:
                    print('File {} exists already, add the -f flag if you want to overwrite it.'.format(args.out))
                else:
                    im.save_as_tiff(out, args.channel, args.zslice, args.time, args.split)
    else:
        print('File does not exist.')


from ndbioimage.readers import *
