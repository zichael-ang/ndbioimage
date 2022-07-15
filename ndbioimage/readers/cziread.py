from ndbioimage import Imread, XmlData, tolist
import czifile
import untangle
import numpy as np
import re
from functools import cached_property


class Reader(Imread):
    priority = 0
    do_not_pickle = 'reader', 'filedict'

    @staticmethod
    def _can_open(path):
        return isinstance(path, str) and path.endswith('.czi')

    def open(self):
        self.reader = czifile.CziFile(self.path)
        filedict = {}
        for directory_entry in self.reader.filtered_subblock_directory:
            idx = self.get_index(directory_entry, self.reader.start)
            for c in range(*idx[self.reader.axes.index('C')]):
                for z in range(*idx[self.reader.axes.index('Z')]):
                    for t in range(*idx[self.reader.axes.index('T')]):
                        if (c, z, t) in filedict:
                            filedict[(c, z, t)].append(directory_entry)
                        else:
                            filedict[(c, z, t)] = [directory_entry]
        self.filedict = filedict

    def close(self):
        self.reader.close()

    def __metadata__(self):
        # TODO: make sure frame function still works when a subblock has data from more than one frame
        self.shape = tuple([self.reader.shape[self.reader.axes.index(directory_entry)] for directory_entry in 'XYCZT'])
        self.metadata = XmlData(untangle.parse(self.reader.metadata()))

        image = [i for i in self.metadata.search_all('Image').values() if i]
        if len(image) and self.series in image[0]:
            image = XmlData(image[0][self.series])
        else:
            image = self.metadata

        pxsize = image.search('ScalingX')[0]
        if pxsize is not None:
            self.pxsize = pxsize * 1e6
        if self.zstack:
            deltaz = image.search('ScalingZ')[0]
            if deltaz is not None:
                self.deltaz = deltaz * 1e6

        self.title = self.metadata.re_search(('Information', 'Document', 'Name'), self.title)[0]
        self.acquisitiondate = self.metadata.re_search(('Information', 'Document', 'CreationDate'),
                                                       self.acquisitiondate)[0]
        self.exposuretime = self.metadata.re_search(('TrackSetup', 'CameraIntegrationTime'), self.exposuretime)
        if self.timeseries:
            self.settimeinterval = self.metadata.re_search(('Interval', 'TimeSpan', 'Value'),
                                                           self.settimeinterval * 1e3)[0] / 1000
            if not self.settimeinterval:
                self.settimeinterval = self.exposuretime[0]
        self.pxsizecam = self.metadata.re_search(('AcquisitionModeSetup', 'PixelPeriod'), self.pxsizecam)
        self.magnification = self.metadata.re_search('NominalMagnification', self.magnification)[0]
        attenuators = self.metadata.search_all('Attenuator')
        self.laserwavelengths = [[1e9 * float(i['Wavelength']) for i in tolist(attenuator)]
                                 for attenuator in attenuators.values()]
        self.laserpowers = [[float(i['Transmission']) for i in tolist(attenuator)]
                            for attenuator in attenuators.values()]
        self.collimator = self.metadata.re_search(('Collimator', 'Position'))
        detector = self.metadata.search(('Instrument', 'Detector'))
        self.gain = [int(i.get('AmplificationGain', 1)) for i in detector]
        self.powermode = self.metadata.re_search(('TrackSetup', 'FWFOVPosition'))[0]
        optovar = self.metadata.re_search(('TrackSetup', 'TubeLensPosition'), '1x')
        self.optovar = []
        for o in optovar:
            a = re.search(r'\d?\d*[,.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))
        self.pcf = [2 ** self.metadata.re_search(('Image', 'ComponentBitCount'), 14)[0] / float(i)
                    for i in self.metadata.re_search(('Channel', 'PhotonConversionFactor'), 1)]
        self.binning = self.metadata.re_search(('AcquisitionModeSetup', 'CameraBinning'), 1)[0]
        self.objective = self.metadata.re_search(('AcquisitionModeSetup', 'Objective'))[0]
        self.NA = self.metadata.re_search(('Instrument', 'Objective', 'LensNA'))[0]
        self.filter = self.metadata.re_search(('TrackSetup', 'BeamSplitter', 'Filter'))[0]
        self.tirfangle = [50 * i for i in self.metadata.re_search(('TrackSetup', 'TirfAngle'), 0)]
        self.frameoffset = [self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetX'))[0],
                            self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetY'))[0]]
        self.cnamelist = [c['DetectorSettings']['Detector']['Id'] for c in
                          self.metadata['ImageDocument']['Metadata']['Information']['Image'].search('Channel')]
        try:
            self.track, self.detector = zip(*[[int(i) for i in re.findall(r'\d', c)] for c in self.cnamelist])
        except ValueError:
            self.track = tuple(range(len(self.cnamelist)))
            self.detector = (0,) * len(self.cnamelist)

    def __frame__(self, c=0, z=0, t=0):
        f = np.zeros(self.file_shape[:2], self.dtype)
        for directory_entry in self.filedict[(c, z, t)]:
            subblock = directory_entry.data_segment()
            tile = subblock.data(resize=True, order=0)
            index = [slice(i - j, i - j + k) for i, j, k in
                     zip(directory_entry.start, self.reader.start, tile.shape)]
            index = tuple([index[self.reader.axes.index(i)] for i in 'XY'])
            f[index] = tile.squeeze()
        return f

    @staticmethod
    def get_index(directory_entry, start):
        return [(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, directory_entry.shape)]

    @cached_property
    def timeval(self):
        tval = np.unique(list(filter(lambda x: x.attachment_entry.filename.startswith('TimeStamp'),
                                     self.reader.attachments()))[0].data())
        return sorted(tval[tval > 0])[:self.shape['t']]