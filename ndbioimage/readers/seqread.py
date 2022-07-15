from ndbioimage import Imread, XmlData
import os
import tifffile
import yaml
import json
import re


class Reader(Imread):
    priority = 10

    @staticmethod
    def _can_open(path):
        return isinstance(path, str) and os.path.splitext(path)[1] == ''

    def __metadata__(self):
        filelist = sorted([file for file in os.listdir(self.path) if re.search(r'^img_\d{3,}.*\d{3,}.*\.tif$', file)])

        try:
            with tifffile.TiffFile(os.path.join(self.path, filelist[0])) as tif:
                self.metadata = XmlData({key: yaml.safe_load(value)
                                         for key, value in tif.pages[0].tags[50839].value.items()})
        except Exception:  # fallback
            with open(os.path.join(self.path, 'metadata.txt'), 'r') as metadatafile:
                self.metadata = XmlData(json.loads(metadatafile.read()))

        # compare channel names from metadata with filenames
        cnamelist = self.metadata.search('ChNames')
        cnamelist = [c for c in cnamelist if any([c in f for f in filelist])]

        self.filedict = {}
        maxc = 0
        maxz = 0
        maxt = 0
        for file in filelist:
            T = re.search(r'(?<=img_)\d{3,}', file)
            Z = re.search(r'\d{3,}(?=\.tif$)', file)
            C = file[T.end() + 1:Z.start() - 1]
            t = int(T.group(0))
            z = int(Z.group(0))
            if C in cnamelist:
                c = cnamelist.index(C)
            else:
                c = len(cnamelist)
                cnamelist.append(C)

            self.filedict[(c, z, t)] = file
            if c > maxc:
                maxc = c
            if z > maxz:
                maxz = z
            if t > maxt:
                maxt = t
        self.cnamelist = [str(cname) for cname in cnamelist]

        X = self.metadata.search('Width')[0]
        Y = self.metadata.search('Height')[0]
        self.shape = (int(X), int(Y), maxc + 1, maxz + 1, maxt + 1)

        self.pxsize = self.metadata.re_search(r'(?i)pixelsize_?um', 0)[0]
        if self.zstack:
            self.deltaz = self.metadata.re_search(r'(?i)z-step_?um', 0)[0]
        if self.timeseries:
            self.settimeinterval = self.metadata.re_search(r'(?i)interval_?ms', 0)[0] / 1000
        if 'Hamamatsu' in self.metadata.search('Core-Camera', '')[0]:
            self.pxsizecam = 6.5
        self.title = self.metadata.search('Prefix')[0]
        self.acquisitiondate = self.metadata.search('Time')[0]
        self.exposuretime = [i / 1000 for i in self.metadata.search('Exposure-ms')]
        self.objective = self.metadata.search('ZeissObjectiveTurret-Label')[0]
        self.optovar = []
        for o in self.metadata.search('ZeissOptovar-Label'):
            a = re.search(r'\d?\d*[,.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))
        if self.pxsize == 0:
            self.magnification = int(re.findall(r'(\d+)x', self.objective)[0]) * self.optovar[0]
            self.pxsize = self.pxsizecam / self.magnification
        else:
            self.magnification = self.pxsizecam / self.pxsize
        self.pcf = self.shape[2] * self.metadata.re_search(r'(?i)conversion\sfactor\scoeff', 1)
        self.filter = self.metadata.search('ZeissReflectorTurret-Label', self.filter)[0]

    def __frame__(self, c=0, z=0, t=0):
        return tifffile.imread(os.path.join(self.path, self.filedict[(c, z, t)]))
