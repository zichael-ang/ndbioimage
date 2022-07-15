from ndbioimage import Imread, XmlData, JVM
import os
import numpy as np
import untangle

if JVM is not None:
    import bioformats

    class Reader(Imread):
        """ This class is used as a last resort, when we don't have another way to open the file. We don't like it
            because it requires the java vm.
        """
        priority = 99  # panic and open with BioFormats
        do_not_pickle = 'reader', 'key', 'jvm'

        @staticmethod
        def _can_open(path):
            return True

        def open(self):
            self.jvm = JVM()
            self.jvm.start_vm()
            self.key = np.random.randint(1e9)
            self.reader = bioformats.get_image_reader(self.key, self.path)

        def __metadata__(self):
            s = self.reader.rdr.getSeriesCount()
            if self.series >= s:
                print('Series {} does not exist.'.format(self.series))
            self.reader.rdr.setSeries(self.series)

            X = self.reader.rdr.getSizeX()
            Y = self.reader.rdr.getSizeY()
            C = self.reader.rdr.getSizeC()
            Z = self.reader.rdr.getSizeZ()
            T = self.reader.rdr.getSizeT()
            self.shape = (X, Y, C, Z, T)

            omexml = bioformats.get_omexml_metadata(self.path)
            self.metadata = XmlData(untangle.parse(omexml))

            image = list(self.metadata.search_all('Image').values())
            if len(image) and self.series in image[0]:
                image = XmlData(image[0][self.series])
            else:
                image = self.metadata

            unit = lambda u: 10 ** {'nm': 9, 'µm': 6, 'um': 6, 'mm': 3, 'm': 0}[u]

            pxsizeunit = image.search('PhysicalSizeXUnit')[0]
            pxsize = image.search('PhysicalSizeX')[0]
            if pxsize is not None:
                self.pxsize = pxsize / unit(pxsizeunit) * 1e6

            if self.zstack:
                deltazunit = image.search('PhysicalSizeZUnit')[0]
                deltaz = image.search('PhysicalSizeZ')[0]
                if deltaz is not None:
                    self.deltaz = deltaz / unit(deltazunit) * 1e6

            if self.path.endswith('.lif'):
                self.title = os.path.splitext(os.path.basename(self.path))[0]
                self.exposuretime = self.metadata.re_search(r'WideFieldChannelInfo\|ExposureTime', self.exposuretime)
                if self.timeseries:
                    self.settimeinterval = \
                        self.metadata.re_search(r'ATLCameraSettingDefinition\|CycleTime', self.settimeinterval * 1e3)[
                            0] / 1000
                    if not self.settimeinterval:
                        self.settimeinterval = self.exposuretime[0]
                self.pxsizecam = self.metadata.re_search(r'ATLCameraSettingDefinition\|TheoCamSensorPixelSizeX',
                                                         self.pxsizecam)
                self.objective = self.metadata.re_search(r'ATLCameraSettingDefinition\|ObjectiveName', 'none')[0]
                self.magnification = \
                    self.metadata.re_search(r'ATLCameraSettingDefinition\|Magnification', self.magnification)[0]
            elif self.path.endswith('.ims'):
                self.magnification = self.metadata.search('LensPower', 100)[0]
                self.NA = self.metadata.search('NumericalAperture', 1.47)[0]
                self.title = self.metadata.search('Name', self.title)
                self.binning = self.metadata.search('BinningX', 1)[0]

        def __frame__(self, *args):
            frame = self.reader.read(*args, rescale=False).astype('float')
            if frame.ndim == 3:
                return frame[..., args[0]]
            else:
                return frame

        def close(self):
            bioformats.release_image_reader(self.key)
