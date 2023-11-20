import multiprocessing
from abc import ABC
from multiprocessing import queues
from traceback import print_exc

import numpy as np

from .. import JVM, AbstractReader

jars = {'bioformats_package.jar':
            'https://downloads.openmicroscopy.org/bio-formats/latest/artifacts/bioformats_package.jar'}


class JVMReader:
    def __init__(self, path, series):
        mp = multiprocessing.get_context('spawn')
        self.path = path
        self.series = series
        self.queue_in = mp.Queue()
        self.queue_out = mp.Queue()
        self.queue_error = mp.Queue()
        self.done = mp.Event()
        self.process = mp.Process(target=self.run)
        self.process.start()
        self.is_alive = True

    def close(self):
        if self.is_alive:
            self.done.set()
            while not self.queue_in.empty():
                self.queue_in.get()
            self.queue_in.close()
            self.queue_in.join_thread()
            while not self.queue_out.empty():
                print(self.queue_out.get())
            self.queue_out.close()
            self.process.join()
            self.process.close()
            self.is_alive = False

    def frame(self, c, z, t):
        self.queue_in.put((c, z, t))
        return self.queue_out.get()

    def run(self):
        """ Read planes from the image reader file.
            adapted from python-bioformats/bioformats/formatreader.py
        """
        jvm = JVM(jars)
        reader = jvm.image_reader()
        ome_meta = jvm.metadata_tools.createOMEXMLMetadata()
        reader.setMetadataStore(ome_meta)
        reader.setId(str(self.path))
        reader.setSeries(self.series)

        open_bytes_func = reader.openBytes
        width, height = int(reader.getSizeX()), int(reader.getSizeY())

        pixel_type = reader.getPixelType()
        little_endian = reader.isLittleEndian()

        if pixel_type == jvm.format_tools.INT8:
            dtype = np.int8
        elif pixel_type == jvm.format_tools.UINT8:
            dtype = np.uint8
        elif pixel_type == jvm.format_tools.UINT16:
            dtype = '<u2' if little_endian else '>u2'
        elif pixel_type == jvm.format_tools.INT16:
            dtype = '<i2' if little_endian else '>i2'
        elif pixel_type == jvm.format_tools.UINT32:
            dtype = '<u4' if little_endian else '>u4'
        elif pixel_type == jvm.format_tools.INT32:
            dtype = '<i4' if little_endian else '>i4'
        elif pixel_type == jvm.format_tools.FLOAT:
            dtype = '<f4' if little_endian else '>f4'
        elif pixel_type == jvm.format_tools.DOUBLE:
            dtype = '<f8' if little_endian else '>f8'
        else:
            dtype = None

        try:
            while not self.done.is_set():
                try:
                    c, z, t = self.queue_in.get(True, 0.02)
                    if reader.isRGB() and reader.isInterleaved():
                        index = reader.getIndex(z, 0, t)
                        image = np.frombuffer(open_bytes_func(index), dtype)
                        image.shape = (height, width, reader.getSizeC())
                        if image.shape[2] > 3:
                            image = image[:, :, :3]
                    elif c is not None and reader.getRGBChannelCount() == 1:
                        index = reader.getIndex(z, c, t)
                        image = np.frombuffer(open_bytes_func(index), dtype)
                        image.shape = (height, width)
                    elif reader.getRGBChannelCount() > 1:
                        n_planes = reader.getRGBChannelCount()
                        rdr = jvm.channel_separator(reader)
                        planes = [np.frombuffer(rdr.openBytes(rdr.getIndex(z, i, t)), dtype) for i in range(n_planes)]
                        if len(planes) > 3:
                            planes = planes[:3]
                        elif len(planes) < 3:
                            # > 1 and < 3 means must be 2
                            # see issue #775
                            planes.append(np.zeros(planes[0].shape, planes[0].dtype))
                        image = np.dstack(planes)
                        image.shape = (height, width, 3)
                        del rdr
                    elif reader.getSizeC() > 1:
                        images = [np.frombuffer(open_bytes_func(reader.getIndex(z, i, t)), dtype)
                                  for i in range(reader.getSizeC())]
                        image = np.dstack(images)
                        image.shape = (height, width, reader.getSizeC())
                        # if not channel_names is None:
                        #     metadata = MetadataRetrieve(self.metadata)
                        #     for i in range(self.reader.getSizeC()):
                        #         index = self.reader.getIndex(z, 0, t)
                        #         channel_name = metadata.getChannelName(index, i)
                        #         if channel_name is None:
                        #             channel_name = metadata.getChannelID(index, i)
                        #         channel_names.append(channel_name)
                    elif reader.isIndexed():
                        #
                        # The image data is indexes into a color lookup-table
                        # But sometimes the table is the identity table and just generates
                        # a monochrome RGB image
                        #
                        index = reader.getIndex(z, 0, t)
                        image = np.frombuffer(open_bytes_func(index), dtype)
                        if pixel_type in (jvm.format_tools.INT16, jvm.format_tools.UINT16):
                            lut = reader.get16BitLookupTable()
                            if lut is not None:
                                lut = np.array(lut)
                                # lut = np.array(
                                #     [env.get_short_array_elements(d)
                                #      for d in env.get_object_array_elements(lut)]) \
                                #     .transpose()
                        else:
                            lut = reader.get8BitLookupTable()
                            if lut is not None:
                                lut = np.array(lut)
                                # lut = np.array(
                                #     [env.get_byte_array_elements(d)
                                #      for d in env.get_object_array_elements(lut)]) \
                                #     .transpose()
                        image.shape = (height, width)
                        if (lut is not None) and not np.all(lut == np.arange(lut.shape[0])[:, np.newaxis]):
                            image = lut[image, :]
                    else:
                        index = reader.getIndex(z, 0, t)
                        image = np.frombuffer(open_bytes_func(index), dtype)
                        image.shape = (height, width)

                    if image.ndim == 3:
                        self.queue_out.put(image[..., c])
                    else:
                        self.queue_out.put(image)
                except queues.Empty:  # noqa
                    continue
        except (Exception,):
            print_exc()
            self.queue_out.put(np.zeros((32, 32)))
        finally:
            jvm.kill_vm()


def can_open(path):
    try:
        jvm = JVM(jars)
        reader = jvm.image_reader()
        reader.getFormat(str(path))
        return True
    except (Exception,):
        return False
    finally:
        jvm.kill_vm()  # noqa


class Reader(AbstractReader, ABC):
    """ This class is used as a last resort, when we don't have another way to open the file. We don't like it
        because it requires the java vm.
    """
    priority = 99  # panic and open with BioFormats
    do_not_pickle = 'reader', 'key', 'jvm'

    @staticmethod
    def _can_open(path):
        """ Use java BioFormats to make an ome metadata structure. """
        with multiprocessing.get_context('spawn').Pool(1) as pool:
            ome = pool.map(can_open, (path,))[0]
            return ome

    def open(self):
        self.reader = JVMReader(self.path, self.series)

    def __frame__(self, c, z, t):
        return self.reader.frame(c, z, t)

    def close(self):
        self.reader.close()
