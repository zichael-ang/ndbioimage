from pathlib import Path
from urllib import request

try:
    class JVM:
        """ There can be only one java virtual machine per python process,
            so this is a singleton class to manage the jvm.
        """
        _instance = None
        vm_started = False
        vm_killed = False
        success = True

        def __new__(cls, *args):
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

        def __init__(self, jars=None):
            if not self.vm_started and not self.vm_killed:
                try:
                    jar_path = Path(__file__).parent / 'jars'
                    if jars is None:
                        jars = {}
                    for jar, src in jars.items():
                        if not (jar_path / jar).exists():
                            JVM.download(src, jar_path / jar)
                    classpath = [str(jar_path / jar) for jar in jars.keys()]

                    import jpype
                    jpype.startJVM(classpath=classpath)
                except Exception:  # noqa
                    self.vm_started = False
                else:
                    self.vm_started = True
                try:
                    import jpype.imports
                    from loci.common import DebugTools  # noqa
                    from loci.formats import ChannelSeparator  # noqa
                    from loci.formats import FormatTools  # noqa
                    from loci.formats import ImageReader  # noqa
                    from loci.formats import MetadataTools  # noqa

                    DebugTools.setRootLevel("ERROR")

                    self.image_reader = ImageReader
                    self.channel_separator = ChannelSeparator
                    self.format_tools = FormatTools
                    self.metadata_tools = MetadataTools
                except Exception:  # noqa
                    pass

            if self.vm_killed:
                raise Exception('The JVM was killed before, and cannot be restarted in this Python process.')

        @staticmethod
        def download(src, dest):
            print(f'Downloading {dest.name} to {dest}.')
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(request.urlopen(src).read())

        @classmethod
        def kill_vm(cls):
            self = cls._instance
            if self is not None and self.vm_started and not self.vm_killed:
                import jpype
                jpype.shutdownJVM()  # noqa
            self.vm_started = False
            self.vm_killed = True

except ImportError:
    JVM = None
