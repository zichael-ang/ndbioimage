try:
    import javabridge
    import bioformats

    class JVM:
        """ There can be only one java virtual machine per python process,
            so this is a singleton class to manage the jvm.
        """
        _instance = None
        vm_started = False
        vm_killed = False

        def __new__(cls, *args):
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

        def start_vm(self):
            if not self.vm_started and not self.vm_killed:
                javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
                outputstream = javabridge.make_instance('java/io/ByteArrayOutputStream', "()V")
                printstream = javabridge.make_instance('java/io/PrintStream', "(Ljava/io/OutputStream;)V", outputstream)
                javabridge.static_call('Ljava/lang/System;', "setOut", "(Ljava/io/PrintStream;)V", printstream)
                javabridge.static_call('Ljava/lang/System;', "setErr", "(Ljava/io/PrintStream;)V", printstream)
                self.vm_started = True
                log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
                log4j.enableLogging()
                log4j.setRootLevel("ERROR")

            if self.vm_killed:
                raise Exception('The JVM was killed before, and cannot be restarted in this Python process.')

        def kill_vm(self):
            javabridge.kill_vm()
            self.vm_started = False
            self.vm_killed = True
except ImportError:
    JVM = None
