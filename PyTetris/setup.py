from distutils.core import setup, Extension
import numpy

import os

ldr = os.listdir("./")
cpplist = [x for x in ldr if x[-4:] == ".cpp"]

setup(name = 'PyTetris', version = '0.1.0',
      ext_modules = [
          Extension('PyTetris', cpplist,
                    include_dirs = ["./SDL2/include"],
                    extra_compile_args = ["/std:c++latest"],
                    library_dirs = ["./SDL2/lib/x64"],
                    libraries = ["SDL2"]
          )],
      include_dirs = [numpy.get_include(), "./SDL2/include"],
      data_files=[('', ['SDL2.dll'])]
)
