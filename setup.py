
import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension

fluids_src = ['fluids.c', 'riemann.c', 'matrix.c']
fish_src = ['fish.c', 'reconstruct.c']

capis = Extension("Mara.capis",
                  sources = ["Mara/capis.c"] +\
                      [os.path.join("src", c) for c in fish_src + fluids_src],
                  extra_compile_args = ["-std=c99"],
                  include_dirs=["src", np.get_include()])

os.system('make -C Mara')
setup(name='Mara',
      packages=['Mara'],
      version='0.6.0',
      ext_modules=[capis])
