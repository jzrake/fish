
import os
import shutil
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

shutil.copyfile('examples/euler.py', 'examples/pymara')

os.system('make -C Mara')
setup(name='Mara',
      version='1.0.0',
      author='Jonathan Zrake',
      author_email='zrake@nyu.edu',
      url='https://github.com/jzrake/fish',
      description='Solves PDE systems of compressible gasdynamics',
      packages=['Mara'],
      scripts=['examples/pymara'],
      ext_modules=[capis])
