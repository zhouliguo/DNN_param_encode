from setuptools import setup
from Cython.Build import cythonize
import glob
import os

setup(ext_modules=cythonize(('arithmetic_compress.pyx', 'arithmetic_decompress.pyx', 'arithmeticcoding.pyx')))

pyd_files = glob.glob('*.pyd')

for file in pyd_files:
    file_s = file.split('.')
    if len(file_s)<=2:
        continue
    os.rename(file, file_s[0]+'.'+file_s[2])


#python setup.py build_ext --inplace
