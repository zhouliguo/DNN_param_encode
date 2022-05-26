from setuptools import setup
from Cython.Build import cythonize
import glob
import os

pyd_files = glob.glob('*.pyd')
for file in pyd_files:
    os.remove(file)

setup(ext_modules=cythonize(('convert.pyx', 'arithmetic_compress.pyx', 'arithmetic_decompress.pyx', 'arithmeticcoding.pyx')))

pyd_files = glob.glob('*.pyd')

for file in pyd_files:
    file_s = file.split('.')
    if len(file_s)<=2:
        continue
    os.rename(file, file_s[0]+'.'+file_s[2])

pyd_files = glob.glob('*.c')
for file in pyd_files:
    os.remove(file)


#python setup.py build_ext --inplace
