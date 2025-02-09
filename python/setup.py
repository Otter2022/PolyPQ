from setuptools import setup, Extension
import os

here = os.path.abspath(os.path.dirname(__file__))

source_file = os.path.join(here, '..', 'c_api', '_hello.c')

hello_extension = Extension(
    '_hello',  
    sources=[source_file]
)

setup(
    name='PolyPQ',
    version='1.0',
    ext_modules=[hello_extension],
)