from setuptools import setup, Extension
import os

module = Extension(
    "PolyPQ",
    sources=[
        os.path.join("..", "PolyPQ", "kmeans.c"),
  #      os.path.join("..", "PolyPQ", "pq.c"),         # <-- New file
        os.path.join("..", "c_api", "pykmeansmodule.c"),# Existing wrapper (or pypqmodule.c if separate)
    ],
    include_dirs=[os.path.join("..", "PolyPQ")],
    extra_compile_args=["-DKMEANS_THREADED", "-pthread"]
)

setup(
    name="PolyPQ",
    version="1.0",
    description="Python interface to the product quantization library including C k-means and PQ functions.",
    ext_modules=[module],
)
