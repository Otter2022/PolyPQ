from setuptools import setup, Extension
import os

# Extension for the k-means module (PolyPQ)
kmeans_module = Extension(
    "PolyPQ",
    sources=[
        os.path.join("..", "PolyPQ", "kmeans.c"),
        os.path.join("..", "c_api", "pykmeansmodule.c"),
    ],
    include_dirs=[os.path.join("..", "PolyPQ")],
    extra_compile_args=["-DKMEANS_THREADED", "-pthread"]
)

# Extension for the k-medoids module (pykmedoidsmodule)
kmedoids_module = Extension(
    "pykmedoidsmodule",
    sources=[
        os.path.join("..", "PolyPQ", "kmedoids.c"),
        os.path.join("..", "c_api", "pykmedoidsmodule.c"),
    ],
    include_dirs=[os.path.join("..", "PolyPQ")],
    extra_compile_args=["-pthread"]
)

setup(
    name="PolyPQ",
    version="1.0",
    description="Python interface to the product quantization library including C k-means, k-medoids and PQ functions.",
    ext_modules=[kmeans_module, kmedoids_module],
)
