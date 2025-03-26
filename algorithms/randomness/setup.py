import os
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Absoluter Pfad zum aktuellen Verzeichnis
current_dir = os.getcwd()

setup(
    ext_modules=cythonize("occ_optimization.pyx"),
    include_dirs=[np.get_include()],
    options={'build_ext': {'build_lib': current_dir}}  # Kopiert die Datei hierher
)
