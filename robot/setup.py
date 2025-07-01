from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


ext_modules = cythonize(
    Extension(
        "robot.rpc", ["rpc.py"], extra_compile_args=["-fno-var-tracking-assignments"]
    ),
    compiler_directives={"language_level": "3"},
)

setup(
    name="fairino",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    setup_requires=["cython>=3.0.12"],
)
