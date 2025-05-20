#!/bin/bash

# Create virtual environment with Python 3.12
uv venv .venv -p 3.12

source venv/bin/activate # linux/macos
# source venv/Scripts/activate # windows

# Install dependencies
uv pip install setuptools cython

# Create empty __init__.py
touch fairino/__init__.py

# Create setup.py in fairino/
# -fno-var-tracking-assignments disables the GCC variable 
# tracking feature that causes size overrun warnings
echo "Creating setup.py in fairino.."
cat > fairino/setup.py << 'EOF'
import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


ext_modules = cythonize(
    Extension(
        "fairino.Robot",
        ["Robot.py"],
        extra_compile_args=["-fno-var-tracking-assignments"]
    ),
    compiler_directives={'language_level': "3"}
)

setup(
    name='fairino',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    setup_requires=['cython>=3.0.12']
)
EOF

# Move to fairino/ and compile
cd fairino/
python setup.py build_ext

cd ..

echo "Build completed successfully."