VENV = .venv
PYTHON = $(VENV)/bin/python3

.PHONY: venv build install run

venv:
	@echo "Creating venv and installing core dependencies..."
	uv venv $(VENV) -p 3.9
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install setuptools cython

build: venv
	@echo "Building cython extension in robot/..."
	cd robot && ../$(PYTHON) setup.py build_ext

install:
	@echo "Installing dependencies..."
	# rm -rf robot/build robot/*.c robot/*.so robot/*.pyd ?????
	$(PYTHON) -m pip install -r requirements.txt

run:
	@echo "Running CLI script (vision.transform)..."
	$(PYTHON) -m vision.transform
