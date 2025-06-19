VENV = .venv
PYTHON = $(VENV)/bin/python3
ACTIVATE = source $(VENV)/bin/activate

.PHONY: venv build install run clean

# Create virtual environment using uv
venv:
	@echo "Creating venv with uv..."
	uv venv $(VENV) -p 3.9

# Build Cython extensions
build: venv
	@echo "Building Cython extension in robot/..."
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip setuptools wheel cython
	cd robot && ../$(PYTHON) setup.py build_ext --inplace

# Install dependencies from requirements.txt
install: venv
	@echo "Installing project dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# Run the CLI script
run: venv
	@echo "Running CLI script (vision.transform)..."
	$(ACTIVATE) && $(PYTHON) -m vision.transform

# Clean up build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(VENV)
	rm -rf robot/build robot/*.c robot/*.so robot/*.pyd
