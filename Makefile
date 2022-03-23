SHELL := /bin/bash

install:
	pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.htm
	pip install -e .[test]

lint: FORCE
	flake8
	black --extend-exclude=\.ipynb --check .
	# isort --check .
	# python scripts/update_headers.py --check
	mypy .

format: FORCE
	black --extend-exclude=\.ipynb .
	# isort .
	# python scripts/update_headers.py

test: lint FORCE
	pytest -v -n auto test

FORCE:
