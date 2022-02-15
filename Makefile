SHELL := /bin/bash

lint: FORCE
	flake8
	black --extend-exclude=\.ipynb --check .
	isort --check .
	# python scripts/update_headers.py --check
	mypy .

format: FORCE
	black --extend-exclude=\.ipynb .
	isort .
	# python scripts/update_headers.py

test: lint FORCE
	pytest -v -n auto test

FORCE:
