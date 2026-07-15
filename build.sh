#!/bin/bash

pip install -e ".[dev]"
ruff format .
ruff check . --fix
pyright
pytest
