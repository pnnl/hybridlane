# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

serve-docs:
    @uv sync --group docs
    @cd docs && uv run sphinx-autobuild source _build/html --watch ../src --ignore "source/_autoapi/**/*.rst" --re-ignore ".*__pycache__.*"

build-docs: test-docs
    @uv sync --group docs
    @cd docs && uv run make html

test-all:
    @uv sync --all-extras --locked
    @uv run pytest

test-docs:
    @uv sync --all-extras --locked
    @uv run pytest -m "docs"

test-core-tensorlibs:
    @uv sync --locked && uv pip install torch --torch-backend=cpu
    @uv run pytest -m "not (bq or qscout or docs)"

test-core:
    @uv sync --locked
    @uv run pytest -m "not (bq or qscout or docs or jax or torch)"

codecov-core:
    @uv sync --locked
    @uv run pytest -m "not (bq or qscout or docs or jax or torch)" --cov=hybridlane --cov-report=html --cov-report=term-missing

codecov:
    @uv sync --all-extras --locked && uv pip install torch --torch-backend=cpu
    @uv run pytest --cov=hybridlane --cov-report=html --cov-report=term-missing
