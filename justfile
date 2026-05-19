# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

serve-docs:
    @uv sync --group docs
    @cd docs && uv run sphinx-autobuild source _build/html --watch ../src --ignore "source/_autoapi/**/*.rst" --re-ignore ".*__pycache__.*"

build-docs: test-docs
    @uv sync --group docs
    @cd docs && uv run make html

test-all:
    @uv sync --all-extras --all-groups && uv pip install jax torch --torch-backend=cpu
    @uv run pytest

test-docs:
    @uv sync --all-extras --all-groups && uv pip install jax torch --torch-backend=cpu
    @uv run pytest -m "docs"

test-core-tensorlibs:
    @uv sync --all-groups && uv pip install jax torch --torch-backend=cpu
    @uv run pytest -m "not (bq or slow or docs)"

test-core:
    @uv sync --all-groups
    @uv run pytest -m "not (bq or slow or docs or jax)"
