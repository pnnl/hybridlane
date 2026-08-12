Getting Started
===============

Installation
------------

This package can be installed from PyPI with pip

.. code-block:: bash

    pip install hybridlane

We have a few optional dependencies:

- ``all``: Installs all optional dependencies.
- ``bq``: Adds support for simulating circuits using Bosonic Qiskit with the ``bosonicqiskit.hybrid`` device.
- ``qscout``: Installs libraries necessary for compiling circuits to Sandia National Laboratory's QSCOUT ion trap using the ``sandiaqscout.hybrid`` device.

.. note::

    The ``sandiaqscout.hybrid`` device does not grant cloud access or the ability to run circuits on their hardware. It only provides the tools to compile circuits to the native hardware gate set.

Developing
----------

To get started developing this package, first install the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
python package manager. Next, clone the package from Github and create the virtual environment

.. code-block:: bash

    git clone https://www.github.com/pnnl/hybridlane
    cd hybridlane
    uv sync --all-extras

This should take care of installing all the developer dependencies for you and build the package.

To help improve code quality, we use the Ruff and Ty linters. You can invoke them with

.. code-block:: bash

    uv run ruff check
    uv run ruff format
    uv run ty check

We have a suite of tests that can be run with pytest, with support for code coverage reporting.

.. code-block:: bash

    uv run pytest [--cov=hybridlane [--cov-report=html]]

Some of the tests can be quite slow, so we have some `Justfile <https://github.com/casey/just>`_ recipes and pytest markers to help you pick only the tests you would like to run.

* ``just test-core``: Runs only the core hybridlane library tests using NumPy.
* ``just test-core-tensorlibs``: Runs only the core hybridlane library tests using NumPy and JAX.
* ``just test-docs``: Run only the documentation tests.
* ``just test-all``: Run all tests, including slow tests and tests requiring external dependencies.

Documentation
-------------

The documentation is automatically produced by Sphinx using docstrings in the code. To build the documentation, we have some Justfile recipes:

.. code-block:: bash

    just build-docs

To enable hot-reloading (live updating) of the documentation, run

.. code-block:: bash

    just serve-docs

and open your browser to `http://localhost:8000 <http://localhost:8000>`_. For users unfamiliar with the Sphinx reStructured Text
format, there is a nice cheatsheet `here <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_.
