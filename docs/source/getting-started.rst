Getting Started
===============

In the future, this package will be available on PyPI and will be installable with

.. code-block:: bash

    pip install hybridlane

But for now, it must be manually installed by cloning with Github:

.. code-block:: bash

    git clone https://www.github.com/pnnl/hybridlane
    pip install ./hybridlane[extras]

The available extra flags are:

- ``bq``: Adds support for the ``hybrid.bosonicqiskit`` device.

Developing
----------

To get started developing this package, first install the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
python package manager. Next, clone the package from Github and create the virtual environment

.. code-block:: bash

    git clone https://www.github.com/pnnl/hybridlane
    cd hybridlane
    uv sync --extra bq

This should take care of installing all the developer dependencies for you and build the package.

Documentation
-------------

The documentation is automatically produced by Sphinx using comments in the code. To build the documentation, run

.. code-block:: bash
    
    cd docs
    uv run make html

To enable hot-reloading (live updating) of the documentation, run

.. code-block:: bash

    cd docs
    uv run sphinx-autobuild source _build/html --watch ../src --ignore "source/api/**/*.rst" --re-ignore ".*__pycache__.*"

and open your browser to `http://localhost:8000 <http://localhost:8000>`_. For users unfamiliar with the Sphinx reStructured Text
format, there is a nice cheatsheet `here <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_.

Testing
-------

Tests can be run by pytest, optionally producing a coverage report

.. code-block:: bash

    uv run pytest [--cov=hybridlane [--cov-report=html]]
