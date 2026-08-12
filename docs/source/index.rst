.. SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
.. SPDX-License-Identifier: BSD-2-Clause

.. |PyPI - Version| image:: https://img.shields.io/pypi/v/hybridlane?logo=pypi
   :target: https://pypi.org/project/hybridlane/
.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/docs.yml?branch=main&logo=githubpages&label=docs
   :target: https://pnnl.github.io/hybridlane/
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/hybridlane?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/hybridlane/
.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/release.yml
   :target: https://github.com/pnnl/hybridlane/actions/workflows/release.yml
.. |License| image:: https://img.shields.io/github/license/pnnl/hybridlane
   :target: LICENSE.txt

|PyPI - Version| |Docs| |PyPI Downloads| |Build Status| |License|

.. image:: _static/draw_mpl/qpe_circuit.png
   :alt: hybridlane banner
   :width: 700px
   :align: center

hybridlane
==========

**hybridlane** is a Python library for designing and manipulating **hybrid continuous-variable (CV) and discrete-variable (DV) quantum circuits** within the `PennyLane <https://pennylane.ai/>`_ ecosystem. It provides a frontend for expressing hybrid quantum algorithms, implementing the concepts from the paper Y. Liu *et al*, 2026 (`PRX Quantum 7, 010201 <https://doi.org/10.1103/4rf7-9tfx>`_).

----

🚀 Features
------------

*   **⚛️ Heterogeneous quantum circuits:** Mix qubits and qumodes in the same circuit, and use our symbolic hybrid gate library to scalably build quantum algorithms.

*   **🤝 PennyLane compatibility:** Utilize existing PennyLane gates, write compilation passes as transforms, build custom hybrid backends for hardware, and perform resource estimation across mixed-variable systems.

*   **💻 Classical simulation:** Dispatch to our Jax-compatible simulator for accelerated CPU and GPU simulation and take gradients using automatic differentiation, or use `Bosonic Qiskit <https://github.com/C2QA/bosonic-qiskit>`_.

*   **💾 OpenQASM-based IR:** Leverage our intermediate representation extending OpenQASM to reduce the effort of building new hybrid backends and to facilitate interoperability with other quantum software.

----

⚙️ Installation
----------------

Install the package from PyPI:

.. code-block:: bash

    pip install hybridlane

For more details on installation and optional dependencies, see the `installation guide <guides/getting-started.html>`_.

.. warning::

    ``hybridlane`` is currently in active development and may experience breaking changes -- consider using version pinning. We welcome your feedback on our `GitHub Issues <https://github.com/pnnl/hybridlane/issues>`_ page to help us improve the software.

----

⚡ Quick Start
--------------

.. code-block:: python

    import numpy as np
    import pennylane as qp
    import hybridlane as hl

    # Create a simulator with a custom Fock truncation
    dev = qp.device("default.hybrid", fock_level=8)

    # Define a hybrid circuit with familiar PennyLane syntax
    @qp.qnode(dev)
    def circuit(n):
        for j in range(n):
            qp.X(0)  # Wire `0` is inferred to be a qubit
            # Use hybrid CV-DV gates from hybridlane
            hl.JC(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, "m"])

        # Mix qubit and qumode observables
        return hl.expval(hl.N("m") @ qp.Z(0))

    # Execute the circuit
    expval = circuit(5)
    # array(5.)

    # Perform wire type checking
    res = hl.type_check(circuit)(5)
    print(res.wire_types)
    # OrderedDict({0: Qubit(), 'm': Qumode()})

For more examples, explore the `documentation <https://pnnl.github.io/hybridlane/>`_.

----

🗺️ Roadmap
-----------

``hybridlane`` is under active development. Here are some of our future goals:

*   **Broader measurement support:** Including mid-circuit measurements and broader measurement capabilities.
*   **Algorithms and transformations:** Implementing popular algorithms and circuit transformations from research papers, including dynamic qumode allocation.
*   **Symbolic Hamiltonians:** Introducing support for symbolic bosonic Hamiltonians.
*   **Noisy simulation:** Supporting noisy quantum simulations, possibly with Dynamiqs.
*   **Catalyst/QJIT support:** Integrating with PennyLane's ``qjit`` capabilities by developing a custom MLIR dialect.
*   **Community-driven features:** Incorporating features requested by the community during usage.

----

Citing hybridlane
-----------------

If you find ``hybridlane`` useful in your research, you can cite our paper:

.. code-block::

    @misc{furches2026hybridlane,
          title={Hybridlane: A Software Development Kit for Hybrid Continuous-Discrete Variable Quantum Computing},
          author={Jim Furches and Timothy J. Stavenger and Carlos Ortiz Marrero},
          year={2026},
          eprint={2603.10919},
          archivePrefix={arXiv},
          primaryClass={quant-ph},
          url={https://arxiv.org/abs/2603.10919},
    }

----

📜 License
----------

This project is licensed under the BSD 2-Clause License - see the `LICENSE.txt <LICENSE.txt>`_ file for details.

----

🙏 Acknowledgements
--------------------

This project was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research program under contract number DE-FOA-0003265.

.. toctree::
    :maxdepth: 2
    :hidden:

    Getting started <guides/getting-started>

.. toctree::
    :maxdepth: 2
    :caption: Concepts
    :hidden:

    Introduction <guides/introduction>
    Type checking <guides/type-checking>
    Exporting to OpenQASM <guides/exporting-circuits>

.. toctree::
    :maxdepth: 1
    :caption: Demos
    :titlesonly:
    :hidden:

    demos/01-first-program
    demos/02-measurements
    demos/03-gate-decompositions
    demos/04-cross-platform-programming
    demos/05-state-and-unitary-synthesis

.. toctree::
    :maxdepth: 1
    :caption: API Reference
    :hidden:

    hl <_autoapi/hybridlane/index>
    hl.devices <_autoapi/hybridlane/devices/index>
    hl.io <_autoapi/hybridlane/io/index>
    hl.math <_autoapi/hybridlane/math/index>
    hl.measurements <_autoapi/hybridlane/measurements/index>
    hl.ops <_autoapi/hybridlane/ops/index>
    hl.templates <_autoapi/hybridlane/templates/index>
    hl.transforms <_autoapi/hybridlane/transforms/index>
    hl.wires <_autoapi/hybridlane/wires/index>
..    _autoapi/hybridlane/util/index
