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
   :alt: hybridlane logo
   :width: 700px
   :align: center

hybridlane
==========

**hybridlane** is a Python library for designing and manipulating **hybrid continuous-variable (CV) and discrete-variable (DV) quantum circuits** within the `PennyLane <https://pennylane.ai/>`_ ecosystem. It provides a frontend for expressing hybrid quantum algorithms, implementing the concepts from the paper Y. Liu *et al*, 2026 (`PRX Quantum 7, 010201 <https://doi.org/10.1103/4rf7-9tfx>`_).

----

✨ Why hybridlane?
--------------------

As quantum computing explores beyond traditional qubit-only models, `hybridlane` offers a powerful and intuitive framework for researchers and developers to:

*   **Design complex hybrid circuits effortlessly:** Seamlessly integrate qubits and qumodes in the same circuit.
*   **Describe large-scale circuits:** Define hybrid gate semantics independently of simulation, enabling fast description of wide and deep circuits with minimal memory.
*   **Leverage the PennyLane ecosystem:** Integrate with PennyLane's extensive tools for transformations, resource estimation, and device support.

----

🚀 Features
----------------

*   **📃 Hybrid Gate Semantics:** Precise, platform-independent definitions for hybrid gates, enabling rapid construction of large-scale quantum circuits.

*   **⚛️ Native Qumode Support:** Qumodes are treated as a fundamental wire type, with automatic type inference that simplifies circuit construction and enhances readability.

*   **🤝 PennyLane Compatibility:** A familiar interface for PennyLane users. Utilize existing PennyLane gates, build custom hybrid devices, write compilation passes, and perform resource estimation across mixed-variable systems.

*   **💻 Classical Simulation:** Dispatch to our Jax-compatible simulator and take gradients using automatic differentiation, or use `Bosonic Qiskit <https://github.com/C2QA/bosonic-qiskit>`_.

*   **💾 OpenQASM-based IR:** An intermediate representation based on an extended OpenQASM, promoting interoperability and enabling advanced circuit manipulations.

----

⚙️ Installation
------------------

`hybridlane` is currently in **early preview**. We welcome your feedback on our `GitHub Issues <https://github.com/pnnl/hybridlane/issues>`_ page to help us improve.

Install the package from PyPI:

.. code-block:: bash

    pip install hybridlane

**Available Extras:**

*   ``[all]``: Installs all extra dependencies.

*   ``[bq]``: Installs support for the ``bosonicqiskit.hybrid`` simulation device.

*   ``[qscout]``: Installs support for the ``sandiaqscout.hybrid`` compilation device.

For detailed instructions, see the `Getting Started Guide <https://pnnl.github.io/hybridlane/getting-started.html>`_ in our documentation.

----

⚡ Quick Start
-----------------

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

For more examples, explore our `Documentation <https://pnnl.github.io/hybridlane/>`_.

----

🗺️ Roadmap
-------------

`hybridlane` is under active development. Here are some of our future goals:

*   **Broader measurement support:** Including mid-circuit measurements and broader measurement capabilities.
*   **Algorithms and transformations:** Implementing popular algorithms and circuit transformations from research papers, including dynamic qumode allocation.
*   **Symbolic Hamiltonians:** Introducing support for symbolic bosonic Hamiltonians.
*   **Noisy simulation:** Supporting noisy simulations with Bosonic Qiskit.
*   **Pulse-level gates:** Allowing pulse-level gates and simulating them in Dynamiqs.
*   **Catalyst/QJIT support:** Integrating with PennyLane's `qjit` capabilities by developing a custom MLIR dialect.
*   **Community-driven features:** Incorporating features requested by the community during usage.

----

📚 Documentation
---------------------

For comprehensive information on `hybridlane`'s API, tutorials, and technical background, please visit our official `Documentation <https://pnnl.github.io/hybridlane/>`_.

----

❓ Support
-------------

For questions, bug reports, or feature requests, please open an issue on our `GitHub Issues page <https://github.com/pnnl/hybridlane/issues>`_.

----

Citing hybridlane
-------------------

If you find ``hybridlane`` useful in your research, please cite our paper:

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
-------------

This project is licensed under the BSD 2-Clause License - see the `LICENSE.txt <LICENSE.txt>`_ file for details.

----

🙏 Acknowledgements
----------------------

This project was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research program under contract number DE-FOA-0003265.

.. toctree::
    :maxdepth: 2
    :caption: Using Hybridlane
    :hidden:

    introduction
    getting-started
    type-checking
    exporting-circuits

.. toctree::
    :maxdepth: 1
    :caption: API Reference
    :hidden:

    hl <_autoapi/hybridlane/index>
    hl.devices <_autoapi/hybridlane/devices/index>
    hl.io <_autoapi/hybridlane/io/index>
    hl.measurements <_autoapi/hybridlane/measurements/index>
    hl.ops <_autoapi/hybridlane/ops/index>
    hl.sa <_autoapi/hybridlane/sa/index>
    hl.templates <_autoapi/hybridlane/templates/index>
    hl.transforms <_autoapi/hybridlane/transforms/index>
..    _autoapi/hybridlane/util/index
