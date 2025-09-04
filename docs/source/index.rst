hybridlane documentation
============================

Welcome to the hybridlane documentation page. hybridlane is a frontend library for expressing hybrid [1]_
CV-DV quantum circuits in Pennylane. It implements the paper "Hybrid Oscillator-Qubit Quantum Processors:
Instruction Set Architectures, Abstract Machine Models, and Applications"[yliu]_.

What does it do?
----------------

Our package provides several features we're quite proud of:

📃 **Hybrid gates** The library has defined hybrid gates present in the paper, along with semantics of those gates. Unlike other libraries, the gate definitions are entirely independent of the way you would simulate them (i.e. our gates don't define their truncated matrix representations). This means describing a hybrid circuit is fast, even for very large circuits (we tried up to 10k qumodes 🚀).

⚛️ **Native qumode support** A wire can represent a qubit or a qumode without any ✨trickery✨ (e.g. interpreting a list of qubits as "qumode"). Additionally, the types of wires are inferred automatically by statically analyzing the circuit structure, meaning the user doesn't have to manually type each wire, making the process rather seamless.

🤝 **Pennylane compatibility** This library is compatible with Pennylane [2]_ and should be familiar to existing users. You can use Pennylane gates and operators, define devices that use hybrid CV-DV gates (either for simulators or actual hardware), write decomposition passes (transforms) to transpile hybrid circuits, and perform resource estimation. Furthermore, in the hybrid paradigm, you are free to mix and match qubits and qumodes in the same circuit, whereas in Pennylane or StrawberryFields you must stick to one or the other.

💻 **Simulator** We provide a classical simulation device that dispatches the computation to `Bosonic Qiskit <https://github.com/C2QA/bosonic-qiskit>`_. This can be used to test small circuits or serve as a reference to build your own device.

💾 **Intermediate representation** hybridlane provides an intermediate representation based on OpenQASM, with modifications to handle the more complex semantics arising from CV-DV computation.



What does it not do?
--------------------

❌ **Quantum error correction** While error correction circuits can be described using this package, we did not intend for it to handle fault-tolerant programs and resource estimation like you'd find in `Qualtran <https://github.com/quantumlib/Qualtran>`_. This package sticks to the circuit model of quantum computing.

❌ **Catalyst support** Currently, we don't have support for Catalyst and ``qjit``, as this would require developing our own dialect of MLIR. This might be a feature in the future, but it's not currently in the roadmap.

❌ **Autodiff** While our gate definitions are compatible with Pennylane's differentiability, we don't provide a differentiable simulator device, nor do we have the gradient recipes defined. This might be another feature added in the future, but for now the ``hybrid.bosonicqiskit`` device will require finite differences.

.. [1] This definition of "hybrid" differs from the usual quantum/classical paradigm. Here, hybrid means the circuit consists of both qubits and qumodes.

.. [2] We did have to redefine some things like ``qml.expval`` and ``qml.var``, so when possible, use our version. Mid-circuit measurements are also a work in progress.

.. [yliu] Y. Liu *et al*, 2024. (`arXiv <https://arxiv.org/abs/2407.10381>`_)



Installation
------------
In the future, this package will be available on PyPI and will be installable with

.. code-block:: bash

    pip install hybridlane

But for now, it must be manually installed by cloning with Github:

.. code-block:: bash

    git clone https://www.github.com/pnnl/hybridlane
    pip install ./hybridlane[extras]

The available extra flags are:

- ``bq``: Adds support for the ``hybrid.bosonicqiskit`` device.

For more detailed instructions, see :doc:`getting-started`.




.. toctree::
    :maxdepth: 2
    :caption: Using Hybridlane
    :hidden:

    introduction
    getting-started
    static-analysis
    jaqal-export

.. toctree::
    :maxdepth: 1
    :caption: API Reference
    :hidden:

    hqml <_autoapi/hybridlane/index>
    hqml.devices <_autoapi/hybridlane/devices/index>
    hqml.measurements <_autoapi/hybridlane/measurements/index>
    hqml.ops <_autoapi/hybridlane/ops/index>
    hqml.sa <_autoapi/hybridlane/sa/index>
..    _autoapi/hybridlane/util/index
..    api/hybridlane/transforms/index
