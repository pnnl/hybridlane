# hybridlane

hybridlane is a frontend library for expressing hybrid CV-DV quantum circuits in Pennylane. It (mostly) implements the paper Y. Liu *et al*, 2024. ([arXiv](https://arxiv.org/abs/2407.10381))

⚠️ **This package is in early preview.** Expect bugs and give us feedback in the Github Issues to help improve it.


## What does it do?

Our package provides several features we're quite proud of:

📃 **Hybrid gates** The library has defined hybrid gates present in the paper, along with semantics of those gates. Unlike other libraries, the gate definitions are entirely independent of the way you would simulate them (i.e. our gates don't define their truncated matrix representations). This means describing a hybrid circuit is fast, even for very large circuits (we tried up to 10k qumodes 🚀).

⚛️ **Native qumode support** A wire can represent a qubit or a qumode without any ✨trickery✨ (e.g. interpreting a list of qubits as "qumode"). Additionally, the types of wires are inferred automatically by statically analyzing the circuit structure, meaning the user doesn't have to manually type each wire, making the process rather seamless.

🤝 **Pennylane compatibility** This library is compatible with Pennylane and should be familiar to existing users. You can use Pennylane gates and operators, define devices that use hybrid CV-DV gates (either for simulators or actual hardware), write decomposition passes (transforms) to transpile hybrid circuits, and perform resource estimation. Furthermore, in the hybrid paradigm, you are free to mix and match qubits and qumodes in the same circuit, whereas in Pennylane or StrawberryFields you must stick to one or the other.

💻 **Simulator** We provide a classical simulation device that dispatches the computation to [Bosonic Qiskit](https://github.com/C2QA/bosonic-qiskit). This can be used to test small circuits or serve as a reference to build your own device.

💾 **Intermediate representation** hybridlane provides an intermediate representation based on OpenQASM, with modifications to handle the more complex semantics arising from CV-DV computation.



## What does it not do?

❌ **Quantum error correction** While error correction circuits can be described using this package, we did not intend for it to handle fault-tolerant programs and resource estimation like you'd find in [Qualtran](https://github.com/quantumlib/Qualtran). This package sticks to the circuit model of quantum computing.

❌ **Catalyst support** Currently, we don't have support for Catalyst and ``qjit``, as this would require developing our own dialect of MLIR. This might be a feature in the future, but it's not currently in the roadmap.

❌ **Autodiff** While our gate definitions are compatible with Pennylane's differentiability, we don't provide a differentiable simulator device, nor do we have the gradient recipes defined. This might be another feature added in the future, but for now the ``hybrid.bosonicqiskit`` device will require finite differences.


## Installation

In the future, this package will be available on PyPI and will be installable with

```bash
pip install hybridlane
```

But for now, it must be manually installed by cloning with Github:

```bash
git clone https://www.github.com/pnnl/hybridlane
pip install ./hybridlane[extras]
```

The available extra flags are:

- ``bq``: Adds support for the ``hybrid.bosonicqiskit`` device.

For more detailed instructions, see the documentation.
