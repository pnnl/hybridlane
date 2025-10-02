# hybridlane

[![PyPI - Version](https://img.shields.io/pypi/v/hybridlane?logo=pypi)](https://pypi.org/project/hybridlane/)
[![Docs](https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/docs.yml?branch=main&logo=githubpages&label=docs)](https://pnnl.github.io/hybridlane/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/hybridlane?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/hybridlane)
[![Build Status](https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/release.yml)](https://github.com/pnnl/hybridlane/actions/workflows/release.yml)
[![License](https://img.shields.io/github/license/pnnl/hybridlane)](LICENSE.txt)

hybridlane is a frontend library for expressing and manipulating **hybrid continuous-variable (CV) and discrete-variable (DV) quantum circuits** within the [PennyLane](https://pennylane.ai/) ecosystem. It implements the concepts introduced in the paper Y. Liu *et al*, 2024 ([arXiv:2407.10381](https://arxiv.org/abs/2407.10381)).

---

## ✨ Why hybridlane?

Quantum computing research is increasingly exploring hybrid (heterogenous) models. hybridlane empowers researchers and developers to:

*   **Design complex hybrid circuits effortlessly:** Mix and match qubits and qumodes in the same circuit without workaround.
*   **Scale to large circuits:** Define hybrid gate semantics independent of simulation, enabling fast circuit description (we tried up to 10k qumodes).
*   **Integrate with PennyLane:** Leverage existing PennyLane tools for transformations, resource estimation, and device integration.

---

## 🚀 Features

hybridlane offers functionality designed for hybrid quantum circuit development:

*   **📃 Hybrid Gate Semantics:**
    The library precisely defines hybrid gates as presented in the reference paper, along with their operational semantics. Crucially, these definitions are entirely independent of their truncated matrix representations, allowing for rapid description of large circuits.

*   **⚛️ Native Qumode Support:**
    Unlike other libraries that might interpret lists of qubits as a "qumode," hybridlane provides native support for qumodes as fundamental wire types. The types of wires (qubit or qumode) are automatically inferred through static analysis of the circuit structure, eliminating the need for manual type annotation and making circuit construction highly intuitive.

*   **🤝 PennyLane Compatibility:**
    hybridlane is built to be compatible with PennyLane. Existing PennyLane users will find the interface familiar, allowing them to utilize PennyLane gates and operators, define custom devices for hybrid CV-DV computation (for simulators or hardware), write decomposition passes (transforms) to transpile hybrid circuits, and perform resource estimation. A key differentiator is the ability to freely mix qubits and qumodes within the same circuit, a capability not natively found in PennyLane or StrawberryFields alone.

*   **💻 Classical Simulation Device:**
    We provide a classical simulation device that dispatches computations to [Bosonic Qiskit](https://github.com/C2QA/bosonic-qiskit). This device is ideal for testing small circuits or serving as a reference for building your own custom hybrid devices.

*   **💾 OpenQASM-based Intermediate Representation (IR):**
    hybridlane includes an intermediate representation based on OpenQASM, extended with modifications to handle the more complex semantics inherent in CV-DV quantum computation. This enables interoperability and advanced circuit manipulation.

---

## ⚙️ Installation

hybridlane is currently in **early preview**. We highly value your feedback on GitHub Issues to help us improve!

The package can be installed from PyPI with

```bash
pip install hybridlane[extras]
```

**Available Extra Flags:**
*   `[all]`: Installs all extra flags.
*   `[bq]`: Installs support for the `bosonicqiskit.hybrid` simulation device.

For more detailed installation instructions and environment setup, please refer to the [Getting Started Guide in our Documentation](https://pnnl.github.io/hybridlane/getting-started.html).

---

## ⚡ Quick Start

Get started with hybridlane in just a few lines of code:

```python
import numpy as np
import pennylane as qml
import hybridlane as hqml

# Create the bosonic qiskit simulator with custom Fock truncation
dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

# Define a hybrid circuit with familiar Pennylane syntax
@qml.qnode(dev)
def circuit(n):
    # Python control flow allowed
    for j in range(n):
        # You can use existing Pennylane gates.
        qml.X(0) # wire `0` inferred to be a qubit

        # Or use the hybrid CV-DV gates in Hybridlane
        # (!) Qubits come before qumodes, allowing m0 to be inferred as a qumode
        hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, "m0"])

    # Freely mix qubit and qumode observables
    # (!) We use `hqml` for measurements and for CV observables
    return hqml.expval(hqml.NumberOperator("m0") @ qml.Z(0))

# Execute the circuit
expval = circuit(5)
# array(5.)

# Or analyze its structure
import hybridlane.sa as sa
res = sa.analyze(circuit._tape)
# StaticAnalysisResult(qumodes=Wires(['m0']), qubits=Wires([0]), schemas=[<hybridlane.sa.base.BasisSchema object at 0x7f504673a090>], wire_order=Wires([0, 'm0']))
```

For more examples and detailed usage, explore the [Documentation](https://pnnl.github.io/hybridlane/).

---

## 🚧 Limitations & Future Work

While hybridlane provides a powerful framework for hybrid quantum circuits, it's currently under active development and has some known limitations:

*   **❌ Quantum Error Correction (QEC):**
    While QEC circuits can be *described* using this package, hybridlane is not designed to handle fault-tolerant programs or extensive resource estimation like specialized libraries such as [Qualtran](https://github.com/quantumlib/Qualtran). Our focus remains on the circuit model of quantum computing.

*   **❌ Catalyst/JIT Compilation Support:**
    Currently, hybridlane does not support PennyLane's Catalyst and `qjit` capabilities. This would require developing a custom MLIR dialect for hybrid operations, which is a significant undertaking. While not on the immediate roadmap, this could be a potential feature in the future.

*   **❌ Automatic Differentiation (Autodiff):**
    While our gate definitions are fundamentally compatible with PennyLane's differentiability paradigm, we do not yet provide a differentiable simulator device, nor are gradient recipes fully defined for all hybrid operations. For now, the `bosonicqiskit.hybrid` device will typically require finite differences for gradient computation. We plan to integrate more robust autodiff capabilities in future releases.

---

## 📚 Documentation

For comprehensive information on hybridlane's API, detailed usage examples, theoretical background, and contributing guidelines, please visit our official [Documentation](https://pnnl.github.io/hybridlane/).

---

## ❓ Support & Feedback

Having trouble? Found a bug? Have a great idea for a new feature?

*   For technical questions, bug reports, or feature requests, please open an issue on our [GitHub Issues page](https://github.com/pnnl/hybridlane/issues).

---

## 📜 License

This project is licensed under a permissive open-source license - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🙏 Acknowledgements

This project was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research program under contract number DE-FOA-0003265.