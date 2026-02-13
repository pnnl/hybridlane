<p align="center">
  <a href="https://pypi.org/project/hybridlane/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/hybridlane?logo=pypi"></a>
  <a href="https://pnnl.github.io/hybridlane/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/docs.yml?branch=main&logo=githubpages&label=docs"></a>
  <a href="https://pepy.tech/projects/hybridlane"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/hybridlane?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
  <a href="https://github.com/pnnl/hybridlane/actions/workflows/release.yml"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/release.yml"></a>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/github/license/pnnl/hybridlane"></a>
</p>

<img src="./docs/source/_static/draw_mpl/qpe_circuit.png" alt="hybridlane logo" width="500" align="center">

<h1 align="center">hybridlane</h1>

**hybridlane** is a Python library for designing and manipulating **hybrid continuous-variable (CV) and discrete-variable (DV) quantum circuits** within the [PennyLane](https://pennylane.ai/) ecosystem. It provides a frontend for expressing hybrid quantum algorithms, implementing the concepts from the paper Y. Liu *et al*, 2026 ([PRX Quantum 7, 010201](https://doi.org/10.1103/4rf7-9tfx)).

---

## ✨ Why hybridlane?

As quantum computing explores beyond traditional qubit-only models, `hybridlane` offers a powerful and intuitive framework for researchers and developers to:

*   **Design complex hybrid circuits effortlessly:** Seamlessly integrate qubits and qumodes in the same circuit.
*   **Describe large-scale circuits:** Define hybrid gate semantics independently of simulation, enabling fast description of wide and deep circuits with minimal memory.
*   **Leverage the PennyLane ecosystem:** Integrate with PennyLane's extensive tools for transformations, resource estimation, and device support.

---

## 🚀 Features

*   **📃 Hybrid Gate Semantics:** Precise, platform-independent definitions for hybrid gates, enabling rapid construction of large-scale quantum circuits.

*   **⚛️ Native Qumode Support:** Qumodes are treated as a fundamental wire type, with automatic type inference that simplifies circuit construction and enhances readability.

*   **🤝 PennyLane Compatibility:** A familiar interface for PennyLane users. Utilize existing PennyLane gates, build custom hybrid devices, write compilation passes, and perform resource estimation across mixed-variable systems.

*   **💻 Classical Simulation:** A built-in device that dispatches to [Bosonic Qiskit](https://github.com/C2QA/bosonic-qiskit) for simulating small hybrid circuits.

*   **💾 OpenQASM-based IR:** An intermediate representation based on an extended OpenQASM, promoting interoperability and enabling advanced circuit manipulations.

---

## ⚙️ Installation

`hybridlane` is currently in **early preview**. We welcome your feedback on our [GitHub Issues](https://github.com/pnnl/hybridlane/issues) page to help us improve.

Install the package from PyPI:
```bash
pip install hybridlane
```

**Available Extras:**
*   `[all]`: Installs all extra dependencies.
*   `[bq]`: Installs support for the `bosonicqiskit.hybrid` simulation device.
*   `[qscout]`: Installs support for the `sandiaqscout.hybrid` compilation device.

For detailed instructions, see the [Getting Started Guide](https://pnnl.github.io/hybridlane/getting-started.html) in our documentation.

---

## ⚡ Quick Start

```python
import numpy as np
import pennylane as qml
import hybridlane as hqml

# Create a bosonic qiskit simulator with a custom Fock truncation
dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

# Define a hybrid circuit with familiar PennyLane syntax
@qml.qnode(dev)
def circuit(n):
    for j in range(n):
        qml.X(0)  # Wire `0` is inferred to be a qubit
        # Use hybrid CV-DV gates from hybridlane
        hqml.JC(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, "m"])

    # Mix qubit and qumode observables
    return hqml.expval(hqml.N("m") @ qml.Z(0))

# Execute the circuit
expval = circuit(5)
# array(5.)

# Analyze its structure
import hybridlane.sa as sa
res = sa.analyze(circuit._tape)
print(res)
# StaticAnalysisResult(qumodes=Wires(['m']), qubits=Wires([0]), schemas=[...])
```

For more examples, explore our [Documentation](https://pnnl.github.io/hybridlane/).

---

## 🗺️ Roadmap

`hybridlane` is under active development. Here are some of our future goals:

*   **Broader measurement support:** Including mid-circuit measurements and broader measurement capabilities.
*   **Algorithms and transformations:** Implementing popular algorithms and circuit transformations from research papers, including dynamic qumode allocation.
*   **Symbolic Hamiltonians:** Introducing support for symbolic bosonic Hamiltonians.
*   **Noisy simulation:** Supporting noisy simulations with Bosonic Qiskit.
*   **Pulse-level gates:** Allowing pulse-level gates and simulating them in Dynamiqs.
*   **Catalyst/QJIT support:** Integrating with PennyLane's `qjit` capabilities by developing a custom MLIR dialect.
*   **Community-driven features:** Incorporating features requested by the community during usage.

---

## 📚 Documentation

For comprehensive information on `hybridlane`'s API, tutorials, and technical background, please visit our official [Documentation](https://pnnl.github.io/hybridlane/).

---

## ❓ Support

For questions, bug reports, or feature requests, please open an issue on our [GitHub Issues page](https://github.com/pnnl/hybridlane/issues).

---

## Citing hybridlane

If you use `hybridlane` in your research, please cite our work:

```
under preparation, check back soon :)
```

---

## 📜 License

This project is licensed under the BSD 2-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🙏 Acknowledgements

This project was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research program under contract number DE-FOA-0003265.
