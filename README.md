<!--
SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
SPDX-License-Identifier: BSD-2-Clause
-->

<p align="center">
    <img src="./docs/source/_static/draw_mpl/qpe_circuit.png" alt="hybridlane banner" width="80%">
</p>

<h1 align="center">hybridlane</h1>

**hybridlane** is a Python library for designing and manipulating **hybrid continuous-variable (CV) and discrete-variable (DV) quantum circuits** within the [PennyLane](https://pennylane.ai/) ecosystem. It provides a frontend for expressing hybrid quantum algorithms, implementing the concepts from the paper Y. Liu _et al_, 2026 ([PRX Quantum 7, 010201](https://doi.org/10.1103/4rf7-9tfx)).

<p align="center">
  <a href="https://pypi.org/project/hybridlane/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/hybridlane?logo=pypi"></a>
  <a href="https://pnnl.github.io/hybridlane/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/docs.yml?branch=main&logo=githubpages&label=docs"></a>
  <a href="https://pepy.tech/projects/hybridlane"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/hybridlane?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
  <a href="https://github.com/pnnl/hybridlane/actions/workflows/release.yml"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/pnnl/hybridlane/release.yml"></a>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/github/license/pnnl/hybridlane"></a>
</p>

## 🚀 Features

- **⚛️ Heterogeneous quantum circuits:** Mix qubits and qumodes in the same circuit, and use our symbolic hybrid gate library to scalably build quantum algorithms.

- **🤝 PennyLane compatibility:** Utilize existing PennyLane gates, write compilation passes as transforms, build custom hybrid backends for hardware, and perform resource estimation across mixed-variable systems.

- **💻 Classical simulation:** Dispatch to our Jax-compatible simulator for accelerated CPU and GPU simulation and take gradients using automatic differentiation, or use [Bosonic Qiskit](https://github.com/C2QA/bosonic-qiskit).

- **💾 OpenQASM-based IR:** Leverage our intermediate representation extending OpenQASM to reduce the effort of building new hybrid backends and to facilitate interoperability with other quantum software.

---

## ⚙️ Installation

Install the package from PyPI:

```bash
pip install hybridlane
```

For more details on installation and optional dependencies, see the [installation guide](https://pnnl.github.io/hybridlane/getting-started.html).

> [!WARNING]
> `hybridlane` is currently in active development and may experience breaking changes -- consider using version pinning. We welcome your feedback on our [GitHub Issues](https://github.com/pnnl/hybridlane/issues) page to help us improve the software.

---

## ⚡ Quick Start

```python
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
```

For more examples, explore the [documentation](https://pnnl.github.io/hybridlane/).

---

## 🗺️ Roadmap

`hybridlane` is under active development. Here are some of our future goals:

- **Broader measurement support:** Including mid-circuit measurements and broader measurement capabilities.
- **Algorithms and transformations:** Implementing popular algorithms and circuit transformations from research papers, including dynamic qumode allocation.
- **Symbolic Hamiltonians:** Introducing support for symbolic bosonic Hamiltonians.
- **Noisy simulation:** Supporting noisy quantum simulations, possibly with Dynamiqs.
- **Catalyst/QJIT support:** Integrating with PennyLane's `qjit` capabilities by developing a custom MLIR dialect.
- **Community-driven features:** Incorporating features requested by the community during usage.

---

## Citing hybridlane

If you find `hybridlane` useful in your research, you can cite our paper:

```
@misc{furches2026hybridlane,
      title={Hybridlane: A Software Development Kit for Hybrid Continuous-Discrete Variable Quantum Computing},
      author={Jim Furches and Timothy J. Stavenger and Carlos Ortiz Marrero},
      year={2026},
      eprint={2603.10919},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.10919},
}
```

---

## 📜 License

This project is licensed under the BSD 2-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🙏 Acknowledgements

This project was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research program under contract number DE-FOA-0003265.
