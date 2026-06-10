# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import jax
import numpy as np
import pennylane as qp
import pytest
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.exceptions import DeviceError

import hybridlane as hl
from hybridlane.devices import DefaultHybrid
from hybridlane.devices.default_hybrid.device import (
    _get_wire_dims,
    is_analytic_mp_supported,
    is_analytic_observable_supported,
    is_sampled_mp_supported,
    is_sampled_observable_supported,
)
from hybridlane.measurements import ComputationalBasis
from hybridlane.sa import BasisSchema


@pytest.mark.unit
def test_importable():
    dev = qp.device("default.hybrid")
    assert isinstance(dev, DefaultHybrid)


@pytest.mark.unit
class TestWireDims:
    def test_only_one_truncation_argument_provided(self):
        # `fock_level` or `wire_dims` must be provided
        dev = qp.device("default.hybrid")
        with pytest.raises(
            DeviceError,
            match="Exactly one of 'wire_dims' or 'fock_level' must be specified",
        ):
            dev.setup_execution_config()

        # Providing both is a problem
        dev = qp.device("default.hybrid", fock_level=8, wire_dims={2: 2, 3: 3})
        with pytest.raises(
            DeviceError,
            match="Exactly one of 'wire_dims' or 'fock_level' must be specified",
        ):
            dev.setup_execution_config()

    def test_inferred_from_fock_level(self):
        def circuit():
            qp.Z("a")
            hl.D(0.123, 0, "b")
            return hl.expval(hl.N("b"))

        qnode = qp.QNode(circuit, qp.device("default.hybrid", fock_level=8))
        tape = qp.workflow.construct_tape(qnode)()
        config = qp.workflow.construct_execution_config(qnode, resolve=True)()  # ty:ignore[missing-argument]
        assert config.device_options["fock_level"] == 8
        wire_dims = _get_wire_dims(tape, config)
        assert wire_dims == {0: 2, 1: 8}

    def test_remaps_wire_dims(self):
        def circuit():
            qp.Z("a")
            hl.D(0.123, 0, "b")
            return hl.expval(hl.N("b"))

        qnode = qp.QNode(
            circuit, qp.device("default.hybrid", wire_dims={"a": 2, "b": 8})
        )
        tape = qp.workflow.construct_tape(qnode)()
        config = qp.workflow.construct_execution_config(qnode, resolve=True)()  # ty:ignore[missing-argument]
        assert config.device_options["wire_dims"] == {"a": 2, "b": 8}
        wire_dims = _get_wire_dims(tape, config)
        assert wire_dims == {0: 2, 1: 8}


@pytest.mark.unit
class TestMeasurementSupport:
    def test_supports_analytic_mp(self):
        for func in (hl.expval, hl.var):
            assert all(
                is_analytic_mp_supported(func(op))
                for op in [
                    hl.X(0),
                    hl.P(0),
                    hl.N(0),
                    0.5 * hl.N(0),
                    0.123 * qp.X(0) + 0.456 * hl.P(1),
                    hl.X(0) @ hl.P(0) + hl.P(0) @ hl.X(0),
                ]
            )

        assert is_analytic_mp_supported(hl.state())
        assert is_analytic_mp_supported(hl.density_matrix())
        assert not is_analytic_mp_supported(hl.sample(hl.N(0)))

    def test_supports_analytic_observable(self):
        # Various symbolic observables
        assert all(
            map(
                is_analytic_observable_supported,
                [
                    hl.X(0),
                    hl.P(0),
                    hl.N(0),
                    0.5 * hl.N(0),
                    0.123 * qp.X(0) + 0.456 * hl.P(1),
                    hl.X(0) @ hl.P(0) + hl.P(0) @ hl.X(0),
                ],
            )
        )

        # Mid-circuit measurement values
        assert is_analytic_observable_supported(qp.measure(0))

        # Matrix-based observables
        obs = qp.Hermitian(np.array([[0, 1], [1, 0]]), wires=0)
        assert is_analytic_observable_supported(obs)

    def test_supports_sampled_mp(self):
        assert all(
            is_sampled_mp_supported(hl.expval(obs))
            for obs in [
                hl.N(0),
                0.5 * hl.N(0),
                hl.N(1) ** 2,
                0.123 * hl.N(0) + 0.456 * hl.N(1),
                qp.Z(0) @ hl.N(1),
                qp.X(0) @ qp.Z(1),
                hl.FockStateProjector(2, wires=0),
            ]
        )

        for func in (hl.var, hl.sample):
            assert all(
                is_sampled_mp_supported(func(obs))
                for obs in [
                    hl.N(0),
                    0.5 * hl.N(0),
                    hl.N(1) ** 2,
                    qp.Z(0) @ hl.N(1),
                    qp.X(0) @ qp.Z(1),
                    hl.FockStateProjector(2, wires=0),
                ]
            )

            assert not any(
                is_sampled_mp_supported(func(obs))
                for obs in [
                    0.123 * hl.N(0) + 0.456 * hl.N(1),
                    hl.X(0),
                    hl.P(0),
                ]
            )

        assert is_sampled_mp_supported(
            hl.sample(schema=BasisSchema({0: ComputationalBasis.Discrete}))
        )

        assert not is_sampled_mp_supported(hl.state())
        assert not is_sampled_mp_supported(
            hl.sample(schema=BasisSchema({0: ComputationalBasis.Position}))
        )

    @pytest.mark.xfail(reason="expval doesn't support mcm")
    def test_supports_sampled_mp_mcm(self):
        # fixme(mcm): with this test, the call to `hl.expval` fails when given a
        # MeasurementValue. should review why we disabled it and if we can just drop the
        # failure inside expval
        assert is_sampled_mp_supported(hl.expval(qp.measure(0)))

    @pytest.mark.xfail(reason="qp.Hermitian is polymorphic")
    def test_supports_sampled_mp_hermitian(self):
        # todo(polymorphic): by itself, hybridlane is unable to infer the wire types of
        # qp.Hermitian. We could restrict them to just qubits, but hybridlane could in
        # principle infer the types if given context from the circuit. Technically, Hermitian
        # is polymorphic and this should be revisited in the future
        assert is_sampled_mp_supported(
            hl.expval(qp.Hermitian(np.array([[0, 1], [1, 0]]), wires=0))
        )

    def test_supports_sampled_observable(self):
        assert all(
            map(
                lambda op: is_sampled_observable_supported(op, is_expval=True),
                [
                    hl.N(0),
                    0.5 * hl.N(0),
                    hl.N(1) ** 2,
                    0.123 * hl.N(0) + 0.456 * hl.N(1),
                    qp.Z(0) @ hl.N(1),
                    qp.X(0) @ qp.Z(1),
                    hl.FockStateProjector(2, wires=0),
                    qp.measure(0),
                    qp.Hermitian(np.array([[0, 1], [1, 0]]), wires=0),
                ],
            )
        )

        assert not any(
            map(
                lambda op: is_sampled_observable_supported(op, is_expval=True),
                [
                    hl.X(0),
                    hl.N(0) @ hl.X(0),
                    hl.X(0) ** 2 + hl.P(0) ** 2,
                    qp.Z(0) @ hl.X(1),
                ],
            )
        )

    # todo: remove in v0.9.0
    def test_operator_batching_isnt_supported(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @qp.qnode(dev)
        def circuit():
            hl.Displacement([0.1, 0.2], 0, wires=0)
            return hl.expval(hl.X(0))

        with pytest.raises(DeviceError, match="Operator batching is not supported"):
            circuit()

    # todo: remove in v0.9.0
    def test_mcm_isnt_supported(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @qp.set_shots(10)
        @qp.qnode(dev, mcm_method="one-shot")
        def circuit():
            qp.H(0)
            mv = qp.measure(0)
            qp.cond(mv, hl.D)(0.123, 0, wires=1)
            return hl.expval(hl.N(1))

        with pytest.raises(DeviceError):
            circuit()

    @pytest.mark.jax
    def test_sample_unsupported_with_jit(self):
        dev = qp.device("default.hybrid", fock_level=8, seed=jax.random.key(42))

        @jax.jit
        @qp.set_shots(10)
        @qp.qnode(dev, interface="jax")
        def circuit():
            hl.D(0.123, 0, wires=0)
            return hl.sample(hl.N(0))

        with pytest.raises(DeviceError, match="`jax.jit` does not support"):
            circuit()


@pytest.mark.unit
class TestExecute:
    @pytest.mark.jax
    def test_with_prng_key(self):
        import jax

        prng_key = jax.random.key(42)
        dev = qp.device("default.hybrid", fock_level=8, seed=prng_key)

        assert dev._prng_key == prng_key

        @qp.qnode(dev, interface="jax")
        def circuit():
            qp.CatState(3, 0, 0, wires=0)
            return hl.expval(hl.X(0))

        assert circuit() == pytest.approx(0)
        assert not dev._prng_key == prng_key  # check that the key was updated

    def test_with_np_rng(self):
        dev = qp.device("default.hybrid", fock_level=8, seed=42)

        assert isinstance(dev._rng, np.random.Generator)
        assert (
            dev._rng.bit_generator.state
            == np.random.default_rng(42).bit_generator.state
        )

        @qp.qnode(dev)
        def circuit():
            qp.CatState(3, 0, 0, wires=0)
            return hl.expval(hl.X(0))

        assert circuit() == pytest.approx(0)
        assert (
            dev._rng.bit_generator.state
            == np.random.default_rng(42).bit_generator.state
        )  # not updated because the circuit has no randomness

    def test_with_max_workers(self):
        dev = qp.device("default.hybrid", fock_level=8, seed=42, max_workers=1)

        rng = np.random.default_rng(42)
        assert dev._rng.bit_generator.state == rng.bit_generator.state

        @qp.batch_input(argnum=0)
        @qp.qnode(dev, executor_backend="serial")
        def circuit(alpha):
            qp.CatState(alpha, 0, 0, wires=0)
            return hl.expval(hl.N(0))

        alphas = hl.math.array([0.123, 0.456, 0.789, 1.0])
        results = circuit(alphas)

        # Sanity check that each circuit is being processed
        assert len(set(results)) == 4
        assert hl.math.argsort(results) == pytest.approx([0, 1, 2, 3])

        # generates a new rng for each worker, so state should be different
        assert dev._rng.bit_generator.state != rng.bit_generator.state


@pytest.mark.unit
class TestSupportsDerivatives:
    def test_supports_backprop(self):
        dev = qp.device("default.hybrid", fock_level=8)
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
