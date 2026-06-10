import numpy as np
import pennylane as qp
import pytest

import hybridlane as hl

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.jax
def test_binomial_state_prep():
    fock_level = 20
    dev = qp.device("default.hybrid", fock_level=fock_level)

    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(params):
        @qp.for_loop(0, 5)
        def loop_body(i):
            hl.D(params[i, 0], params[i, 1], wires=0)

            for j in range(8):
                hl.SNAP(params[i, 2 + j], j, wires=0)

        loop_body()
        return hl.density_matrix()

    # Target state is the binomial codeword |0_L> = (|0> + |4>)/sqrt(2)
    codeword = hl.math.concatenate(
        [
            hl.math.array([1, 0, 0, 0, 1], like="jax") / np.sqrt(2),
            hl.math.zeros(fock_level - 5, like="jax"),
        ]
    )
    target = hl.math.outer(codeword, codeword)

    @jax.jit
    def loss(params):
        state = circuit(params)
        return 1 - hl.math.real(hl.math.fidelity(state, target))

    # Initialize parameters randomly
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (5, 10))

    starting_loss = loss(params)

    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state

    for _ in range(100):
        params, opt_state = train_step(params, opt_state)

    final_loss = loss(params)
    assert final_loss < starting_loss


@pytest.mark.integration
@pytest.mark.jax
def test_displacement_grad():
    fock_level = 10
    dev = qp.device("default.hybrid", fock_level=fock_level)

    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(params):
        hl.D(params[0], params[1], wires=0)
        return hl.expval(hl.X(0))

    params = hl.math.array([0.123, 0], like="jax")
    grad_fn = hl.math.grad(circuit)

    grad = grad_fn(params)
    assert grad.shape == (2,)
    assert not hl.math.isnan(grad).any()

    assert grad[0] > 0
    assert grad[1] == 0
