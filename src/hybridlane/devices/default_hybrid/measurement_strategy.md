I think we will restrict ourselves to an observable that follows this form:

  <observable> ::= op | <term> | <sum>
  <term> ::= op | <prod> | <symb>
  <prod> ::= Prod((op | <symb>){2+})
  <symb> ::= ScalarSymbolicOp(op) = Pow(op) | SProd(op)
  <sum> ::= Sum(<term>{2+})
  op ::= any non-symbolic or composite Hermitian operator

This ensures that our observable, if it has a sum, is only the outermost operation
in the expression tree and that inner terms are not sums.

Additionally each op has some important properties:

- `diag (bool)`: Whether or not we can know how to diagonalize the operator into the
  Fock + Z basis symbolically

- `eigvals (Tensor?)`: The eigenvalues of the operator as a 1D tensor, only available
  if `diag` is true. Operators that have `eigvals` defined know their eigenvalues
  without computing them numerically through `math.linalg.eigvalsh` or similar functions.

- `matrix (Tensor)`: The matrix representation of the operator in the Fock + Z basis.
  Every base `op` operator that we'll be working with has this property, but as it can be
  expensive, this will be a fallback for operators that don't have `diag` set to true.

Finally, we have references to `state` and `probs`. `probs` is just `math.abs(state) ** 2`.

### Measurement strategies

**expval**

Returns the expectation value of the observable. Can be analytic or finite-shot.

  `op`: If `diag` is true, we can rotate the state into the eigenbasis of `op` using the
  diagonalizing gates of `op`, then compute `dot(op.eigvals, probs)`. If false, we resort
  to `state.T @ op.matrix @ state`, which fails for finite-shot mode.

  `SProd(c, op)`: `c * expval(op)`

  `Pow(z, op)`: If `op.diag` is true, we compute `dot(op.eigvals ** z, probs)`. If false, we 
  compute `state.T @ (op.matrix ** z) @ state`, which fails for finite-shot mode.

  `Sum(ops)`: `sum(expval(op) for op in ops)`

  `Prod(ops)`: If it's a tensor product (`has_overlapping_wires = False`) and all terms
  have `diag` set to true, we diagonalize each subsystem independently and then compute
  `e = reduce(math.kron(eigvals))`. Then we compute `dot(e, probs)`. If it's not a tensor product or if any term has `diag` set to false, we compute the matrix representation of the product and then compute `state.T @ prod.matrix @ state`, which fails for finite-shot mode.

**var**

Returns the variance of the observable. Can be analytic or finite-shot

  `op`: If `diag` is true, we can compute `dot(op.eigvals ** 2, probs) - dot(op.eigvals, probs) ** 2`. If false, we compute `expval(op^2) - expval(op) ** 2`, which probably fails for finite-shot mode.

  `SProd(c, op)`: `c ** 2 * var(op)`

  `Pow(z, op)`: If `op.diag` is true, we compute `e = op.eigvals ^ z` and fall back to
  `dot(e ** 2, probs) - dot(e, probs) ** 2`. If false, we compute `expval(op^(2z)) - expval(op^z) ** 2`, which probably fails for finite-shot mode.

  `Sum(ops)`: For now we should probably just do `expval(sum(ops ** 2)) - expval(sum(ops)) ** 2`, which fails for finite-shot mode. We might be able to optimize this later

  `Prod(ops)`: If it's a tensor product, then we can individually diagonalize each term
  and then compute `e = reduce(math.kron(eigvals))` and then compute `dot(e ** 2, probs) - dot(e, probs) ** 2`. If it's not a tensor product, we can compute the matrix representation of the product and then compute `expval(prod^2) - expval(prod) ** 2`, which fails for finite-shot mode.
  
**sample**

Returns samples of the eigenvalues of the observable. Explicitly finite shot mode.

  We require that `diag` is always true.

  `op`: Diagonalize the state and then sample `eigvals` based on `probs`.

  `SProd(c, op)`: Sample from `op` and then multiply the resulting eigenvalues by `c`.

  `Pow(z, op)`: Sample from `op` and then raise the resulting eigenvalues to the power of `z`.

  `Sum(ops)`: Probably just fail for now, but technically doable if all terms commute.

  `Prod(ops)`: If it's a tensor product, we can individually diagonalize each term and then sample from the resulting eigenvalues, computed as `e = reduce(math.kron(...))` based on the probabilities. If it's not a tensor product, then fail for now, but it follows similar logic as sum.

**probs**
If `obs` is None, then this just returns `|state|^2`. Otherwise, it diagonalizes the state
in the eigenbasis of the observable and then returns `|state|^2` in that basis, which corresponds to the probabilities of measuring each eigenvalue of the observable. This is only well-defined if `diag` is true for the observable.
