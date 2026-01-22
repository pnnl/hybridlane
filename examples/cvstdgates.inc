// Hybrid gate standard library

// Position measurement x
defcal measure_x qumode m -> float {}

// Fock measurement n
defcal measure_n qumode m -> uint {}

// Phase-space rotation gate
gate cv_r(θ) qumode m {}

// Displacement gate
gate cv_d(r, ϕ) qumode m {}

// Squeezing gate
gate cv_sq(r, φ) qumode m {}

// Kerr gate
gate cv_k(κ) qumode m {}

// Cubic-phase gate
gate cv_c(r) qumode m {}

// Fourier gate
gate cv_f qumode m { cv_r(π/2) m; }

// Beamsplitter gate
gate cv_bs(θ, φ) qumode m1, qumode m2 {}

// Two-mode squeezing gate
gate cv_tms(r, φ) qumode m1, qumode m2 {}

// Two-mode sum gate
gate cv_sum(λ) qumode m1, qumode m2 {}

// Conditioned-rotation gate
gate cv_cr(θ) qubit q, qumode m { negctrl @ cv_r(θ) q, m; ctrl @ inv @ cv_r(θ) q, m; }

// Conditioned-displacement gate
gate cv_cd(r, ϕ) qubit q, qumode m { negctrl @ cv_d(r, ϕ) q, m; ctrl @ inv @ cv_d(r, ϕ) q, m; }

// Conditioned-squeezing gate
gate cv_cs(r, φ) qubit q, qumode m { negctrl @ cv_sq(r, φ) q, m; ctrl @ inv @ cv_sq(r, φ) q, m; }

// Conditioned-parity gate
gate cv_cp qubit q, qumode m { cv_cr(π) q, m; }

// Selective-qubit rotation (SQR) gate
gate cv_sqr(θ, φ, n) qubit q, qumode m {}

// Selective Number-dependent Arbitrary Phase (SNAP) gate
gate cv_snap(φ, n) qubit q, qumode m { cv_sqr(π, φ, n) q, m; cv_sqr(-π, φ, n) q, m; }

// Swap gate
gate cv_swap qumode m1, qumode m2 { cv_bs(π, 0) m1, m2; cv_r(-π/2) m1; cv_r(-π/2) m2; }

// Jaynes-Cummings gate
gate cv_jc(θ, φ) qubit q, qumode m {}

// Anti-Jaynes-Cummings gate
gate cv_ajc(θ, φ) qubit q, qumode m {}

// Conditioned-beamsplitter gate
gate cv_cbs(θ, φ) qubit q, qumode m1, qumode m2 { negctrl @ cv_bs(θ, φ) q, m1, m2; ctrl @ inv @ cv_bs(θ, φ) q, m1, m2; }

// Conditioned-two mode squeezing gate
gate cv_ctms(r, φ) qubit q, qumode m1,  qumode m2 { negctrl @ cv_tms(r, φ) q, m1, m2; ctrl @ inv @ cv_tms(r, φ) q, m1, m2; }

// Conditioned-two-mode sum gate
gate cv_csum(λ) qubit q, qumode m1, qumode m2 { negctrl @ cv_sum(λ) q, m1, m2; ctrl @ inv @ cv_sum(λ) q, m1, m2; }
