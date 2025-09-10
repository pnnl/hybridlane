// Hybrid gate standard library
// 
// This library should be imported after declaring the constants `homodyne_precision_bits`, `fock_readout_bits`
defcal measure_x m -> float[homodyne_precision_bits] {}

// Fock measurement n
defcal measure_n m -> uint[fock_readout_precision_bits] {}

// Phase-space rotation gate
gate cv_r(θ) m {}

// Displacement gate
gate cv_d(r, ϕ) m {}

// Squeezing gate
gate cv_sq(r, φ) m {}

// Kerr gate
gate cv_k(κ) m {}

// Cubic-phase gate
gate cv_p3(γ) m {}

// Fourier gate
gate cv_f m { cv_r(π/2) m; }

// Beamsplitter gate
gate cv_bs(θ, φ) m1, m2 {}

// Two-mode squeezing gate
gate cv_sq2(r, φ) m1, m2 {}

// Two-mode sum gate
gate cv_tms(λ) m1, m2 {}

// Conditioned-rotation gate
gate cv_cr(θ) q, m { negctrl @ cv_r(θ) q, m; ctrl @ inv @ cv_r(θ) q, m; }

// Conditioned-displacement gate
gate cv_cd(r, ϕ) q, m { negctrl @ cv_d(r, ϕ) q, m; ctrl @ inv @ cv_d(r, ϕ) q, m; }

// Conditioned-parity gate
gate cv_cp q, m { cv_cr(π) q, m; }

// Selective-qubit rotation (SQR) gate
gate cv_sqr(θ, φ, n) q, m {}

// Selective Number-dependent Arbitrary Phase (SNAP) gate
gate cv_snap(φ, n) q, m { cv_sqr(π, φ, n) q, m; cv_sqr(-π, φ, n) q, m; }

// Swap gate
gate cv_swap m1, m2 { cv_bs(π, π/2) m1, m2; }

// Jaynes-Cummings gate
gate cv_jc(θ, φ) q, m {}

// Anti-Jaynes-Cummings gate
gate cv_ajc(θ, φ) q, m {}

// Conditioned-beamsplitter gate
gate cv_cbs(θ, φ) q, m1, m2 { negctrl @ cv_bs(θ, φ) q, m1, m2; ctrl @ inv @ cv_bs(θ, φ) q, m1, m2; }

// Conditioned-two mode squeezing gate
gate cv_csq2(r, φ) q, m1, m2 { negctrl @ cv_sq2(r, φ) q, m1, m2; ctrl @ inv @ cv_sq2(r, φ) q, m1, m2; }

// Conditioned-two-mode sum gate
gate cv_ctms(λ) q, m1, m2 { negctrl @ cv_tms(λ) q, m1, m2; ctrl @ inv @ cv_tms(λ) q, m1, m2; }
