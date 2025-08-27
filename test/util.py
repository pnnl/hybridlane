import numpy as np
from scipy import stats


def poisson_test(samples: np.ndarray, mu: float) -> float:
    """Utility function to check that a set of samples is consistent with a poisson distribution"""

    n = samples.size
    domain, f_obs = np.unique_counts(samples)
    expected_probs = stats.poisson.pmf(domain, mu=mu)
    expected_freqs = n * (expected_probs / expected_probs.sum())

    mask = (n - np.cumsum(expected_freqs)) < 5
    if np.any(mask):
        f_obs = np.concatenate([f_obs[~mask], [f_obs[mask].sum()]])
        expected_freqs = np.concatenate(
            [expected_freqs[~mask], [expected_freqs[mask].sum()]]
        )

    return stats.chisquare(f_obs, expected_freqs).pvalue
