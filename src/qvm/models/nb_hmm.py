"""Negative-Binomial Hidden Markov Model, fit by EM — written from scratch.

WHY from scratch: no maintained Python library offers a trustworthy NB-HMM
(hmmlearn has Gaussian/Poisson-family only), and patent counts are
overdispersed — Poisson's variance=mean assumption does not hold — so NB
emissions are the spec and the implementation must be ours, testable.

Numerical discipline (non-negotiable, see tests/test_hmm_synthetic.py):
  * forward-backward runs entirely in LOG SPACE (logsumexp), no underflow;
  * 20 random EM restarts, best log-likelihood kept, spread reported;
  * LABEL-SWITCHING GUARD: after fitting, states are reordered by emission
    mean so state 0 = low tempo, state 1 = high tempo — without this,
    cross-company comparison is meaningless.

The model emits BOTH posteriors:
  filtered  P(S_t | y_1:t)  — causal: computable in real time at t;
  smoothed  P(S_t | y_1:T)  — retrospective, uses the whole series.

HONESTY CAVEAT (stamped into outputs too): even the *filtered* probabilities
come from parameters estimated on the FULL sample — future leaks through the
parameters. True out-of-sample requires expanding-window re-estimation, which
is Step 5 validation work. This baseline documents the leak, it does not fix
it.

Emission families: 'nb' (state-dependent mean, SHARED dispersion r — parameter
parsimony with ~80 observations) and 'poisson' (the "was NB necessary?"
comparison). Means have a closed-form M-step (the weighted mean is the exact
MLE for both families); NB's r has no closed form and is optimized numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

_MU_FLOOR = 1e-6
_PROB_FLOOR = 1e-10


# --------------------------------------------------------------------------- #
# emission log-pmfs (vectorized over observations)
# --------------------------------------------------------------------------- #
def nb_logpmf(y: np.ndarray, mu: float, r: float) -> np.ndarray:
    """NB in mean/dispersion form: E=mu, Var=mu + mu^2/r (r->inf = Poisson)."""
    mu = max(mu, _MU_FLOOR)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def poisson_logpmf(y: np.ndarray, mu: float) -> np.ndarray:
    mu = max(mu, _MU_FLOOR)
    return y * np.log(mu) - mu - gammaln(y + 1)


def _log_emissions(y: np.ndarray, mus: np.ndarray, r: float | None,
                   family: str) -> np.ndarray:
    """(T, K) matrix of log P(y_t | S_t = k)."""
    if family == "nb":
        return np.column_stack([nb_logpmf(y, m, r) for m in mus])
    if family == "poisson":
        return np.column_stack([poisson_logpmf(y, m) for m in mus])
    raise ValueError(f"unknown emission family {family!r}")


# --------------------------------------------------------------------------- #
# log-space inference
# --------------------------------------------------------------------------- #
def forward_backward(log_b: np.ndarray, log_a: np.ndarray, log_pi: np.ndarray):
    """Log-space forward-backward.

    Returns (log_alpha, log_beta, loglik, gamma, xi_sum) where gamma[t,k] =
    P(S_t=k | y_1:T) and xi_sum[i,j] = sum_t P(S_t=i, S_t+1=j | y_1:T).
    """
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_a,
                                            axis=0)
    loglik = float(logsumexp(log_alpha[-1]))

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :],
                                axis=1)

    gamma = np.exp(log_alpha + log_beta - loglik)
    # transition posteriors, accumulated over t (log-space per step)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi = (log_alpha[t][:, None] + log_a
                  + (log_b[t + 1] + log_beta[t + 1])[None, :] - loglik)
        xi_sum += np.exp(log_xi)
    return log_alpha, log_beta, loglik, gamma, xi_sum


def filtered_probs(log_alpha: np.ndarray) -> np.ndarray:
    """P(S_t | y_1:t) — causal posterior, row-normalized forward variables."""
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def viterbi(log_b: np.ndarray, log_a: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Most likely state path (log-space DP)."""
    T, K = log_b.shape
    delta = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_b[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_a
        back[t] = np.argmax(scores, axis=0)
        delta[t] = log_b[t] + np.max(scores, axis=0)
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path


# --------------------------------------------------------------------------- #
# EM
# --------------------------------------------------------------------------- #
@dataclass
class HMMFit:
    family: str
    n_states: int
    mus: np.ndarray            # ordered: mus[0] < mus[1] < ... (label guard)
    r: float | None            # NB shared dispersion; None for Poisson
    transmat: np.ndarray
    startprob: np.ndarray
    log_likelihood: float
    n_params: int
    bic: float
    n_obs: int
    n_iter: int
    converged: bool
    restart_logliks: list = field(default_factory=list)  # best-of-run per restart
    filtered: np.ndarray | None = None   # (T,K) P(S_t|y_1:t) — causal
    smoothed: np.ndarray | None = None   # (T,K) P(S_t|y_1:T) — retrospective
    viterbi_path: np.ndarray | None = None

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected regime length in quarters: 1 / (1 - p_ii)."""
        return 1.0 / np.maximum(1.0 - np.diag(self.transmat), 1e-12)


def _m_step_r(y: np.ndarray, gamma: np.ndarray, mus: np.ndarray) -> float:
    """Shared NB dispersion: numerical MLE (no closed form exists)."""
    def neg_q(log_r: float) -> float:
        r = float(np.exp(log_r))
        ll = sum(float(gamma[:, k] @ nb_logpmf(y, mus[k], r))
                 for k in range(len(mus)))
        return -ll
    res = minimize_scalar(neg_q, bounds=(np.log(1e-3), np.log(1e6)),
                          method="bounded")
    return float(np.exp(res.x))


def _em_once(y: np.ndarray, n_states: int, family: str, rng: np.random.Generator,
             max_iter: int, tol: float):
    """One EM run from a random init. Returns (params dict, loglik, it, conv)."""
    T = len(y)
    # init: means from jittered quantiles (forced distinct), sticky transitions
    qs = np.quantile(y, np.linspace(0.25, 0.75, n_states))
    mus = np.maximum(qs * rng.uniform(0.6, 1.4, n_states), _MU_FLOOR)
    mus = np.sort(mus) + np.arange(n_states) * 1e-3
    r = float(rng.uniform(1.0, 20.0))
    a = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
    np.fill_diagonal(a, 0.9)
    a = a * rng.uniform(0.8, 1.2, a.shape)
    a /= a.sum(axis=1, keepdims=True)
    pi = np.full(n_states, 1.0 / n_states)

    prev_ll, ll, converged, it = -np.inf, -np.inf, False, 0
    for it in range(1, max_iter + 1):
        log_b = _log_emissions(y, mus, r, family)
        _, _, ll, gamma, xi_sum = forward_backward(log_b, np.log(a), np.log(pi))
        # M-step
        pi = np.maximum(gamma[0], _PROB_FLOOR)
        pi /= pi.sum()
        a = np.maximum(xi_sum, _PROB_FLOOR)
        a /= a.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mus = np.maximum((gamma * y[:, None]).sum(axis=0) / np.maximum(w, 1e-12),
                         _MU_FLOOR)
        if family == "nb":
            r = _m_step_r(y, gamma, mus)
        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * abs(prev_ll):
            converged = True
            break
        prev_ll = ll
    return {"mus": mus, "r": (r if family == "nb" else None),
            "transmat": a, "startprob": pi}, ll, it, converged


def _order_states(params: dict) -> dict:
    """Label-switching guard: permute so emission means are ascending."""
    order = np.argsort(params["mus"])
    return {
        "mus": params["mus"][order],
        "r": params["r"],
        "transmat": params["transmat"][np.ix_(order, order)],
        "startprob": params["startprob"][order],
    }


def _n_params(n_states: int, family: str) -> int:
    # means + shared r (NB) + transition rows (K-1 free each) + init (K-1 free)
    return (n_states + (1 if family == "nb" else 0)
            + n_states * (n_states - 1) + (n_states - 1))


def fit_hmm(y, n_states: int = 2, family: str = "nb", n_restarts: int = 20,
            max_iter: int = 500, tol: float = 1e-6, seed: int = 0) -> HMMFit:
    """Fit by EM with random restarts; return the best fit, states ordered."""
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("counts must be non-negative")
    rng = np.random.default_rng(seed)
    best, best_ll, logliks = None, -np.inf, []
    for _ in range(n_restarts):
        params, ll, it, conv = _em_once(y, n_states, family, rng, max_iter, tol)
        logliks.append(ll)
        if ll > best_ll:
            best, best_ll, best_it, best_conv = params, ll, it, conv
    params = _order_states(best)

    log_b = _log_emissions(y, params["mus"], params["r"], family)
    log_a, log_pi = np.log(params["transmat"]), np.log(params["startprob"])
    log_alpha, _, ll, gamma, _ = forward_backward(log_b, log_a, log_pi)
    k = _n_params(n_states, family)
    return HMMFit(
        family=family, n_states=n_states, mus=params["mus"], r=params["r"],
        transmat=params["transmat"], startprob=params["startprob"],
        log_likelihood=ll, n_params=k, bic=-2.0 * ll + k * np.log(len(y)),
        n_obs=len(y), n_iter=best_it, converged=best_conv,
        restart_logliks=logliks,
        filtered=filtered_probs(log_alpha), smoothed=gamma,
        viterbi_path=viterbi(log_b, log_a, log_pi),
    )


def sample_nb_hmm(T: int, mus, r: float, transmat, startprob,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (states, counts) — used by the recovery test."""
    rng = np.random.default_rng(seed)
    mus, transmat = np.asarray(mus, float), np.asarray(transmat, float)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(len(mus), p=np.asarray(startprob, float))
    for t in range(1, T):
        states[t] = rng.choice(len(mus), p=transmat[states[t - 1]])
    # NB(mean mu, dispersion r) == Gamma-Poisson mixture
    lam = rng.gamma(shape=r, scale=mus[states] / r)
    return states, rng.poisson(lam).astype(float)
