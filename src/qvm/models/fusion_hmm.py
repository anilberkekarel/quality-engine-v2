"""Joint multi-channel HMM: one hidden state, conditionally independent
channels (NB counts + Gaussian financials), EM in log space.

THE STEP-4 MODEL. The same latent state S_t generates every channel
(state-conditionally independent):
    patent_count_t ~ NegativeBinomial(mu_S, r)     [the Step-3 machinery]
    gross_margin_t ~ Normal(m_S, sigma_S)          [as-filed]
    revenue_yoy_t  ~ Normal(g_S, tau_S)
A quarter's log-emission is the SUM of the log-densities of the channels
OBSERVED that quarter; a missing channel's term is simply SKIPPED — the
natural mixed-frequency treatment (pre-2009 financials absent, patents
complete). Zero-filling is forbidden: a missing observation carries no
information, a zero carries a lot.

Inference reuses nb_hmm's log-space forward-backward / filtered / viterbi
verbatim (they only see a log-emission matrix). M-step: NB exactly as Step 3
(weighted-mean means, numerical shared r — both restricted to observed
quarters); Gaussians in closed form (posterior-weighted mean/variance over
observed quarters) with a VARIANCE FLOOR so no state collapses onto a single
point (the classic Gaussian-EM singularity).

LABEL ORDERING: states are sorted by the ordering channel's mean (the patent
channel for joint fits, gross margin for financial-only fits) so state 1 =
"high". All channel means per state are reported regardless — if a company's
high-patent state carries LOWER margins, that is a finding, not a bug.

Same honesty caveat as Step 3 (see nb_hmm): parameters are full-sample;
this is a DESCRIPTIVE baseline, the real-time version is Step 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .nb_hmm import (_PROB_FLOOR, filtered_probs, forward_backward, nb_logpmf,
                     viterbi)

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class Channel:
    """One observation channel: values on the common quarterly grid,
    np.nan = not observed that quarter (term skipped in the emission).

    `weight` tempers the channel's log-density in the EMISSION ONLY (E-step
    state inference): P(y|S) ∝ prod_ch density^weight. The M-step stays
    unweighted — with weight=0 the channel's parameters would be unidentified
    under a weighted M-step (any value optimal); unweighted, they remain the
    descriptive per-state fit given the states the other channels infer.
    Step 5b's tempered-joint competitor (Cw) sets weight=w on the NB channel."""
    name: str
    family: str          # 'nb' (counts) | 'normal'
    y: np.ndarray        # float array, nan = missing
    weight: float = 1.0

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.y)


@dataclass
class FusionFit:
    channel_names: list[str]
    families: dict[str, str]
    n_states: int
    # per channel: {'mus': (K,)} for nb, {'means': (K,), 'sigmas': (K,)} for normal
    channel_params: dict[str, dict] = field(default_factory=dict)
    r: float | None = None            # shared NB dispersion (None if no nb channel)
    transmat: np.ndarray | None = None
    startprob: np.ndarray | None = None
    log_likelihood: float = -np.inf
    n_params: int = 0
    bic: float = np.nan
    n_obs: int = 0                    # number of quarters on the grid (BIC n)
    n_iter: int = 0
    converged: bool = False
    order_channel: str = ""           # which channel defined the state order
    restart_logliks: list = field(default_factory=list)
    filtered: np.ndarray | None = None
    smoothed: np.ndarray | None = None
    viterbi_path: np.ndarray | None = None

    @property
    def expected_durations(self) -> np.ndarray:
        return 1.0 / np.maximum(1.0 - np.diag(self.transmat), 1e-12)

    def warm_start_dict(self) -> dict:
        """Solution in the form fit_fusion_hmm(warm_start=...) accepts."""
        return {"params": self.channel_params, "r": self.r,
                "transmat": self.transmat, "startprob": self.startprob}


def _normal_logpdf(x: np.ndarray, m: float, sigma: float) -> np.ndarray:
    return -0.5 * (_LOG_2PI + 2.0 * np.log(sigma) + ((x - m) / sigma) ** 2)


def _log_emissions(channels: list[Channel], params: dict, r: float | None,
                   n_states: int) -> np.ndarray:
    """(T,K): per quarter, SUM of observed channels' log-densities."""
    T = len(channels[0].y)
    log_b = np.zeros((T, n_states))
    for ch in channels:
        obs = ch.observed
        if not obs.any():
            continue
        yv = ch.y[obs]
        p = params[ch.name]
        for k in range(n_states):
            if ch.family == "nb":
                log_b[obs, k] += ch.weight * nb_logpmf(yv, p["mus"][k], r)
            else:
                log_b[obs, k] += ch.weight * _normal_logpdf(
                    yv, p["means"][k], p["sigmas"][k])
    return log_b


def _m_step_r_multi(channels: list[Channel], gamma: np.ndarray,
                    params: dict) -> float:
    """Shared NB dispersion across all nb channels, observed quarters only."""
    nb_chs = [c for c in channels if c.family == "nb"]

    def neg_q(log_r: float) -> float:
        r = float(np.exp(log_r))
        ll = 0.0
        for ch in nb_chs:
            obs = ch.observed
            for k in range(gamma.shape[1]):
                ll += float(gamma[obs, k] @ nb_logpmf(ch.y[obs],
                                                      params[ch.name]["mus"][k], r))
        return -ll
    res = minimize_scalar(neg_q, bounds=(np.log(1e-3), np.log(1e6)),
                          method="bounded")
    return float(np.exp(res.x))


def _variance_floor(ch: Channel) -> float:
    """Scale-aware floor: 1e-4 x the channel's overall variance (>=1e-12).
    Prevents a state collapsing onto a single point (Gaussian singularity)."""
    v = float(np.nanvar(ch.y))
    return max(1e-12, 1e-4 * v) if v > 0 else 1e-12


def _init_params(channels: list[Channel], n_states: int,
                 rng: np.random.Generator) -> tuple[dict, float | None]:
    params = {}
    has_nb = False
    for ch in channels:
        yv = ch.y[ch.observed]
        qs = np.quantile(yv, np.linspace(0.25, 0.75, n_states))
        if ch.family == "nb":
            has_nb = True
            mus = np.maximum(qs * rng.uniform(0.6, 1.4, n_states), 1e-6)
            params[ch.name] = {"mus": np.sort(mus) + np.arange(n_states) * 1e-3}
        else:
            means = qs + rng.normal(0, max(yv.std(), 1e-3) * 0.3, n_states)
            sig = max(yv.std(), 1e-3) * rng.uniform(0.6, 1.4)
            params[ch.name] = {"means": means,
                               "sigmas": np.full(n_states, sig)}
    r = float(rng.uniform(1.0, 20.0)) if has_nb else None
    return params, r


def _jitter_init(init: dict, channels: list[Channel], n_states: int,
                 rng: np.random.Generator, scale: float) -> tuple:
    """Perturb a warm-start solution (scale=0 -> exact reuse)."""
    params = {}
    for ch in channels:
        p = init["params"][ch.name]
        if ch.family == "nb":
            params[ch.name] = {"mus": np.maximum(
                p["mus"] * rng.uniform(1 - scale, 1 + scale, n_states), 1e-6)}
        else:
            params[ch.name] = {
                "means": p["means"] + rng.normal(0, scale, n_states) * p["sigmas"],
                "sigmas": p["sigmas"] * rng.uniform(1 - scale, 1 + scale, n_states)}
    a = init["transmat"] * rng.uniform(1 - scale, 1 + scale,
                                       (n_states, n_states))
    a = np.maximum(a, _PROB_FLOOR)
    a /= a.sum(axis=1, keepdims=True)
    pi = np.maximum(init["startprob"], _PROB_FLOOR)
    pi /= pi.sum()
    return params, init["r"], a, pi


def _em_once(channels: list[Channel], n_states: int, rng: np.random.Generator,
             max_iter: int, tol: float, init: tuple | None = None):
    if init is not None:
        params, r, a, pi = init
        params = {k: {kk: np.array(vv, float) for kk, vv in v.items()}
                  for k, v in params.items()}
    else:
        params, r = _init_params(channels, n_states, rng)
        a = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
        np.fill_diagonal(a, 0.9)
        a = a * rng.uniform(0.8, 1.2, a.shape)
        a /= a.sum(axis=1, keepdims=True)
        pi = np.full(n_states, 1.0 / n_states)
    floors = {ch.name: _variance_floor(ch) for ch in channels
              if ch.family == "normal"}

    prev_ll, ll, converged, it = -np.inf, -np.inf, False, 0
    for it in range(1, max_iter + 1):
        log_b = _log_emissions(channels, params, r, n_states)
        _, _, ll, gamma, xi_sum = forward_backward(log_b, np.log(a), np.log(pi))
        pi = np.maximum(gamma[0], _PROB_FLOOR)
        pi /= pi.sum()
        a = np.maximum(xi_sum, _PROB_FLOOR)
        a /= a.sum(axis=1, keepdims=True)
        for ch in channels:
            obs = ch.observed
            w = gamma[obs]                       # (T_obs, K) — observed only
            yv = ch.y[obs]
            wsum = np.maximum(w.sum(axis=0), 1e-12)
            means = (w * yv[:, None]).sum(axis=0) / wsum
            if ch.family == "nb":
                params[ch.name]["mus"] = np.maximum(means, 1e-6)
            else:
                var = (w * (yv[:, None] - means) ** 2).sum(axis=0) / wsum
                params[ch.name]["means"] = means
                params[ch.name]["sigmas"] = np.sqrt(
                    np.maximum(var, floors[ch.name]))   # degeneracy guard
        if any(c.family == "nb" for c in channels):
            r = _m_step_r_multi(channels, gamma, params)
        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * abs(prev_ll):
            converged = True
            break
        prev_ll = ll
    return params, r, a, pi, ll, it, converged


def _order_states(params: dict, r, a, pi, channels: list[Channel],
                  order_channel: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Label guard: sort states by the ordering channel's mean, ascending."""
    fam = next(c.family for c in channels if c.name == order_channel)
    key = (params[order_channel]["mus"] if fam == "nb"
           else params[order_channel]["means"])
    order = np.argsort(key)
    out = {}
    for name, p in params.items():
        out[name] = {k: v[order] for k, v in p.items()}
    return out, a[np.ix_(order, order)], pi[order]


def _n_params(channels: list[Channel], n_states: int) -> int:
    k = n_states * (n_states - 1) + (n_states - 1)
    if any(c.family == "nb" for c in channels):
        k += 1  # shared r
    for c in channels:
        k += n_states if c.family == "nb" else 2 * n_states
    return k


def fit_fusion_hmm(channels: list[Channel], n_states: int = 2,
                   order_channel: str | None = None, n_restarts: int = 20,
                   max_iter: int = 500, tol: float = 1e-6,
                   seed: int = 0, warm_start: dict | None = None) -> FusionFit:
    """Fit the joint HMM by EM with restarts; states ordered, best fit kept.

    warm_start (Step 5b expanding-window protocol): dict with keys
    params/r/transmat/startprob from the previous window's solution. The
    pre-registered restart plan then becomes 5 runs total — warm exact,
    2 warm-jittered, 2 random — instead of n_restarts random runs.
    """
    T = len(channels[0].y)
    assert all(len(c.y) == T for c in channels), "channels must share the grid"
    order_channel = order_channel or channels[0].name
    rng = np.random.default_rng(seed)

    if warm_start is not None:
        inits = [_jitter_init(warm_start, channels, n_states, rng, s)
                 for s in (0.0, 0.1, 0.1)] + [None, None]
    else:
        inits = [None] * n_restarts

    best, best_ll, logliks = None, -np.inf, []
    for init in inits:
        out = _em_once(channels, n_states, rng, max_iter, tol, init=init)
        logliks.append(out[4])
        if out[4] > best_ll:
            best, best_ll = out, out[4]
    params, r, a, pi, ll, it, conv = best
    params, a, pi = _order_states(params, r, a, pi, channels, order_channel)

    log_b = _log_emissions(channels, params, r, n_states)
    log_a, log_pi = np.log(a), np.log(pi)
    log_alpha, _, ll, gamma, _ = forward_backward(log_b, log_a, log_pi)
    k = _n_params(channels, n_states)
    return FusionFit(
        channel_names=[c.name for c in channels],
        families={c.name: c.family for c in channels},
        n_states=n_states, channel_params=params, r=r, transmat=a,
        startprob=pi, log_likelihood=ll, n_params=k,
        bic=-2.0 * ll + k * np.log(T), n_obs=T, n_iter=it, converged=conv,
        order_channel=order_channel, restart_logliks=logliks,
        filtered=filtered_probs(log_alpha), smoothed=gamma,
        viterbi_path=viterbi(log_b, log_a, log_pi),
    )


def sample_fusion(T: int, spec: dict, transmat, startprob,
                  seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Synthetic joint data for the recovery tests.

    spec: {name: {'family', 'mus'|('means','sigmas'), 'r' for nb}}.
    Returns (states, {name: values}).
    """
    rng = np.random.default_rng(seed)
    transmat = np.asarray(transmat, float)
    K = transmat.shape[0]
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=np.asarray(startprob, float))
    for t in range(1, T):
        states[t] = rng.choice(K, p=transmat[states[t - 1]])
    data = {}
    for name, p in spec.items():
        if p["family"] == "nb":
            mus, r = np.asarray(p["mus"], float), p["r"]
            lam = rng.gamma(shape=r, scale=mus[states] / r)
            data[name] = rng.poisson(lam).astype(float)
        else:
            m = np.asarray(p["means"], float)[states]
            s = np.asarray(p["sigmas"], float)[states]
            data[name] = rng.normal(m, s)
    return states, data
