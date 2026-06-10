"""NB-HMM checks: brute-force inference validation + parameter recovery.

The Step 1-2 synthetic-validation tradition applied to the model layer:
  1. forward-backward (log-space) against BRUTE-FORCE enumeration of all
     state paths on a short sequence — likelihood AND posteriors must match;
  2. generate synthetic NB-HMM data -> EM must recover the true parameters;
  3. label-ordering guard, filtered-vs-smoothed sanity, switch detector.

Run:  ./venv/bin/python tests/test_hmm_synthetic.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import itertools

import numpy as np
import pandas as pd

from qvm.analysis.regimes import persistent_switches
from qvm.models.nb_hmm import (
    _log_emissions, fit_hmm, forward_backward, sample_nb_hmm, viterbi)


def test_forward_backward_bruteforce():
    """T=6, K=2: compare log-space FB against summing all 2^6 paths."""
    rng = np.random.default_rng(7)
    T, K = 6, 2
    y = rng.poisson(8, size=T).astype(float)
    mus, r = np.array([4.0, 14.0]), 5.0
    a = np.array([[0.85, 0.15], [0.25, 0.75]])
    pi = np.array([0.6, 0.4])
    log_b = _log_emissions(y, mus, r, "nb")

    # brute force: P(y) = sum over every state path of P(path) * P(y|path)
    total = 0.0
    post = np.zeros((T, K))
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * np.exp(log_b[0, path[0]])
        for t in range(1, T):
            p *= a[path[t - 1], path[t]] * np.exp(log_b[t, path[t]])
        total += p
        for t in range(T):
            post[t, path[t]] += p
    post /= total

    log_alpha, _, loglik, gamma, _ = forward_backward(log_b, np.log(a), np.log(pi))
    assert abs(loglik - np.log(total)) < 1e-10, (loglik, np.log(total))
    assert np.max(np.abs(gamma - post)) < 1e-10
    # viterbi path must be the argmax single path from brute force
    best = max(itertools.product(range(K), repeat=T), key=lambda path: (
        np.log(pi[path[0]]) + log_b[0, path[0]]
        + sum(np.log(a[path[t - 1], path[t]]) + log_b[t, path[t]]
              for t in range(1, T))))
    assert tuple(viterbi(log_b, np.log(a), np.log(pi))) == best
    print("OK  forward-backward + viterbi match brute-force enumeration (T=6)")


def test_parameter_recovery():
    """EM on synthetic NB-HMM data recovers truth (the Step-1/2 tradition)."""
    true_mus, true_r = [20.0, 60.0], 8.0
    true_a = np.array([[0.95, 0.05], [0.10, 0.90]])
    states, y = sample_nb_hmm(600, true_mus, true_r, true_a, [0.5, 0.5], seed=11)
    fit = fit_hmm(y, n_states=2, family="nb", n_restarts=8, seed=3)
    assert fit.mus[0] < fit.mus[1]                      # label guard
    assert abs(fit.mus[0] - 20) / 20 < 0.15, fit.mus
    assert abs(fit.mus[1] - 60) / 60 < 0.15, fit.mus
    assert abs(fit.r - true_r) / true_r < 0.6, fit.r    # r is hard; loose tol
    assert abs(fit.transmat[0, 0] - 0.95) < 0.05
    assert abs(fit.transmat[1, 1] - 0.90) < 0.07
    # smoothed posterior should track the true states closely
    acc = ((fit.smoothed[:, 1] >= 0.5).astype(int) == states).mean()
    assert acc > 0.9, acc
    # filtered uses only past data -> never identical to smoothed everywhere
    assert np.max(np.abs(fit.filtered - fit.smoothed)) > 1e-4
    print(f"OK  EM recovery: mus={fit.mus.round(1)}, r={fit.r:.1f}, "
          f"diag(A)={np.diag(fit.transmat).round(3)}, state-acc={acc:.0%}")


def test_poisson_and_bic():
    """On overdispersed data, NB must beat Poisson on BIC."""
    _, y = sample_nb_hmm(300, [15.0, 45.0], 3.0, # r=3 => strong overdispersion
                         np.array([[0.9, 0.1], [0.1, 0.9]]), [0.5, 0.5], seed=5)
    nb = fit_hmm(y, 2, "nb", n_restarts=6, seed=1)
    po = fit_hmm(y, 2, "poisson", n_restarts=6, seed=1)
    assert nb.bic < po.bic, (nb.bic, po.bic)
    print(f"OK  NB beats Poisson on overdispersed data (BIC {nb.bic:.0f} vs {po.bic:.0f})")


def test_switch_detector():
    q = pd.period_range("2010Q1", periods=10, freq="Q")
    #            blip at t=2 must NOT count; persistent rise at t=5 must
    p = np.array([0.1, 0.2, 0.7, 0.2, 0.1, 0.8, 0.9, 0.95, 0.9, 0.85])
    sw = persistent_switches(q, p, min_persist=2)
    assert [s["quarter"] for s in sw] == ["2011Q2"] and sw[0]["direction"] == "up"
    # persistent fall detected symmetrically
    sw2 = persistent_switches(q, 1 - p, min_persist=2)
    assert [s["direction"] for s in sw2] == ["down"]
    print("OK  switch detector: blips ignored, persistent crossings dated")


def main():
    test_forward_backward_bruteforce()
    test_parameter_recovery()
    test_poisson_and_bic()
    test_switch_detector()
    print("\nALL HMM SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    main()
