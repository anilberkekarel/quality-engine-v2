"""Joint multi-channel HMM checks (Step 4, the Step 1-3 synthetic tradition).

  1. three-channel synthetic joint data -> EM recovers every channel's
     state-conditional parameters;
  2. MISSING-DATA test: mask a financial-history prefix + random gaps
     (the real mixed-frequency pattern) -> EM still recovers the truth;
  3. degeneracy guard: variance floor holds even when a state has almost
     no spread; label ordering follows the requested channel.

Run:  ./venv/bin/python tests/test_fusion_synthetic.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from qvm.models.fusion_hmm import Channel, fit_fusion_hmm, sample_fusion

_SPEC = {
    "patent_count": {"family": "nb", "mus": [30.0, 90.0], "r": 8.0},
    "gross_margin": {"family": "normal", "means": [0.30, 0.55],
                     "sigmas": [0.06, 0.05]},
    "revenue_yoy":  {"family": "normal", "means": [-0.05, 0.25],
                     "sigmas": [0.10, 0.12]},
}
_A = np.array([[0.93, 0.07], [0.08, 0.92]])


def _channels(data, mask=None):
    chs = []
    for name, p in _SPEC.items():
        y = data[name].copy()
        if mask is not None and name in mask:
            y[mask[name]] = np.nan
        chs.append(Channel(name=name, family=p["family"], y=y))
    return chs


def _check_recovery(fit, states, tag):
    pp = fit.channel_params
    assert abs(pp["patent_count"]["mus"][0] - 30) / 30 < 0.2, pp
    assert abs(pp["patent_count"]["mus"][1] - 90) / 90 < 0.2, pp
    assert abs(pp["gross_margin"]["means"][0] - 0.30) < 0.05
    assert abs(pp["gross_margin"]["means"][1] - 0.55) < 0.05
    assert abs(pp["revenue_yoy"]["means"][1] - 0.25) < 0.08
    assert abs(fit.transmat[0, 0] - 0.93) < 0.06
    acc = ((fit.smoothed[:, 1] >= 0.5).astype(int) == states).mean()
    assert acc > 0.9, acc
    print(f"OK  {tag}: patents={pp['patent_count']['mus'].round(1)}, "
          f"margins={pp['gross_margin']['means'].round(3)}, "
          f"yoy={pp['revenue_yoy']['means'].round(3)}, state-acc={acc:.0%}")


def test_full_recovery():
    states, data = sample_fusion(500, _SPEC, _A, [0.5, 0.5], seed=21)
    fit = fit_fusion_hmm(_channels(data), n_states=2,
                         order_channel="patent_count", n_restarts=8, seed=2)
    assert fit.order_channel == "patent_count"
    assert fit.channel_params["patent_count"]["mus"][0] \
        < fit.channel_params["patent_count"]["mus"][1]  # label guard
    _check_recovery(fit, states, "full three-channel recovery")


def test_missing_data():
    """Mask the real pattern: financials absent for a long prefix + gaps.
    Skipping missing terms (NOT zero-filling) must still recover the truth."""
    rng = np.random.default_rng(4)
    states, data = sample_fusion(500, _SPEC, _A, [0.5, 0.5], seed=22)
    mask = {
        "gross_margin": np.concatenate([np.ones(120, bool),       # no early history
                                        rng.random(380) < 0.15]),  # sporadic gaps
        "revenue_yoy": np.concatenate([np.ones(140, bool),
                                       rng.random(360) < 0.15]),
    }
    fit = fit_fusion_hmm(_channels(data, mask), n_states=2,
                         order_channel="patent_count", n_restarts=8, seed=5)
    _check_recovery(fit, states, "missing-data (prefix+gaps) recovery")
    # the variance floor must have kept every sigma strictly positive
    for name, fam in fit.families.items():
        if fam == "normal":
            assert (fit.channel_params[name]["sigmas"] > 0).all()
    print("OK  variance floor: all Gaussian sigmas strictly positive")


def test_financial_only_ordering():
    """No NB channel: ordering by gross margin must define state 1 = high."""
    states, data = sample_fusion(400, _SPEC, _A, [0.5, 0.5], seed=23)
    chs = [c for c in _channels(data) if c.name != "patent_count"]
    fit = fit_fusion_hmm(chs, n_states=2, order_channel="gross_margin",
                         n_restarts=6, seed=7)
    m = fit.channel_params["gross_margin"]["means"]
    assert m[0] < m[1] and fit.r is None
    print(f"OK  financial-only fit: margin-ordered states {m.round(3)}, no NB part")


def main():
    test_full_recovery()
    test_missing_data()
    test_financial_only_ordering()
    print("\nALL FUSION SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    main()
