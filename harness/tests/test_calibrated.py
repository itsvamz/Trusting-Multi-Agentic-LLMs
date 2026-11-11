import numpy as np
from masredteam import calibrated as C

def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(0)
    qs, sv = [0.3, 0.7, 0.5], [1, 3, 2]
    draws = rng.random((200000, 3)) < qs
    mc = (draws * np.array(sv)).max(axis=1).mean()
    assert abs(C.expected_max_severity(qs, sv) - mc) < 0.01

def test_adding_layer_never_raises_success():
    P = C.default_params()
    for a in C.ATTACK_IDS:
        for m in ("static", "adaptive"):
            for r in C.all_rules():
                for d in C.LAYERS:
                    if d not in r:
                        assert C.attack_success(P, r + (d,), a, m) <= C.attack_success(P, r, a, m) + 1e-12

def test_adaptive_never_weaker_than_static():
    P = C.default_params()
    for a in C.ATTACK_IDS:
        for r in C.all_rules():
            assert C.attack_success(P, r, a, "adaptive") >= C.attack_success(P, r, a, "static") - 1e-12

def test_mixed_value_not_worse_than_best_pure():
    res = C.solve(C.default_params(), True)
    assert res["security_value"] <= res["best_pure_value"] + 1e-9
