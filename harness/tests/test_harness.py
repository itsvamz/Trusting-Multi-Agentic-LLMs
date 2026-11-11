"""Unit tests for the MAS red-team harness. Run: python -m pytest -q
These run in mock mode, no network or keys needed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MAS_LLM_MODE"] = "mock"

from masredteam.mas import Pipeline, Message, MemoryItem
from masredteam.attacks import ALL_ATTACKS, clean_run, legit_action
from masredteam.defenses import FUSION_RULES, rule_none, rule_all
from masredteam.eval.scoring import caught_in_time, severity, bootstrap_ci
from masredteam.game.solver import solve_zero_sum


def _run(attack_fn, rule_factory, seed=1):
    pipe = Pipeline(defenses=rule_factory())
    inj, meta = attack_fn("task", seed)
    if meta.get("two_session"):
        pipe.run("task", seed, injected=inj)
        trig = Message("world", "Researcher", work=meta["trigger_task"],
                       provenance="agent", trust="trusted")
        tr = pipe.run(meta["trigger_task"], seed, injected=trig)
    else:
        tr = pipe.run("task", seed, injected=inj)
    return tr, meta


def test_all_attacks_succeed_without_defense():
    for afn in ALL_ATTACKS:
        tr, meta = _run(afn, rule_none)
        if meta["id"] == "A6":
            # A6 is blocked by the structural permission check even with no
            # extra defense: only the Executor holds dangerous tools.
            assert not meta["success"](tr)
        else:
            assert meta["success"](tr), f"{meta['id']} should succeed undefended"


def test_defense_in_depth_stops_everything():
    for afn in ALL_ATTACKS:
        tr, meta = _run(afn, rule_all)
        assert not meta["success"](tr), f"{meta['id']} should be stopped by all-defenses"


def test_each_defense_has_a_hole():
    # No single non-stacked defense stops every attack.
    singles = {k: v for k, v in FUSION_RULES.items()
               if k not in ("defense_in_depth",)}
    for name, rf in singles.items():
        def stopped(afn):
            tr, meta = _run(afn, rf)
            return not meta["success"](tr)
        stopped_all = all(stopped(afn) for afn in ALL_ATTACKS)
        assert not stopped_all, f"{name} should NOT stop every attack"


def test_tagging_stops_injection_and_infection():
    for afn in (ALL_ATTACKS[0], ALL_ATTACKS[1]):  # A1, A2
        tr, meta = _run(afn, FUSION_RULES["tagging_only"])
        assert not meta["success"](tr)


def test_provenance_stops_memory_laundering():
    tr, meta = _run(ALL_ATTACKS[3], FUSION_RULES["provenance_only"])  # A4
    assert not meta["success"](tr)


def test_verification_stops_byzantine_and_hijack():
    for afn in (ALL_ATTACKS[2], ALL_ATTACKS[4]):  # A3, A5
        tr, meta = _run(afn, FUSION_RULES["verify_only"])
        assert not meta["success"](tr)


def test_permission_blocks_out_of_role_tool():
    tr, meta = _run(ALL_ATTACKS[5], FUSION_RULES["perms_only"])  # A6
    assert not meta["success"](tr)
    assert any(c[2] == "BLOCKED" for c in tr.tool_calls)


def test_legit_action_completes_without_overstrict_defense():
    pipe = Pipeline(defenses=rule_none())
    pipe.memory.write(MemoryItem("Vendor Y approved account", "tool", "finance", "shared"))
    inj, meta = legit_action("t", 1)
    tr = pipe.run(inj.work, 1, injected=inj)
    assert any(c[1] == "payment" and c[2] == "ALLOWED" for c in tr.tool_calls)


def test_timing_onset_before_harm_when_both_exist():
    tr, meta = _run(ALL_ATTACKS[2], rule_none)  # A3 reaches harm
    if tr.onset_step is not None and tr.harm_step is not None:
        assert tr.onset_step <= tr.harm_step


def test_bootstrap_ci_bounds():
    base, lo, hi = bootstrap_ci([1, 1, 0, 1, 0, 1, 1, 0], n=500)
    assert 0.0 <= lo <= base <= hi <= 1.0


def test_solver_basic_zero_sum():
    # defender has a dominant safe rule in column 0; value should be low
    P = [[0.0, 1.0], [1.0, 0.0]]
    out = solve_zero_sum(P, ["r0", "r1"], ["a0", "a1"])
    # mixing 50/50 gives value 0.5 for this symmetric game
    assert abs(out["security_value"] - 0.5) < 1e-6


def test_solver_dominant_rule():
    P = [[0.1, 0.1], [0.9, 0.9]]
    out = solve_zero_sum(P, ["safe", "bad"], ["a0", "a1"])
    assert out["best_pure_rule"] == "safe"
    assert out["security_value"] <= 0.1 + 1e-6


def test_clean_run_no_success():
    tr, meta = _run(clean_run, rule_none)
    assert not meta["success"](tr)


def test_severity_scale():
    tr, meta = _run(ALL_ATTACKS[2], rule_none)  # A3, dangerous tool fires
    assert severity(meta, tr) >= 3
