"""Calibrated simulation + empirical game for compound (simultaneous) attacks.

WHAT THIS IS (and is not)
-------------------------
The mock pipeline in mas.py is deterministic: a defense "defuses" exactly the
marker kinds its own family targets, so its 0/1 outcomes follow by construction.
This module adds a stochastic layer on top of the same six attacks (A1-A6) and
four defense families (T, V, P, R):

  * each attack has an undefended success probability p_i taken from the
    literature where a number exists, otherwise marked ASSUMED (dagger);
  * each defense layer d scales that probability by a residual multiplier
    m_{d,i} for a STATIC attacker and a larger one for an ADAPTIVE attacker
    (adaptive multipliers follow Zhan et al. / Lee & Tiwari, see paper);
  * layers combine multiplicatively (independence assumption, optimistic for
    the defender);
  * an attacker strategy is a NON-EMPTY SUBSET of attacks launched
    simultaneously, plus a mode (static|adaptive); payoff is the expected
    maximum severity achieved, minus effort costs;
  * the defender picks any subset of the four layers (16 pure strategies) and
    pays each layer's utility cost; the zero-sum game is solved by LP.

It is a model of published evidence, NOT a measurement of any live LLM. All
assumed parameters are perturbed in a sensitivity sweep (run_calibrated.py).
"""
from __future__ import annotations
import itertools
import numpy as np
from .game.solver import solve_zero_sum

ATTACK_IDS = ["A1", "A2", "A3", "A4", "A5", "A6"]
ATTACK_NAMES = {
    "A1": "Indirect injection", "A2": "Prompt infection",
    "A3": "Control-flow hijack", "A4": "Memory laundering",
    "A5": "Byzantine agent", "A6": "Tool misuse",
}
# (undefended success p, severity 1-3, p_is_assumed, source)
ATTACKS = {
    "A1": (0.24, 2, False, "InjecAgent, GPT-4 ReAct [31]"),
    "A2": (0.66, 2, False, "Prompt infection, GPT-4o [11]"),
    "A3": (0.97, 3, False, "Magentic-One local file, GPT-4o [10]"),
    "A4": (0.80, 3, False, "AgentPoison, >80% at 0.1% poison [25]"),
    "A5": (0.60, 1, True,  "assumed; no direct measurement"),
    "A6": (0.60, 3, False, "ConVerse security attacks, GPT-5 [14]"),
}
LAYERS = ["T", "V", "P", "R"]
LAYER_NAMES = {"T": "Tagging", "V": "Verification", "P": "Permissions",
               "R": "Provenance"}
# layer -> (utility cost, cost_assumed, {attack: (static, adaptive, s_asm, a_asm)})
DEFENSES = {
    "T": (0.01, True, {
        "A1": (0.03, 0.76, False, False),
        "A2": (0.03, 0.76, False, False)}),
    "V": (0.075, False, {
        "A1": (0.50, 0.80, True, True),
        "A2": (0.50, 0.80, True, True),
        "A3": (0.35, 0.65, True, True),
        "A5": (0.04, 0.50, False, False),
        "A6": (0.20, 0.60, True, True)}),
    "P": (0.02, True, {
        "A6": (0.02, 0.05, True, True)}),
    "R": (0.05, True, {
        "A4": (0.02, 0.45, False, True),
        "A3": (0.60, 0.90, True, True),
        "A6": (0.60, 0.90, True, True)}),
}
EFFORT = 0.03          # cost per attack launched
ADAPT_EFFORT = 0.04    # extra cost per attack to adapt to the deployed defense


def default_params():
    return {
        "p": {a: ATTACKS[a][0] for a in ATTACK_IDS},
        "sev": {a: ATTACKS[a][1] for a in ATTACK_IDS},
        "cost": {d: DEFENSES[d][0] for d in LAYERS},
        "m": {d: {a: [v[0], v[1]] for a, v in DEFENSES[d][2].items()}
              for d in LAYERS},
    }


def assumed_slots():
    """List of mutable parameter slots flagged as assumed (for the sweep)."""
    slots = []
    for a in ATTACK_IDS:
        if ATTACKS[a][2]:
            slots.append(("p", a))
    for d in LAYERS:
        if DEFENSES[d][1]:
            slots.append(("cost", d))
        for a, (s, ad, sa, aa) in DEFENSES[d][2].items():
            if sa:
                slots.append(("ms", d, a))
            if aa:
                slots.append(("ma", d, a))
    return slots


def perturb(params, rng, width=0.2):
    """Return a copy with every ASSUMED slot perturbed (uniform +-width)."""
    import copy
    q = copy.deepcopy(params)
    for s in assumed_slots():
        if s[0] == "p":
            q["p"][s[1]] = float(np.clip(q["p"][s[1]] + rng.uniform(-width, width), 0.05, 1.0))
        elif s[0] == "cost":
            q["cost"][s[1]] *= float(rng.uniform(0.5, 1.5))
        elif s[0] == "ms":
            q["m"][s[1]][s[2]][0] = float(np.clip(q["m"][s[1]][s[2]][0] + rng.uniform(-width, width), 0.0, 1.0))
        else:
            q["m"][s[1]][s[2]][1] = float(np.clip(q["m"][s[1]][s[2]][1] + rng.uniform(-width, width), 0.0, 1.0))
    for d in LAYERS:                       # adaptive residual never below static
        for a, v in q["m"][d].items():
            v[1] = max(v[1], v[0])
    return q


def attack_success(params, rule, attack, mode):
    """P(attack succeeds) against a rule (tuple of layers) and attacker mode."""
    q = params["p"][attack]
    k = 0 if mode == "static" else 1
    for d in rule:
        q *= params["m"][d].get(attack, [1.0, 1.0])[k]
    return q


def expected_max_severity(qs, sevs):
    """E[max severity of succeeding attacks] for independent Bernoulli(q_i)."""
    order = sorted(range(len(qs)), key=lambda i: -sevs[i])
    e, none_before = 0.0, 1.0
    for i in order:
        e += sevs[i] * qs[i] * none_before
        none_before *= (1.0 - qs[i])
    return e


def all_rules():
    return [tuple(c) for r in range(len(LAYERS) + 1)
            for c in itertools.combinations(LAYERS, r)]


def all_subsets():
    return [tuple(c) for r in range(1, len(ATTACK_IDS) + 1)
            for c in itertools.combinations(ATTACK_IDS, r)]


def harm(params, rule, subset, mode):
    qs = [attack_success(params, rule, a, mode) for a in subset]
    sv = [params["sev"][a] for a in subset]
    return expected_max_severity(qs, sv) / 3.0      # normalised to [0,1]


def rule_cost(params, rule):
    return sum(params["cost"][d] for d in rule)


def build_game(params, adaptive_allowed=True):
    rules = all_rules()
    strategies = [("abstain", None)]
    modes = ["static", "adaptive"] if adaptive_allowed else ["static"]
    for sub in all_subsets():
        for m in modes:
            strategies.append((m, sub))
    U = np.zeros((len(rules), len(strategies)))
    for i, r in enumerate(rules):
        c = rule_cost(params, r)
        for j, (m, sub) in enumerate(strategies):
            if sub is None:
                gain = 0.0
            else:
                gain = harm(params, r, sub, m) - EFFORT * len(sub) \
                    - (ADAPT_EFFORT * len(sub) if m == "adaptive" else 0.0)
            U[i, j] = gain + c              # defender loss = attacker gain + cost
    return rules, strategies, U


def rname(rule):
    return "+".join(rule) if rule else "none"


def sname(s):
    m, sub = s
    return "abstain" if sub is None else f"{'+'.join(sub)}/{m[:3]}"


def solve(params, adaptive_allowed=True):
    rules, strategies, U = build_game(params, adaptive_allowed)
    res = solve_zero_sum(U, [rname(r) for r in rules], [sname(s) for s in strategies])
    x = np.array([res["defender_mix"].get(rname(r), 0.0) for r in rules])
    col = x @ U
    res["attacker_best_reply"] = sname(strategies[int(np.argmax(col))])
    res["rules"], res["strategies"], res["U"] = rules, strategies, U
    return res
