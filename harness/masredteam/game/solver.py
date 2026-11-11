"""Game-theoretic security value.

We build a payoff matrix P where rows are defender fusion rules and columns are
attacker styles. Each cell P[i,j] is the attacker's success rate for that
(rule, style) pair, measured from simulated runs (empirical game-theoretic
analysis). The defender wants to MINIMISE the attacker's best column; the
attacker, knowing the defender's (possibly randomised) rule, picks its best
reply. For a zero-sum game this is a linear program.

Security value v = min over defender mixes x of  max over attacker j of sum_i x_i P[i,j].

We also report:
  - the best PURE rule (lowest worst-case column), and
  - the gain from randomising (pure worst-case minus mixed value).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def solve_zero_sum(P, rule_names, attack_names):
    """P: (R x A) attacker-success matrix. Returns defender mix, value, details."""
    P = np.asarray(P, dtype=float)
    R, A = P.shape

    # Defender minimises v s.t. for every attacker column j: sum_i x_i P[i,j] <= v
    # Variables: [x_0..x_{R-1}, v]. Minimise v.
    c = np.zeros(R + 1); c[-1] = 1.0
    # columns constraints: P^T x - v <= 0  ->  [P[:,j], -1] . [x,v] <= 0
    A_ub = np.hstack([P.T, -np.ones((A, 1))])
    b_ub = np.zeros(A)
    # x sums to 1
    A_eq = np.zeros((1, R + 1)); A_eq[0, :R] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, 1)] * R + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    x = res.x[:R]
    v = res.x[-1]

    # best pure rule = row with smallest max
    worst_per_rule = P.max(axis=1)
    best_pure_idx = int(np.argmin(worst_per_rule))
    best_pure_val = float(worst_per_rule[best_pure_idx])

    # attacker best reply to the mixed defender
    col_vals = x @ P
    atk_best_idx = int(np.argmax(col_vals))

    return {
        "security_value": round(float(v), 4),
        "defender_mix": {rule_names[i]: round(float(x[i]), 3)
                         for i in range(R) if x[i] > 1e-6},
        "best_pure_rule": rule_names[best_pure_idx],
        "best_pure_value": round(best_pure_val, 4),
        "randomization_gain": round(best_pure_val - float(v), 4),
        "attacker_best_reply": attack_names[atk_best_idx],
        "attacker_reply_value": round(float(col_vals[atk_best_idx]), 4),
    }
