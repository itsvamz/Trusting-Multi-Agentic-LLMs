"""Run the calibrated compound-attack game; write JSON + figures.

    python3 run_calibrated.py            # ~seconds, no keys, no network
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from masredteam import calibrated as C

OUT, FIG = "results/calibrated", "figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "gray": "#6b6b6b",
      "orange": "#E69F00", "sky": "#56B4E9"}
P0 = C.default_params()
NAMED = [(), ("T",), ("V",), ("P",), ("R",), ("T", "V", "P", "R")]
NLAB = ["none", "T", "V", "P", "R", "all"]

# ---- 1. single-attack ASR matrix (static / adaptive) ------------------------
asr = {m: np.array([[C.attack_success(P0, r, a, m) for r in NAMED] for a in C.ATTACK_IDS])
       for m in ("static", "adaptive")}
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True)
for ax, m in zip(axs, ("static", "adaptive")):
    im = ax.imshow(asr[m], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(NLAB)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{a} {C.ATTACK_NAMES[a]}" for a in C.ATTACK_IDS])
    ax.set_title(f"{m.capitalize()} attacker", fontsize=8)
    for i in range(6):
        for j in range(6):
            v = asr[m][i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else "black")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("success probability")
fig.savefig(f"{FIG}/fig_asr_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 2. compound attacks: best subset of size k ------------------------------
subs = C.all_subsets()
def best_k(rule, mode):
    out = []
    for k in range(1, 7):
        out.append(max(C.harm(P0, rule, s, mode) for s in subs if len(s) == k))
    return out
fig, ax = plt.subplots(figsize=(3.4, 2.5))
spec = [((), "static", OI["gray"], "-", "none"),
        (("T", "V", "P", "R"), "static", OI["blue"], "-", "T+V+P+R, static"),
        (("T", "V", "P", "R"), "adaptive", OI["verm"], "-", "T+V+P+R, adaptive"),
        (("R",), "adaptive", OI["green"], "--", "R only, adaptive")]
compound = {}
for rule, mode, col, ls, lab in spec:
    y = best_k(rule, mode); compound[lab] = y
    ax.plot(range(1, 7), y, color=col, ls=ls, lw=1.4, marker="o", ms=3)
    ax.annotate(lab, (6, y[-1]), xytext=(-3, 4 if lab != "R only, adaptive" else -9),
                textcoords="offset points", ha="right", fontsize=6.5, color=col)
ax.set_xlabel("attacks launched simultaneously (k)")
ax.set_ylabel("expected harm (0-1)"); ax.set_ylim(0, 1.0)
fig.savefig(f"{FIG}/fig_compound.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 3. game solutions --------------------------------------------------------
full, stat = C.solve(P0, True), C.solve(P0, False)
rules = full["rules"]
def worst(res_static, rule_i):
    return None
def summarise(res):
    return {k: res[k] for k in ("security_value", "defender_mix", "best_pure_rule",
                                "best_pure_value", "randomization_gain",
                                "attacker_best_reply", "attacker_reply_value")}
none_i = rules.index(())
all_i = rules.index(("T", "V", "P", "R"))
summ = {"static_only": summarise(stat), "adaptive_allowed": summarise(full),
        "no_defense_loss_full": float(full["U"][none_i].max()),
        "all_layers_loss_full": float(full["U"][all_i].max()),
        "no_defense_loss_static": float(stat["U"][none_i].max()),
        "all_layers_loss_static": float(stat["U"][all_i].max())}

# ---- 4. security / cost frontier ---------------------------------------------
costs = np.array([C.rule_cost(P0, r) for r in rules])
w_full = full["U"].max(axis=1) - costs      # worst-case attacker gain (adaptive allowed)
w_stat = stat["U"].max(axis=1) - costs
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.scatter(costs, w_stat, s=14, color=OI["blue"], label="static attacker", zorder=3)
ax.scatter(costs, w_full, s=14, color=OI["verm"], marker="s", label="adaptive allowed", zorder=3)
for i, r in enumerate(rules):
    if r in [(), ("T", "V", "P", "R"), ("R",), ("V", "R"), ("P", "R"), ("V", "P", "R")]:
        ax.annotate(C.rname(r), (costs[i], w_full[i]), xytext=(3, 3),
                    textcoords="offset points", fontsize=6, color=OI["verm"])
ax.set_xlabel("defense utility cost"); ax.set_ylabel("worst-case attacker gain")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.savefig(f"{FIG}/fig_frontier.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 5. sensitivity sweep over ASSUMED parameters -----------------------------
rng = np.random.default_rng(7)
vals, bests, in_support, vs_all = [], [], [], []
for _ in range(400):
    q = C.perturb(P0, rng)
    r = C.solve(q, True)
    vals.append(r["security_value"]); bests.append(r["best_pure_rule"])
    in_support.append("T+V+P+R" in r["defender_mix"])
    ri = r["rules"]; U = r["U"]
    vs_all.append(float(U[ri.index(("T", "V", "P", "R"))].max()) - r["security_value"])
from collections import Counter
sens = {"n": 400, "value_p05": float(np.percentile(vals, 5)),
        "value_p50": float(np.percentile(vals, 50)),
        "value_p95": float(np.percentile(vals, 95)),
        "best_pure_counts": dict(Counter(bests).most_common(6)),
        "all_layers_in_eq_support": float(np.mean(in_support)),
        "all_layers_regret_p95": float(np.percentile(vs_all, 95))}

# ---- 6. severity table (undefended) ------------------------------------------
sev_tab = {a: {"p": P0["p"][a], "sev": P0["sev"][a],
               "harm": P0["p"][a] * P0["sev"][a] / 3} for a in C.ATTACK_IDS}
sev_tab["all_six_undefended"] = C.harm(P0, (), tuple(C.ATTACK_IDS), "static")

json.dump({"asr_static": asr["static"].tolist(), "asr_adaptive": asr["adaptive"].tolist(),
           "compound_best_by_k": compound, "games": summ, "sensitivity": sens,
           "severity": sev_tab, "rules": [C.rname(r) for r in rules],
           "worst_case_adaptive": dict(zip([C.rname(r) for r in rules], w_full.tolist())),
           "worst_case_static": dict(zip([C.rname(r) for r in rules], w_stat.tolist())),
           "costs": dict(zip([C.rname(r) for r in rules], costs.tolist()))},
          open(f"{OUT}/calibrated.json", "w"), indent=1)
print(json.dumps({"games": summ, "sensitivity": sens, "severity": sev_tab}, indent=1))
