# MAS Red-Team Harness

A red-team harness for the security of LLM multi-agent systems. It runs six
attacks mapped to the review paper's taxonomy against an AutoGen-style 4-agent
pipeline, scores them (success, timing, severity, false positives, with
bootstrap confidence intervals), and solves a zero-sum game between an adaptive
attacker and a defender to produce a single **security value**.

It runs **today, with no API key**, in a deterministic mock mode, and switches
to a real model (OpenAI or local Ollama) or the real AutoGen framework with one
environment variable.

---

## 1. What is in here

```
run_experiment.py          Main experiment: runs every (defense, attack) cell, scores, solves the game
run_all.sh                 Convenience wrapper
requirements.txt           numpy, scipy (plus optional pyautogen, pytest)
masredteam/
  llm.py                   Backends: mock (default), openai, ollama
  mas.py                   The 4-agent pipeline under test (Researcher->Analyst->Verifier->Executor)
  autogen_adapter.py       Seam to run the SAME attacks on the real AutoGen framework
  attacks/__init__.py      Six attacks (A1-A6) + clean and legitimate-action baselines
  defenses/__init__.py     Five defense families + fusion rules
  eval/scoring.py          Metrics and bootstrap CIs
  game/solver.py           Zero-sum LP solver for the security value
tests/test_harness.py      14 unit tests (run in mock mode, no network)
results/                   Demo output from a 30-seed mock run (cells.csv, game.json, summary.md)
```

## 2. Quick start (no keys, runs now)

```bash
pip install -r requirements.txt
python3 run_experiment.py --seeds 30 --trials 20
python3 -m pytest tests/ -q        # 14 tests
```

Outputs land in `results/`:
- `cells.csv` one row per (defense rule, attack) with success, timing, severity
- `game.json` the payoff matrix and the solved security value
- `summary.md` a human-readable write-up you can paste into the paper

## 3. Run against a real model

**Local, free (Ollama):**
```bash
ollama serve &            # in another terminal
ollama pull llama3.1:8b
MAS_LLM_MODE=ollama MAS_OLLAMA_MODEL=llama3.1:8b \
  python3 run_experiment.py --seeds 10 --trials 10
```

**OpenAI:**
```bash
MAS_LLM_MODE=openai OPENAI_API_KEY=sk-... MAS_OPENAI_MODEL=gpt-4o-mini \
  python3 run_experiment.py --seeds 5 --trials 10
```

## 4. Run against the REAL AutoGen framework

The light pipeline in `mas.py` is AutoGen-shaped so the harness runs anywhere.
To use the actual framework:

```bash
pip install pyautogen
MAS_LLM_MODE=openai OPENAI_API_KEY=sk-... \
  python3 -c "from masredteam.autogen_adapter import demo; demo()"
```

`autogen_adapter.py` builds the four agents as AutoGen `ConversableAgent`s,
registers the dangerous tools only on the Executor, feeds each attack's content
as the first message, and records a trace in the same shape the scorer expects.
Numbers for publication should come from this path plus a real model; the mock
numbers demonstrate the pipeline logic only.

## 5. The attacks (mapped to the review paper)

| ID | Attack | Dimension | Why it is strong |
|----|--------|-----------|------------------|
| A1 | Indirect prompt injection | instruction integrity | baseline, stated-intent |
| A2 | Prompt infection (self-replicating) | inter-agent trust | spreads agent to agent; success needs >=2 agents compromised |
| A3 | Control-flow hijack (forged error) | coordination | laundered through a trusted sub-agent; per-agent input filters never see it |
| A4 | Cross-session memory laundering | memory integrity | plant in session 1, act in session 2; injection and harm are different agents + sessions |
| A5 | Byzantine agent (confidence spoofing) | agent reliability | confident fabricated claim; defeats self-reported trust |
| A6 | Tool misuse / confused deputy | authority | low-privilege agent steered at a dangerous tool |

A1 and A6 are "mid-level"; A2-A5 are the high-level attacks that only become
harmful through propagation, laundering, or cross-session memory, which is the
point the review paper makes.

## 6. The defenses

| Rule | Family | Stops |
|------|--------|-------|
| `tagging_only` | prompt-level (LLM tagging + marking) | A1, A2 |
| `verify_only` | agent-guard (challenger/inspector) | A3, A5 |
| `perms_only` | authority (hard permission check) | A6 |
| `provenance_only` | memory (origin-bound authority) | A4 |
| `defense_in_depth` | all four together | A1-A6, at a utility cost |

No single defense stops everything. That gap is the experiment's main finding.

## 7. The game and the security value

For every (defense rule, attack) pair the harness measures the attacker's
success rate. This is the payoff matrix. The defender picks a (possibly
randomised) rule; the attacker, knowing that, picks its best attack. We solve
the resulting zero-sum game with a linear program (`game/solver.py`) and report
the **security value**: the attacker's success at equilibrium. We fold each
rule's false-positive rate against a 5% budget into the payoff, so an
over-strict defense (which blocks legitimate actions) is not free. This is what
makes `defense_in_depth` carry a real cost and the game non-trivial.

## 8. How to extend

- **Add an attack:** write a function in `attacks/__init__.py` that returns an
  injected `Message` with a `[[KIND:payload]]` marker and a `success` predicate,
  then add it to `ALL_ATTACKS`.
- **Add a defense:** add a class in `defenses/__init__.py` with `on_message`
  and/or `on_tool` hooks, then add a fusion rule to `FUSION_RULES`.
- **Change the utility cost model:** edit `Verification.CHALLENGE_RATE` or the
  `FPR_BUDGET` in `run_experiment.py`.

## 9. Honest limitations (put these in the paper)

- The mock backend is deterministic and models a plausibly-compliant agent. It
  proves the harness logic end to end but is **not** evidence about a real model.
  Run the Ollama or OpenAI backend, or the AutoGen adapter, for real numbers.
- The defenses are faithful-but-simplified instances of each family, enough for
  a red-team comparison, not production implementations.
- The six attacks target known weaknesses; a black-box attacker could find
  variants outside this set. Read the security value as an upper bound on safety.
- Report which backend and model produced any number you publish.
