# CLAUDE.md — Hybrid QML Atomization Energy (QM7)

3-person QCI R&D project. MVP complete (all 5 phases) as of 2026-06-10.
Read README.md for results and the full standards/baselines table.

## State

- Best hybrid: 6 qubits, depth 3, lr 1e-3, 400 epochs → 16.05 kcal/mol test MAE
- KRR baseline 10.76 (matches eigenvalue-feature literature ~10; the often-cited
  ~3.1 needs richer features — do NOT use 3.1 as the bar)
- Classical ablation 16.13 → quantum layer shows no advantage at this scale
- All runs logged in results/experiments.csv; histories in results/history/
- Open direction if iterating: beat KRR via richer encoding (IQP) or removing
  the Linear(23→n_qubits) bottleneck

## Environment

- Use `.venv/bin/python` (Python 3.12 — DeepChem doesn't support 3.13)
- Data downloads need `SSL_CERT_FILE="$(python -m certifi)"` (python.org build quirk)
- NEVER `python -m jupyter <cmd>` — it resolves subcommands via PATH and silently
  runs Anaconda's jupyter. Use `.venv/bin/python -m nbconvert ...` directly
- Shell is zsh: unquoted `$var` does not word-split; `status` is read-only

## Commands

```bash
.venv/bin/python -m unittest discover tests          # 6 tests
.venv/bin/python scripts/train.py --model-name X ... # logs to experiments.csv
.venv/bin/python scripts/baseline_krr.py
.venv/bin/python scripts/make_plots.py               # PNGs to results/
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

## Conventions

- Atomic commits, conventional prefixes (feat/fix/docs/refactor/results)
- Standardize features and normalize labels with TRAIN statistics only
- Test-set evaluation happens ONCE per final model (--eval-test), never during tuning
- Seed 42 everywhere; runs are deterministic and reproducible
