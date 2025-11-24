# Review process & findings

driftguard's modules were built in parallel by four agents against
`docs/CONTRACT.md`, with all MLflow use isolated to `model.py`. The plan was the
same as the sibling projects: build → integrate → three adversarial reviewers
(drift math / registry & retrain loop / serving & leakage).

## What actually happened (honest account)

The four **builders finished**; the **integration and all three review agents
failed on a platform session limit** mid-run. So none of the swarm's own
verification ran. Rather than trust unverified output, I performed the
integration and the full adversarial review **by hand** — which is how the most
important issue was found.

## Finding 1 (critical): the demo evaluated the retrained model on its own training data

The original `simulate` retrained v2 on **2011 + all of 2012**, then reported
v2's MAE **on 2012** — i.e. scoring a model on data it had trained on. That
produced an inflated headline (MAE 89 → 43). The flaw originated in the
**contract itself** ("retrain on 2011+2012, evaluate on 2012"), so the builder
was faithfully implementing a bad spec — exactly the class of error the
(missing) review layer exists to catch.

**Verified:** on a *chronological* held-out future slice, the retrained model is
no better than the original (both ≈ 92 MAE) — the 89→43 gain was entirely a
train-on-test artifact.

**Fix:** 2012 is split into an **observed** slice (80%, used to detect drift and
retrain) and a **held-out** slice (20%, excluded from all training). Both models
are scored only on the held-out slice. The honest result — v1 87.3 → v2 48.4 MAE
(−44.6%) on unseen data, drift PSI 0.223 → 0.067 — is in `artifacts/lifecycle.json`
and pinned by `tests/test_cli.py`. Why a *random* rather than chronological split
is the right question to ask here is documented in the [model card](model_card.md).

## Finding 2 (accepted deviation): what "prediction drift" measures

Builder B/D noted that the contract's literal reading — prediction drift =
`PSI(model preds on reference inputs, model preds on current inputs)` — is
≈ 0.006 on this data, because the input features are stationary year-over-year,
so it would never fire. The real 2011→2012 shift is **concept/label drift**
(demand up ~63%), visible only with labels. The monitor therefore measures
prediction drift as `PSI(predictions, actual outcomes)` on the monitored period,
which fires at 0.223 and resolves to 0.067 after retraining. This is a deliberate,
documented improvement over the naive spec, not a workaround — and it is *why*
the retrain trigger requires `current_y`.

## Manual verification I ran (the reviews the swarm couldn't)

- **Drift math:** PSI identical→0, shifted→12.4, all-out-of-range→finite (no inf/nan), constant-reference→0 (no div-by-zero); KS matches `scipy`. Plus the builder's 17 drift tests.
- **Registry:** confirmed alias API (`set_registered_model_alias`, `models:/…@production`), no deprecated stages; increasing versions, promote/load round-trip, `load_production` raises before any promote. Plus 9 real-MLflow tests on a temp sqlite store.
- **Retrain loop:** reproduced end-to-end on real data — drift detected → retrain → new version registered + promoted → drift re-measured with the new model, all on the held-out slice.
- **Whole suite:** 39 tests pass, ruff + mypy clean, Docker build wired.
