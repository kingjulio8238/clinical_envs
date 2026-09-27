# Evaluation protocol

Every number this fork reports — README, DATA_CARD, `results/`, papers or posts built on them — follows the
rules below. Anything that does not is labeled *non-protocol* (the paper's Tables 2–3 and Appendix A are
non-protocol: zero-model floors beat the models on several columns, failures were dropped, the agentic
sample was the longest charts minus a ceiling filter, and the data generator sat in the ranked panel).

## 1. Instances and splits
- Scores are computed on **scorable instances** (`is_diagnostic = 1`) of one named split; the default is
  `public`. `heldout` is a second validation set; `train` is for training; `private` labels are not in the
  repository, so private scores come only from the operator's scorer.
- A model's per-instance reward is what `/score`, `/env` and `LocalEnv` return: the task's primary metric on
  that one instance (`eval/score_one.py PRIMARY_METRIC`). A batch metric is the mean of the per-instance
  rewards (Stage 3 convention), so aggregates and episode rewards agree.
- **Failures count.** An API error, an unparsable or missing answer, or an exhausted action budget scores
  0 on that instance and is reported (`errors`, `forced`); no instance is ever dropped. Sample sizes per cell
  are reported.

## 2. Floors and ceilings
- The **floor** of a (task, split) is the best zero-model degenerate policy in `eval/floors.json`
  (`eval/degenerate.py`: empty answer, copy the problem list, frequency prior, chart dump, order everything,
  majority class, …), recomputed by `python -m eval.floors --split <split> --write` and checked by CI.
- The **ceiling** is the label oracle on the same instances (1.0 for every task by construction; 0.99+ for
  retrieval where a patient has fewer relevant sections than the cut-off).
- Every raw score is also reported **normalized**: `(raw − floor) / (ceiling − floor)`, clipped to [−∞, 1].
  A normalized score ≤ 0 means "no better than a policy that never read the chart". Rankings use the
  normalized score; raw scores are shown beside it.

## 3. Intervals and comparisons
- Every mean carries a **95% bootstrap interval over instances** (2,000 resamples, seed 0).
- Comparisons between models are **paired**: both models are scored on the identical instance set, the
  per-instance differences are bootstrapped, and a difference whose 95% interval excludes 0 is called
  significant. Cells with different instance sets are never compared directly.
- Tables report `n`, raw mean [CI], normalized mean [CI], and Δ vs the best model in the column [paired CI].

## 4. The panel and the generator
- The ranked panel contains models that did **not** generate the benchmark. **Kimi (kimi-k2.5, the
  extraction/generation model of the data pipeline) is evaluated with identical settings and reported in a
  separate, unranked row**, because its familiarity with the material is a confound, not a capability.
- Model identity is pinned per run: provider, model id, prices at run time, the prompt hash, the budget, the
  seed, the git commit and the floors file version are written to `results/<run>/manifest.json`.
- **The panel serves the fork's question — does RL on this environment improve a model?** — so it is
  built around **one RL candidate** (an open-weights model small enough to train, tool-calling, hosted for a
  cheap baseline: `qwen3.5-9b` first) plus **one frontier anchor** (`gpt-6-sol`) that shows the headroom
  prompting alone reaches. The RL result itself is a paired before/after comparison of the base and the
  trained weights under **identical local serving**, on the same instances, with the intervals of §3; the
  hosted baseline is a preview, never the comparison. More models join the panel only after RL shows a
  gain on the first one. `scripts/refresh_model_registry.py --check` keeps every model id honest against the
  provider's live catalog (`eval/model_prices.json`).

## 5. Ablations
- An ablation changes **exactly one factor** and keeps the instances, seed, prompts, budget and model fixed:
  - **tools vs no tools**: the same instances as one-call answers over the full visible chart (`arm=single`)
    vs the tool-using agent (`arm=agent`);
  - **typical vs atypical**: `patient_diagnosis` vs `atypical_diagnosis` on the same parent instances
    (robustness = paired difference);
  - **prompt strategy**: one strategy switch on the same instances.
- Ablation results are paired differences with intervals (§3), never two independently sampled cells.

## 6. Agentic evaluation
- Agentic runs sample **patients at random with a fixed seed** from the split (at most one instance per
  patient per task, patients with ≥ 2 encounters), not the longest charts and not a ceiling-filtered set.
- Episodes run through the environment (`LocalEnv` or `/env`), which enforces the hidden outcome, the
  point-in-time cutoff, the action budget (default 40) and the forced final submission; the reward is the
  environment's. The number of steps, forced submissions and `order_test` calls are reported.
- Both environments are parity-tested (`eval/tests/test_local_env.py`, `eval/tests/test_stage8_audit.py` for the
  Stage-7 tasks), so LocalEnv results are HTTP results.
- `order_test` matches an order by test name to the findings its result text documents (`eval/stage7.py
  order_matches`: the finding's name, or every content word of the order in the documenting result text, with
  qualifiers such as "obstetric" or "exercise" optional); generic orders ("labs", "imaging") match by name only.
  Each prediction records its orders and what they matched (`order_log`, `unmatched_orders`).
- Diagnosis credit is the ICD credit, or a lower name credit when the model named the reference diagnosis but
  coded it differently (`eval/scoring.py dx_credit`: 0.75 same name, 0.5 related name in the same ICD block,
  0 on a contradicting qualifier), in every diagnosis-scored task.
- Arms: the agent arm caps each model turn at 4,096 output tokens (an RL-rollout-like limit); the single arm is
  one call and gets the registry's 16,384 so a thinking model is not truncated before it answers. Sampling
  settings differ by provider where the API forces them (GPT-6 accepts only its default temperature and
  `reasoning_effort: none` with tools); each manifest records them.

## 7. Artifacts
Each run writes `results/<model>__<task>__<arm>__<split>__s<seed>/predictions.jsonl` (one line per
instance: gt_id, reward, metric, steps, tokens, cost, error, forced, the submitted answer) and
`manifest.json`. `python -m eval.protocol report` builds `results/leaderboard.md` and `results/summary.json`
from every run directory. Runs are reproducible from the manifest; raw predictions are committed.

## 8. Cost discipline
Before a paid run: smoke 3 instances per model, project the total from measured tokens and live prices,
check the account balance, and launch only when the projection fits with margin. Every run has a hard
`--max-usd` cap and prints the running spend; a run that is provably wrong is killed, not finished.
Costs are the provider's billed cost when it reports one (OpenRouter `usage.cost`, recorded per episode as
`billed_usd` beside the list-price `list_cost_usd`), and OpenRouter runs also stop when the live account balance
falls below `--min-balance`. All units of a (model, arm) share one interleaved worker pool, one read-only release
DB and one search index (`eval/protocol_run.py run_units`).
