# Stage 8: evaluation protocol — todo

Scope: ROADMAP Stage 8 only. Goal: a protocol every number this fork reports follows (floors, ceilings,
paired intervals, the generator outside the ranked panel, one-factor ablations, random agentic samples,
failures counted), the code that enforces it, and the first protocol-conformant results committed under
`results/`. Done = every box checked, a leaderboard with intervals for the chosen panel on the public
split, validation filled in.

## Design decisions (fixed before coding)
- **Protocol = `EVAL_PROTOCOL.md`**; every table in README / DATA_CARD / results cites it, and the paper's
  Tables 2–3 / Appendix A are labeled non-protocol.
- **One runner for every task and both arms** (`eval/protocol_run.py`): a model plays instances as
  `LocalEnv` episodes (parity-tested against the HTTP env), so the reward is the environment's, the cutoff /
  hidden sections / overrides / budget are enforced, and the new Stage-7 tasks (incl. `order_test`) run the
  same way as the old ones. `arm=agent` = tool loop; `arm=single` = one call over the visible chart with
  only the submit tool: the tools-vs-no-tools ablation on identical instances. No Postgres needed for runs;
  artifacts are JSONL + manifest per run.
- **Sampling**: seeded random instances, at most one per patient first (`sample_instances`); the legacy
  agent runner's `select_patients` now samples at random with a seed (default public; `val` rejected; the
  ceiling filter is off by default and documented as the paper's bias).
- **Failures score 0 and stay in `n`** (API errors after the adapter's retries, unparsable answers, budget
  exhaustion → forced empty submission), each recorded with its error string.
- **Statistics** (`eval/protocol.py`): normalized score from `eval/floors.json`; 95% bootstrap intervals
  (2,000 resamples, seed 0); paired bootstrap of per-instance differences vs the best model; ablation tables
  paired on instances (tools − no tools; atypical − typical via `parent_gt_id`).
- **Panel selection** (Artificial Analysis intelligence vs price, September 2026, tool-calling models on
  OpenRouter / OpenAI): a frontier anchor on a smaller sample, the value frontier, the best open-weights
  models, the cheapest tool-capable model, and Kimi K2.5 (the generator) in its own unranked row. Prices are
  re-read from OpenRouter's live list at run time and written to the manifest.
- **Cost discipline**: smoke 3 instances per model first; project from measured tokens; `--max-usd` per
  run; running spend printed every 10 episodes; launch only when the projection fits the balance.

## Protocol and code
- [x] `EVAL_PROTOCOL.md` (instances/splits, floors/ceilings, intervals, panel vs generator, ablations, agentic sampling, artifacts, cost discipline)
- [x] `eval/protocol.py`: bootstrap CI, paired differences, normalization, leaderboard + ablation tables, `report` CLI
- [x] `eval/protocol_run.py`: LocalEnv episodes with the model adapters, both arms, seeded sampling, failure = 0, resume, cost cap, manifest
- [x] registry: protocol panel (`gpt-6-sol`, `gpt-6-luna` via OpenAI; `mimo-v2.6-pro`, `muse-spark-1.3`, `deepseek-v4-pro`, `glm-5.3-flash`, `opus-5.5`, `kimi-k2.5` via OpenRouter) with prices
- [x] runner fixes: `select_patients` seeded random / default public / ceiling filter off; `eval/split.py` and `scripts/build_rl_split.py` no longer reference the retired task
- [ ] `eval/tests/test_protocol.py` green (arithmetic; scripted-adapter runs: oracle agent → 1.0 incl. `order_test`, failure → 0 recorded, resume, single arm sees the visible chart only)

## Runs (public split; need OPENROUTER_API_KEY / OPENAI_API_KEY and balance)
- [ ] smoke: 3 instances × each panel model; measured tokens/episode and $/episode recorded here
- [ ] agent arm: 120 instances × 9 scoring units for the value/open models; frontier anchor on 60/unit
- [ ] single arm (no tools) on 3 units for 3 models — tools ablation
- [ ] Kimi K2.5 with identical settings — separate row
- [ ] `python -m eval.protocol report` → `results/leaderboard.md`, `results/summary.json`; predictions + manifests committed

## Apply to what the fork reports
- [ ] README results section cites the protocol and the leaderboard; paper numbers labeled non-protocol
- [ ] DATA_CARD / ROADMAP / STATE updated; memory

## Validate
- [ ] `pytest eval/tests etl/tests` green
- [ ] committed and pushed; main synced
