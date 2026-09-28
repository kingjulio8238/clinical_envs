# Oct 1 runbook — stage C on GPU (P9 + P11)

The ordered commands for G1 → projection → C2 / C4 → C5 → after-evaluation → verdict, with the workspace plan. Every
paid step follows the same four moves: **smoke → project from the smoke → `gpu/budget.py check` → launch through
`gpu/launch.sh`** (which refuses without a budget pass or a server-side `--minutes`). Paid runs need the user's go.

## 0. Rules in force (~/.claude/CLAUDE.md, modal)

- $30 credit per workspace per calendar month; **stop launching on a workspace at $25 month-to-date**; a job
  launches only if MTD + projection ≤ $25 and MTD + worst case (its timeout) ≤ $30 (`gpu/budget.py`).
- Every command pins its profile (`MODAL_PROFILE=<p>`; `gpu/launch.sh` does it and prints `modal profile current`).
- A billing query that fails is UNKNOWN and blocks the launch; never read as $0.
- Month-to-date lags up to one hour (full hours only): while a job runs, add its own spend since the last full hour
  (elapsed × $4.10/h).
- Poll every running job every few minutes on its **artifact** (records landing on the volume), and MTD at least every
  15 min of GPU time. Report "x/y done (%), ETA".
- After any disconnect, crash or interrupt: **first** `bash gpu/kill_sweep.sh <profile>`, then diagnose.
- A failed full run is never re-run to see if it repeats: reproduce the mechanism small first.

## 0b. Watching a run (every GPU job, from launch to end)

Every job writes the same two streams (`eval/run_log.py`, `gpu/telemetry.py`):
- **stdout**, one formatted line per event, prefixed `HH:MM:SS [run]`: episode progress in batches with per-unit mean
  reward and ETA; **every training step** (reward ± sd and per unit, groups with no reward spread, ART's `loss`,
  `entropy`, `kl`, `grad_norm`, importance ratio mean / p95, clipped-token fraction, generated tokens and tokens/s,
  rollout / train time, exceptions, ETA); every dev evaluation (selection score vs base and best ★, per unit, named /
  coded, probe rate, answer size, alerts); every episode **error immediately** with its instance id; `ALERT`, `STOP`,
  `FATAL` lines; a `[telemetry]` line each minute (GPU util / VRAM / power, vLLM running / waiting / KV cache / tokens/s
  / preemptions, host load). A failed start prints the last 40 lines of the vLLM log.
- **files on the results volume** (committed every 30 s): `events.jsonl` (the same events as JSON), `telemetry.jsonl`
  (every 15 s), `steps.jsonl` / `best.json` (training), `train.log` / `client.log` / `vllm.log`, `job.json` (rc, or
  the error of a failed job).

Two terminal views:
```bash
# live dashboard (PufferLib-style: summary, losses table, trends, dev table, utilization, errors, live log tail)
python scripts/rl_watch.py --profile newacc --path rl/c5-main --logs clinical-envs-train
python scripts/rl_watch.py --profile sales-32662 --path c2 --logs clinical-envs-vllm-eval
# the raw formatted stream only
MODAL_PROFILE=newacc modal app logs <ap-id from the launch> -f
```
(`rl_watch.py` needs a Python with `modal` and `rich`, e.g. the one `modal` is installed in; `--once` prints a single
snapshot for scripted polling; the header turns red on errors, on a stop, or when no event has arrived for 5 min.)
After the run, the same dashboard works on the pulled directory: `--local results/modal/<p>/rl/c5-main`.

## 1. Workspace plan

Nine workspaces (`modal profile list`; `sales-45040` and `sales-credits` are one workspace). Each has $25 of usable
room on Oct 1 and pays a one-time ≈ $0.5 on first use (image build with ART's runtime ≈ 10 CPU-min; 19 GB of weights
to its own `clinical-envs-hf-cache` volume on a CPU container).

| # | profile | jobs | projected | why this split |
|---|---|---|---|---|
| W1 | founders-78536 | G1 smoke; C5-smoke; P3 check | $3 + $5 + $1 | the measurements everything else is projected from |
| W2 | sales-32662 | C2 base, shard 0/2 (public + heldout, all units) + no-tools arms, shard 0/2 | $10–16 | C2 ≈ $16–28 does not fit one workspace |
| W3 | juliansaks | C2 base, shard 1/2 + no-tools arms, shard 1/2 | $10–16 | |
| W4 | kingjwow | C4 group variance (4 × 250 prompts × 8) | $12–20 | |
| W5 | newacc (founders-79895) | C5-main segment 1 | ≤ $20 | training in segments that each fit a workspace |
| W6 | sales-45040 | C5-main segment 2 | ≤ $20 | resumed from W5's checkpoints |
| W7 | kernel+ (kingjulio8238) | C5-main segment 3 (if needed) or after-eval shard 0/2 | ≤ $20 | |
| W8 | julian-41138 | after-evaluation, shard 1/2 (+ 0/2 if W7 trained) + no-tools arms | $10–16 | |
| W9 | credited (info-58879) | reserve: a retry, the private run (P5), overflow | — | |

Projected total ≈ $95–150 of $225 (C5-main is the uncertain term: $33–62). The table is re-cut after G1 and C5-smoke
with `gpu/project.py`; any job whose projection exceeds a workspace's room is split further (`--shard i/k` for
evaluations and C4, `--max-steps` segments for training).

## 2. Before the first launch

```bash
python3 gpu/budget.py mtd                        # every workspace: MTD (should be ~$0 on Oct 1) — none UNKNOWN
for p in founders-78536 sales-32662 juliansaks kingjwow newacc sales-45040 kernel+ julian-41138 credited; do
  bash gpu/kill_sweep.sh $p; done                # nothing of ours running anywhere
```
U1 (usage limit $30 per workspace, dashboard) must be done. Tests green on the commit being launched
(`python -m pytest -q eval/tests`), reward lock clean (`python -m eval.reward_version --check`).

## 3. G1 — smoke on W1 (the measurement)

```bash
P=founders-78536
bash gpu/launch.sh $P 0.5 gpu/vllm_eval.py --download-only --minutes 30      # CPU: weights → W1's volume
bash gpu/launch.sh $P 3 gpu/vllm_eval.py --run-name g1 --minutes 60 --cmd \
  "-m eval.protocol_run --model qwen3.5-9b-local --local --panel --smoke --workers 32 && \
   -m eval.protocol_run --model qwen3.5-9b-local --local --tasks patient_diagnosis,differential_diagnosis,evidence_retrieval,test_selection,atypical_diagnosis --n 16 --workers 80"
python scripts/sync_runs.py pull --profile $P --run g1
python gpu/project.py results/modal/$P/g1 --episodes 11699          # C2 projection
```
The first build of the image installs ART's runtime (uv sync of its lockfile) — build errors surface here, before any GPU.

**Go / no-go (all must hold):**
- `served.json`: the base served under `Qwen/Qwen3.5-9B`; `vllm.log` has no LoRA / parser warnings.
- Every unit has scored episodes; tool calls parsed (`steps` > 0 on the agent units); some rewards > 0; errors = 0.
- Truncated turns and `limit_reason` counts are plausible (not every episode at the 32,768-token cap).
- Mean reward per unit within ±0.15 of the hosted Qwen3.5-9B numbers (same model; a larger gap means a serving
  problem — parsers, sampling, template — to fix before C2).
- Measured episodes/GPU-hour and tokens/s recorded in `audit/STAGE_C_TODO.md` G1/G2; the section-1 table re-cut.

## 4. C2 base and C4 — in parallel on W2 / W3 / W4

```bash
C2='-m eval.protocol_run --model qwen3.5-9b-local --local --panel --n 100000 --workers 128'
AB='-m eval.protocol_run --model qwen3.5-9b-local --local --arm single --tasks patient_diagnosis,differential_diagnosis,test_selection --n 100000 --workers 128'
for i in 0 1; do P=$([ $i = 0 ] && echo sales-32662 || echo juliansaks)
  bash gpu/launch.sh $P 0.5 gpu/vllm_eval.py --download-only --minutes 30
  bash gpu/launch.sh $P <proj> gpu/vllm_eval.py --run-name c2 --minutes <timeout> --cmd \
    "$C2 --split public --shard $i/2 && $C2 --split heldout --shard $i/2 && $AB --split public --shard $i/2 && $AB --split heldout --shard $i/2"
done
P=kingjwow
bash gpu/launch.sh $P 0.5 gpu/vllm_eval.py --download-only --minutes 30
bash gpu/launch.sh $P <proj> gpu/vllm_eval.py --run-name c4 --minutes <timeout> --cmd \
  "scripts/group_variance.py --model qwen3.5-9b-local --per-unit 250 --k 8 --workers 128"
```
`<proj>` / `<timeout>` from `gpu/project.py` (C2 shard: half of 11,699 + the no-tools arms; C4: 8,000 episodes with
`--tokens-per-episode` of the training units). Monitor:
```bash
python scripts/sync_runs.py pull --profile $P --run c2 && wc -l results/modal/$P/c2/part*/*/predictions.jsonl
python3 gpu/budget.py mtd
```
**If a job stops early** (spend limit, crash): kill sweep; pull; `push` the partial run directory to a workspace with
room at the same `c2/partN/<run_id>` path; relaunch the same command there with `--retry-errors` added — it resumes by
`gt_id`. After both shards: `python scripts/sync_runs.py merge --out results/local/base results/modal/*/c2` (exit 0 =
every unit complete) and `python scripts/sync_runs.py merge-gv --out results/local/c4 results/modal/*/c4`.

Then: fill the criterion-2 "local" columns in `audit/RL_SUCCESS_CRITERIA.md` from the base vs GPT-6 Sol (same rule),
and the judge audit of the base answers (`scripts/reward_noise_audit.py`, OpenAI ≈ $1).

## 5. C5 — training

**C5-smoke on W1** (after G1; needs no C2 / C4):
```bash
bash gpu/launch.sh founders-78536 5 gpu/train_app.py --run-name c5-smoke --minutes 90 --args \
  "--max-steps 3 --groups-per-step 4 --rollouts-per-group 8 --val-every 3 --val-per-unit 5"
python scripts/sync_runs.py pull --profile founders-78536 --run rl/c5-smoke
```
Check the five risks of `audit/RL_TRAINING_PLAN.md` (loads; tool calls parse; tokenization / importance ratio ≈ 1 on
step 1; memory; step time). Then the **P3 check** — the after-evaluation path on the smoke's step-3 adapter:
```bash
bash gpu/launch.sh founders-78536 1 gpu/vllm_eval.py --run-name p3-check --minutes 30 \
  --lora-path /results/rl/c5-smoke/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/0003 \
  --cmd "-m eval.protocol_run --model qwen3.5-9b-rl --local --tasks patient_diagnosis --n 3 --workers 3"
```
`served.json` must show the adapter at that path and `adapter_effect.gap` > 0.

**C5-main in segments** (after C4 and the smoke; cost per step from the smoke's `steps.jsonl` × $4.10/h):
The C4 filter reaches the training container through the results volume (the image's repo mount excludes `results/`):
```bash
python scripts/sync_runs.py push-dir --profile newacc --src results/local/c4 --remote c4-merged
bash gpu/launch.sh newacc 0.5 gpu/vllm_eval.py --download-only --minutes 30
ARGS="--groups-per-step 8 --rollouts-per-group 8 --val-every 10 --val-per-unit 20 --prompts /results/c4-merged/prompts.json"
bash gpu/launch.sh newacc <proj> gpu/train_app.py --run-name c5-main --minutes <timeout> --args "$ARGS --max-steps <S>"
```
Segment 1 on W5 with `<S>` = the largest multiple of 10 whose projection fits ($25 − MTD); at its end:
```bash
python scripts/sync_runs.py pull --profile newacc --run rl/c5-main
python scripts/sync_runs.py push-dir --profile sales-45040 --src results/modal/newacc/rl/c5-main --remote rl/c5-main
python scripts/sync_runs.py push-dir --profile sales-45040 --src results/local/c4 --remote c4-merged
bash gpu/launch.sh sales-45040 0.5 gpu/vllm_eval.py --download-only --minutes 30
bash gpu/launch.sh sales-45040 <proj> gpu/train_app.py --run-name c5-main --minutes <timeout> --args "$ARGS --max-steps <S2>"
```
ART resumes from the latest checkpoint; `steps.jsonl` / `best.json` carry the history. Stop per the plan's rules
(printed `STOP ...`), or at 60 steps.

## 6. After-evaluation (W7 / W8)

The chosen step = `best.json`; its adapter goes to each evaluation workspace:
```bash
S=$(python -c "import json;print('%04d'%json.load(open('results/modal/<last>/rl/c5-main/best.json'))['step'])")
CK=results/modal/<last>/rl/c5-main/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/$S
python scripts/sync_runs.py push-dir --profile julian-41138 --src $CK --remote ckpt/c5-main-$S
bash gpu/launch.sh julian-41138 <proj> gpu/vllm_eval.py --run-name after --minutes <timeout> \
  --lora-path /results/ckpt/c5-main-$S --cmd "<the C2 command with --model qwen3.5-9b-rl, same shards>"
```
Merge to `results/local/trained`; judge audit of the trained answers; then the verdict:
```bash
python scripts/rl_before_after.py --base results/local/base --base-model qwen3.5-9b-local \
  --trained results/local/trained --trained-model qwen3.5-9b-rl \
  --trained-audit results/local/trained/reward_noise_audit.json --out results/rl_before_after.md
```
Criterion 7 (private split, once): P5.

## 7. After the study

`python3 gpu/budget.py mtd` (final spend per workspace, recorded in `audit/STAGE_C_TODO.md`); results mirrored
locally (`sync_runs.py pull` of every run); weights volumes deleted (`MODAL_PROFILE=<p> modal volume delete
clinical-envs-hf-cache -y`); the checkpoints of the chosen step kept locally, the rest deleted from the volumes.
