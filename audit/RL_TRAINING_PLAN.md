# RL training plan — Qwen3.5-9B on the environment (P7)

What is trained, how, when it stops, what it costs, and what to do when it breaks. The judge of the result is
`audit/RL_SUCCESS_CRITERIA.md` (fixed before training); this plan only produces the checkpoint it judges.
Numbers marked *proj.* are projections from the hosted runs, replaced by the G1 / C5-smoke measurements.

## Setup

| item | value | where |
|---|---|---|
| policy | `Qwen/Qwen3.5-9B` (bf16), LoRA rank 16, alpha 32 | `scripts/train_rl.py` |
| LoRA targets | q/k/v/o, linear-attention `in_proj_qkv` / `in_proj_z` / `out_proj`, MLP gate/up/down | `QWEN3_5_LORA_TARGETS` (P6) |
| trainer | ART 0.5.20 LocalBackend: vLLM samples, Unsloth trains, one H100 80 GB, sleep mode between phases | `gpu/train_app.py` |
| loss | ART default: CISPO (clipped importance weights, token level), group-normalized advantages (GRPO), KL off | `train_kwargs` |
| sampling (train and dev) | temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5 (Qwen3.5 thinking defaults) | `eval/rl_rollout.py` |
| limits (= evaluation's) | turns per unit (budget 40 + 3; error_detection 16), 4,096 output tokens per turn, 32,768 per episode | `eval/episode.py` |
| reward | frozen `reward-v3`; the run refuses to start on a drifted scorer | `eval/reward_lock.json` |
| sequence length | 65,536 | P6 |

## Data

- **Training prompts:** train-split instances of the 4 training units, dev patients excluded (P4): 6,194 prompts
  (evidence_retrieval 1,998, patient_diagnosis 1,944, differential_diagnosis 1,365, test_selection 887), mixed and
  shuffled with seed 0; one pass over them is an epoch.
- **C4 filter:** k = 8 samples per prompt on 250 prompts per unit (1,000 measured). `prompts.json` lists the prompts
  with reward spread (`keep`, std > 0.01) and the measured rest (`drop`). Training uses only `keep` when it holds at
  least 400 prompts (a pure-signal pool, about one epoch in C5-main), else every prompt except `drop` (a small
  keep-list would be repeated many times) — `--min-keep`, the mode recorded in `config.json`.
- **Dev set (checkpoint selection):** 52 train-split patients (`sha256("rl-dev:<id>") % 10 == 0`), 20 instances per
  unit per evaluation, plus atypical_diagnosis (logged, not selected on). Public, heldout and private are not touched.

## Schedule

| phase | steps | per step | evaluation | purpose |
|---|---|---|---|---|
| C5-smoke | 3 | 4 groups × 8 rollouts | dev 5 per unit at step 0 and 3 | throughput, memory, tokenization check, cost per step |
| C5-main | up to 60 | 8 groups × 8 rollouts = 64 episodes | dev 20 per unit at step 0 and every 10 steps | the checkpoint |

60 steps × 64 = 3,840 episodes = 480 prompt groups: about one epoch of a keep-list of ~500 prompts, or 0.08 epoch of
the drop-filtered pool — in neither case is a prompt trained on more than about once, so a train-only gain is not
memorization of repeated prompts.

**Hyperparameters and the order to change them:** learning rate 1e-5 first. If train reward is flat after 20 steps
and the C4 filter kept enough prompts: 3e-5. If train rises and dev does not (overfitting alert): KL 0.02 to the base,
then fewer steps. If many groups are all-zero: k = 16 at 4 groups per step (same episodes per step).

## Stopping rules (`scripts/train_rl.py`)

- **Max steps** reached (60), or the workspace's step budget (the job is sized so its timeout fits the credit; a
  resumed job continues from ART's latest checkpoint with the logged history on the next workspace, P10 `push-dir`).
- **Dev plateau:** no new best dev selection score (macro mean over the 4 training units) in 3 evaluations (30 steps).
- **Reward hack:** the same monitor alert (probe pattern > 5% of answers, answer size or name length > 2× base,
  "reward up while naming fell") on 2 consecutive evaluations. The run stops; the scorer is fixed and re-locked
  before any retraining (criteria outcome table).
- **Manual kill:** > 10% of a step's episodes are exceptions (infrastructure), step time > 2× the smoke's, or the
  month-to-date spend reaches the $25 stop line.

## Observability

Every step, dev evaluation, episode error, alert and stop is a formatted stdout line (streamed by `modal app logs -f`)
and a JSON event in `events.jsonl`; GPU / vLLM / host telemetry every 15 s. `scripts/rl_watch.py` renders them live
as a terminal dashboard (after PufferLib's `print_dashboard`: summary, ART losses per step, reward / loss / entropy / KL
/ dev trends, per-unit rewards, dev table with the best checkpoint, performance breakdown, utilization, error tail).
What to watch, per step: reward and its spread (a collapse of spread = no signal); the share of groups with no spread;
`kl` and the importance ratio (≈ 1 at step 1; growing ratios / clipped fraction = the sampler and trainer diverge);
`entropy` (a fast fall = mode collapse); `grad_norm` spikes; exceptions; tokens per episode (a climb toward the cap
= a length hack). The runbook (§0b) has the commands.

## Checkpoints

ART saves a LoRA checkpoint every step; after each dev evaluation only the evaluated steps and the latest are kept
(`prune`). `results/rl/<run>/best.json` names the best evaluated step (ties → earlier step), with the base's dev
score from step 0. That checkpoint — `/results/rl/<run>/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/<step:04d>`
— is the "after" of the before/after evaluation (P3), run once on public + heldout; the private split once at the end.
Optimizer state is saved every 5 steps: a job moved between workspaces should stop on a multiple of 5.

## Budget per phase (*proj.*, one H100 at ~$4.10/h with CPU and memory)

| phase | work | GPU-hours *proj.* | cost *proj.* |
|---|---|---|---|
| C5-smoke | 3 steps × 32 episodes + 2 dev passes | 0.7–1.2 | $3–5 |
| C5-main | 60 steps × 64 episodes + 7 dev passes of 100 | 8–15 | $33–62 |
| after-evaluation | as C2 (11,699 episodes) on the trained policy | 4–7 | $16–28 |

Basis: 5,200–7,400 generated tokens per episode (hosted Qwen3.5-9B, public), rollouts at 2,000–3,500 tokens/s during
training (ART shares the GPU with the trainer), a step's rollout phase bounded by its longest episode (up to 32,768
tokens), and the LoRA update over **~2.0M tokens per step** (below). The C5-smoke replaces all of these with measured
values; the C5-main projection is recomputed from it before launch (`gpu/budget.py check`).

**Training tokens per step (checked on CPU, 2026-09-28).** ART 0.5.20 assembles each trajectory from vLLM's
`prompt_token_ids` + `token_ids` per turn and merges turns into one sequence only when the next turn's prompt tokens
extend the previous prompt + completion exactly. Our episodes send earlier turns back without their thinking (the
reasoning parser strips it; Qwen3.5's template then renders an empty `<think>` block for them), so the prefixes never
match and **every turn is its own training sequence** with its full prompt: training tokens per trajectory = the
episode's Σ prompt tokens + output tokens = 24.4k (patient_diagnosis), 22.6k (differential), 26.5k (retrieval), 52.5k
(test_selection; p90 142k, as many sequences ≤ 65k) → ~31.5k mean, ~2.0M per 64-trajectory step (less after ART drops
zero-advantage trajectories). At ~40% MFU on one H100 that is ~4–6 min of update per step, so a step is ~10–13 min and
C5-main ≈ 10–13 GPU-hours (≈ $41–53), inside the range above. This is correct training (each turn is scored in the
context the model actually saw), only not the cheapest; passing the thinking back (Qwen3.5's native interleaved
format) would merge turns and cut update cost, but it changes the evaluated context relative to the Stage-8 runs, so it
is kept as is unless the C5-smoke shows the update phase dominating.

## Risks checked in the C5-smoke (before C5-main)

1. **The 9B loads in ART** with `allow_unvalidated_arch` and the explicit targets (the LoRA has parameters in the
   linear-attention layers: count them in the checkpoint's `adapter_model.safetensors`).
2. **Tool calls parse in training** (`qwen3_coder`): the smoke's train episodes have tool calls and nonzero rewards.
3. **Tokenization of the trajectory matches what was sampled.** ART trains on vLLM's own `prompt_token_ids` /
   `token_ids` per turn (`return_token_ids`, injected by ART's client and merged with our `extra_body`) and the
   sampled logprobs, so the scored tokens are the sampled ones by construction; turns become separate sequences (see
   the budget note). ART's client also sends `chat_template_kwargs.preserve_thinking`, which Qwen3.5's template does
   not read (checked: the template keeps a turn's thinking only if the message carries it). Check in the smoke:
   ART's `data/step_trainable_assistant_tokens` ≈ the episodes' output tokens, the importance ratio ≈ 1 on step 1.
4. **Memory:** peak GPU memory in the train phase with 64 trajectories of up to ~40k tokens; if it runs out,
   `--gpu-memory-utilization 0.7`, then `--groups-per-step 4` with `--rollouts-per-group 8`.
5. **Step time** within the projection; the longest-episode tail is what dominates the rollout phase.

## Fallbacks

| failure | fallback | cost of the fallback |
|---|---|---|
| 9B does not load or train in ART | `--base-model Qwen/Qwen3.5-4B` (validated in ART; same architecture) | a new C2 base run for 4B (≈ $10–18) and hosted-4B planning numbers; criteria unchanged |
| out of memory | lower `--gpu-memory-utilization`, then fewer groups per step, then LoRA rank 8 | slower steps |
| ART's vLLM runtime fails on Modal | pin `ART_VLLM_RUNTIME_BIN` to a runtime built in the image by hand from ART's lockfile | image rebuild |
| tokenization mismatch (risk 3) | turn off thinking carry-over: `chat_template_kwargs={"enable_thinking": false}` for training **and** evaluation (a new base C2 run), or train on single-turn arms only | re-baseline |
| ART unusable | TRL `GRPOTrainer` with vLLM colocate and a custom multi-turn rollout over `EpisodeDriver` | a new trainer integration |
