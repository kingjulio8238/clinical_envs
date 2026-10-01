# Stage C: RL pipeline — todo

Scope: RL readiness stage C (audit/RL_READINESS_TODO.md C1–C6) for the round-1 training units (patient_diagnosis,
differential_diagnosis, evidence_retrieval, test_selection; atypical_diagnosis as the held-out transfer test) with
Qwen3.5-9B. All non-GPU work first; every GPU run is smoked first, its cost projected from the smoke, the Modal
workspace headroom checked, and the full run launched only after the resources are confirmed.

## Design decisions (fixed before coding)
- **Trainer: ART (openpipe-art 0.5.20, GRPO + LoRA, vLLM inference, Unsloth training).** Its rollout is ordinary
  Python against an OpenAI-compatible client, so the environment's episode logic is reused as is; the policy's
  choices are recorded in the trajectory. verl / TRL / prime-rl were the alternatives: heavier integration (their own
  agent-loop or environment classes) for the same GRPO.
- **One episode driver for evaluation and training** (`eval/episode.py`): the protocol runner and the RL rollout drive
  the same `EpisodeDriver`, which owns the prompt, the tools, the step rules, the single-arm retry, forced submission,
  the limits and the monitors. The model call is the only thing that differs.
- **Deterministic limits (C3):** turns per unit, output tokens per turn, output tokens per episode; no wall-clock
  deadline in local runs (a per-call safety timeout remains, and a timeout is an error, not a limit).
- **Hardware:** Modal. One H100 80 GB serves Qwen3.5-9B (bf16 ~19 GB) with vLLM for C2/C4; ART (vLLM + LoRA trainer on
  one GPU) for C5. Every Modal job carries a server-side timeout ~1.5x its projection; profiles are pinned per command;
  month-to-date spend is checked before, during and after each run.
- **Sampling:** evaluation (C2 and every before/after) and training use Qwen3.5's recommended thinking settings
  (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5); evaluation with a fixed per-episode seed. Recorded in
  manifests. (Superseded an earlier temperature-0.6 evaluation setting: the evaluated policy is the sampled one.)
- **Serving engine (P3):** ART's managed vLLM runtime for both the before and the after evaluation (the engine that
  samples the policy in training); the trained adapter is loaded into the base server as `qwen3.5-9b-rl`.
- **Operations:** `audit/OCT1_RUNBOOK.md` (order, workspaces, guarded launches) and `audit/RL_TRAINING_PLAN.md`.

## Non-GPU work (first)
- [x] N1 `eval/episode.py`: `EpisodeDriver` + `EpisodeLimits`; `protocol_run.run_episode` rebuilt on it; the existing
      protocol / audit tests pass unchanged (behaviour parity) — done: 88 protocol / audit / task-set tests unchanged and green
- [x] N2 C3 deterministic limits: per-unit turn caps, per-turn and per-episode output-token caps; wall clock off in
      local mode; limits recorded in every manifest; tests — done: `EpisodeLimits` (turns per unit, 4,096 tokens per turn in
      the request, 32,768 per episode), `protocol_run --local`, `finish_reason` recorded (truncated turns)
- [x] N3 C6 monitors (`eval/rl_monitor.py`): per-episode signals (answer length, entries, name length, duplicates,
      named vs coded, tool calls, turns, truncations, probe-like patterns) + batch aggregates + alert rules (reward up
      while heldout flat; answer size growing; probe patterns); tests
- [x] N4 C1 rollout core (`eval/rl_rollout.py`): episode against any OpenAI-compatible client, recording the policy's
      choices; reward from the frozen scorer; `require_frozen()` at start-up; tests with a scripted client
- [x] N5 C1 ART wrapper + training script (`scripts/train_rl.py`): unit mix, groups of k, GRPO steps, heldout eval
      every N steps with monitors, checkpoints; dry-run on CPU with the scripted client — done against the installed
      ART 0.5.20 API (`backend.train(model, groups, learning_rate=...)`, `art.Trajectory(messages_and_choices, tools, reward,
      metrics)`, `model.get_inference_name()`); the ART path itself runs only on a GPU (G5)
- [x] N6 local model config (`qwen3.5-9b-local`, vLLM endpoint) + C2 baseline driver (protocol runner, deterministic
      limits, eval sampling) + C4 group-variance script (k samples per train prompt, zero-variance share, prompt
      filter); tests against a local mock OpenAI server
- [x] N7 Modal apps: vLLM server (C2/C4), ART trainer (C5), each with server-side timeouts; a kill-sweep command;
      the image builds and imports cleanly (image build is CPU) — done: `gpu/vllm_eval.py`, `gpu/train_app.py`,
      `gpu/kill_sweep.sh`, `requirements-rl.txt` (the minimal dependency set, verified on a clean Python 3.12 env: an oracle
      episode of all 10 units scores 1.0). The Modal image build itself is part of G1 (it needs a workspace with headroom)
- [x] N8 docs, tests green, commit

## GPU work (smoke first, then ask)
- [x] G0 Modal: list profiles / workspaces, month-to-date spend per workspace, pick one with headroom
      → 2026-09-28: every workspace is at or past its $30 September credit (highest headroom $2.62, founders-78536, at
      $27.38 MTD — above the $25 stop rule). No GPU run can start before the credits reset on Oct 1 or another source is chosen.
      → The approved smoke ("smoke now on an account with headroom, the rest on Oct 1") was attempted on the two
      workspaces with headroom left: founders-78536 ($2.62) and sales-32662 ($0.94) both refused the job with
      "Workspace ... has exceeded its spend limit" before anything started (no spend; kill sweep: 0 apps running).
      The smoke moves to Oct 1 with the rest of the GPU work.
- [x] G1 smoke: vLLM serves Qwen3.5-9B on Modal; 3 episodes per unit through the protocol runner (tool calls parse,
      rewards score, monitors fill); measure tokens/s and $/episode
      → **GO** (2026-10-01, founders-78536, ~$2.6 incl. two failed starts). 102 episodes, 0 errors, tool calls parse on
        every task, 1 forced answer, 3 truncated turns, 0 limit stops; on the same 73 instances the local means are
        within 0.07 of hosted Qwen (pd 0.318 vs 0.349, dd 0.395 vs 0.447, er 0.537 vs 0.503, ts 0.243 vs 0.306, aty
        0.278 vs 0.319). Fixed on the way: Modal 1.x has no `Function.with_options` (timeout/GPU now set at
        registration from `SH_JOB_MINUTES`/`SH_JOB_GPU`, exported by `gpu/launch.sh`); FlashInfer JIT needs nvcc (CUDA
        12.9 devel base image); JIT starved on Modal's default CPU (8 cores); compile caches moved to the weights volume.
        **Measured throughput:** prefill-bound — 14–18k prompt tokens/s against 1.5–2.6k generated tokens/s at 30–72
        concurrent episodes, GPU 95–100%, KV cache ≤ 26%, **prefix caching off** (vLLM's default for this hybrid
        model). Server start 591 s cold (FlashInfer GDN-prefill JIT; cached for later starts). At the measured rate a
        full evaluation (15,529 episodes) projects to ~16 GPU-hours / ~$70 — 2.5–5x the plan.
- [ ] G2 projection for C2 (all public + heldout instances), C4 (k = 8 on train prompts) and C5; resource ask
- [ ] G3 C2 local baseline ("before") — after approval
- [ ] G4 C4 group variance + prompt filter — after approval
- [ ] G5 C5 training smoke (a few GRPO steps: throughput, memory, cost) — after approval
- [ ] G6 validation, docs, commit

## Resources (projection from the hosted runs — replaced by the G1 smoke's measurements)

| job | work | projection (one H100 on Modal, ~$3.95/h) |
|---|---|---|
| G1 smoke | 3 episodes/unit + ~12/unit at 64 workers, weights download, image build | ~40 min, ≈ $3 |
| C2 baseline | 11,699 episodes (public 5,012 + heldout 6,687, all 10 units); ~50M generated tokens | 4–7 h, ≈ $16–28 |
| C4 group variance | 4 units × 250 train prompts × 8 samples = 8,000 episodes; ~44M generated tokens | 3–5 h, ≈ $12–20 |
| C5 training smoke | 3–5 GRPO steps × 64 rollouts + one heldout pass | ~1–1.5 h, ≈ $4–6 |
| stage C total | | ≈ $35–57 of Modal credit |

Projection basis: hosted Qwen3.5-9B token counts per episode (stored predictions) and an assumed 2,500–5,000 generated
tokens/s for a 9B model at 64–128 concurrent sequences on one H100 with prefix caching — not yet measured. Each job must
fit one workspace's headroom (stop at $25 MTD), so C2 and C4 are split across workspaces by unit if needed. No API
credits are needed for stage C (the OpenAI judge audits of the base model's answers run after C2: ≈ $1).
