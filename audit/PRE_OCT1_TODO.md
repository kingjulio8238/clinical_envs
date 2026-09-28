# Before the Oct 1 GPU runs — todo

Goal: on Oct 1 the only unknowns left are the ones a GPU must measure (throughput, memory, the base model's real
scores). Everything else — how the result is judged, how the trained checkpoint is served and compared, how the budget
is split, what to do when something breaks — is built, tested and written down first. No GPU; no spend except the
optional P8.

## A. Judging the result (audit/RL_SUCCESS_CRITERIA.md)

- [x] **P1 Before/after evaluator** (`scripts/rl_before_after.py`): from the base and trained run directories, compute
      criteria 1–6 (paired gains with 95% intervals on public and heldout, the 25%-of-gap bars, named vs coded,
      monitor signals vs base, atypical transfer, heldout/public ratio, untrained-unit regressions, tools − no-tools)
      and print a PASS / FAIL table with the outcome row from the criteria. Validated now on a stand-in pair (hosted
      Qwen as "base", GPT-6 Sol as "trained") so the code path is proven before real checkpoints exist.
- [x] **P2 Frequency stratification (D2)** inside P1: the gain by tercile of how often the reference diagnosis occurs in
      train; criterion 4's "rarest third" check.
      → P1 + P2 done: `scripts/rl_before_after.py` (criteria 1–7, PASS / FAIL / N/A with evidence, the outcome row,
        markdown + JSON). Stand-in (hosted Qwen as base, GPT-6 Sol as "trained", public only):
        `results/rl_before_after_standin.md` — criteria 1, 2, 3a–3c, 4a, 4c, 6b PASS; 4b / 6a / 7 N/A (inputs absent);
        5 correctly reported as underpowered (n = 40, every mean ≥ 0). Tests: the stand-in, an identical pair (→ weak
        signal), a synthetic padded-answer hack (→ 3c FAIL, "reward hack"), balanced rarity terciles. Monitor signals are
        now backfilled for the stored runs by the rescore script, and a long ranked retrieval list is no longer a
        "many entries" probe (it is scored at a fixed k).
- [x] **P3 Serving the trained checkpoint:** `gpu/vllm_eval.py` serves base + LoRA from the ART checkpoint directory on
      the results volume; the trained policy is requested by its own name; the same limits and sampling as the base run.
      → done, with a change of engine: plain vLLM 0.30 was the evaluation engine, but LoRA on Qwen3.5's linear-attention
        (Gated DeltaNet) modules is what ART patches into its own pinned runtime (vLLM 0.25.1 + `art_vllm_runtime`,
        "validated" for Qwen3.5 dense). Both sides now run on **ART's runtime — the engine the policy is sampled from
        during training**: one server starts on the base (`gpu/serving.py launch_config`: LoRA enabled, room for one
        adapter of the checkpoint's rank, the training parsers, `generation_config=vllm` as ART sets it); for the after
        run the adapter is loaded into it at run time (`/v1/load_lora_adapter`) as `qwen3.5-9b-rl`. Guards: the job
        refuses to start unless `/v1/models` shows the adapter at that path, and a greedy logprob probe (base vs adapter,
        one fixed prompt) must differ — a zero gap means requests for the adapter are answered by the base.
        `eval/config.py qwen3.5-9b-rl` (same limits and sampling as `qwen3.5-9b-local`); `SH_VLLM_MODEL` is set only when
        an adapter is loaded, so `--model qwen3.5-9b-rl` against a base-only server fails instead of scoring the base.
        The runtime is installed at image-build time (`with_art_runtime`, CUDA profile pinned to cuda12 so the build and
        the GPU container agree on its hash) in both the evaluation and the training images. Checkpoint path:
        `/results/rl/<run>/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/<step:04d>`. Test:
        `test_trained_checkpoint_serving_configuration`; the launch command was built with ART 0.5.20's own
        `build_vllm_runtime_server_cmd`. Not verifiable without a GPU: that the runtime serves Qwen3.5-9B and loads the
        adapter — G1 now smokes the base on this engine, and the first after-run's `served.json` records the probe gap.
- [x] **P4 Checkpoint selection on a train-dev split:** `scripts/train_rl.py` validates on a fixed, seeded slice of
      train-split patients excluded from the training prompts (today it validates on a heldout sample — contrary to the
      criteria doc); test that dev patients never appear in training batches.
      → done: `eval/rl_rollout.py` dev patients = sha256("rl-dev:<id>") % 10 == 0 (52 of 600 train patients; 78–175 dev
        instances per unit); `train_prompts` excludes them from every unit, `dev_instances` serves checkpoint selection;
        `train_rl.py` logs `dev_reward` / `dev_by_unit`; C4 samples non-dev prompts only; test
        `test_dev_patients_are_never_training_prompts`.
- [x] **P5 Private-split confirmation path (criterion 7):** decide and script where the one private run happens, so the
      private labels never leave the operator's machine (recommended: the client and scorer run locally with the
      overlay, against the Modal vLLM server exposed as a web endpoint for that one run).
      → approach written, script waits on U2. Size: the private split has 6,184 instances; criterion 7 needs the 5 RL
        units (patient_diagnosis 711, differential 686, retrieval 733, test_selection 302, atypical 406 = 2,838) for
        the base and the trained policy on one server (adapter loaded, P3) ≈ 5,700 episodes ≈ $8–14 of GPU (W9).
        Recommended path: a `serve` mode of `gpu/vllm_eval.py` (`@modal.web_server`, a bearer token from a Modal
        Secret, the same `launch_config`, server-side timeout) and `eval.protocol_run --split private` run on this
        machine with the overlay, pointed at it by `SH_VLLM_URL`; the model only ever receives prompts built from the
        release DB (which holds no private labels), and the scorer runs locally. The laptop must stay online for the
        run; a disconnect costs at most the capped server, and the run resumes by `gt_id`.
      → U2 confirmed (local scorer). Built: `gpu/private_serve.py` (`modal serve`: ART's runtime on the base with a
        bearer token from the Modal secret `clinical-envs-vllm-key`, the adapter from `SH_LORA_PATH` loaded once up,
        telemetry; scales down after 10 min idle, 6 h lifetime cap, and the app dies with the local `modal serve`
        command); `scripts/run_private.sh` (refuses without the overlay; checks both names are served and that the
        adapter's greedy logprobs differ from the base's; runs base then trained on the 5 RL units' private instances
        with `--retry-errors`, into `results/local/private` for `rl_before_after.py --private`); the registry sends the
        token only when `SH_VLLM_API_KEY` is set. Test: `test_private_endpoint_token_is_sent_only_when_set`.

## B. Training

- [x] **P6 ART configuration for Qwen3.5-9B:** ART 0.5.20 validates Qwen3.5-4B / 27B, not 9B, and defaults to 4-bit
      training weights. Set `allow_unvalidated_arch`, bf16 LoRA training (`load_in_4bit=False`, so the trained policy
      matches the bf16 vLLM server), `max_seq_length`, GPU sharing (sleep mode) in `scripts/train_rl.py`; write the
      fallbacks if 9B fails on the GPU (Qwen3.5-4B, validated in ART, with its own hosted baseline; or TRL's GRPO).
      → done (`scripts/train_rl.py art_configs / lora_config`, keys checked against ART 0.5.20's TypedDicts), with three
        findings from reading ART's source:
        1. ART's vLLM server defaults to the `hermes` tool parser; Qwen3.5 emits `<tool_call><function=...>` XML →
           `tool_call_parser=qwen3_coder`, `reasoning_parser=qwen3` (without this, no tool call would parse in training).
        2. For an unvalidated model ART falls back to generic LoRA targets, which would leave the linear-attention
           (Gated DeltaNet) projections of 3 in 4 layers untrained → the validated Qwen3.5 dense targets set explicitly
           (`in_proj_qkv`, `in_proj_z`, `out_proj` + attention + MLP); 9B has the same architecture as the validated 4B.
        3. ART's trainer refuses a Choice without token log-probabilities → training rollouts request `logprobs`.
        Also: bf16 LoRA (ART defaults to 4-bit training weights), 65,536-token sequences (the model default is 262,144),
        vision off. ART runs its own pinned vLLM (0.25.1, installed with uv on first use); the before/after evaluation
        uses `gpu/vllm_eval.py` for both sides. Fallbacks (Qwen3.5-4B; TRL) are in the training plan (P7). Test:
        `test_art_configuration_for_qwen35_9b`.
- [x] **P7 Training plan** (`audit/RL_TRAINING_PLAN.md`): hyperparameters (rollouts per group 8, groups per step,
      learning rate, KL / clipping), steps and epochs, dev-evaluation cadence, checkpoint retention, stopping rules
      (monitor alerts, dev reward plateau), and the budget per phase.
      → done: `audit/RL_TRAINING_PLAN.md` (setup, data and the C4 pool rule, C5-smoke / C5-main schedule, the order to
        change hyperparameters, stopping rules, checkpoints, budget per phase, the five risks the C5-smoke checks —
        incl. the thinking / tokenization mismatch —, fallbacks with their costs). Implemented in `scripts/train_rl.py`:
        a step-0 dev evaluation of the base (the monitor baseline and the reference score), selection on the training
        units' macro dev reward with `best.json`, stopping on a dev plateau (`--patience 3`) or a repeated monitor alert
        (`--alert-repeats 2`), checkpoint pruning to the evaluated steps, resume with the logged history, ART loss
        settings (`--loss-fn cispo`, `--kl-coef`, `--epsilon`), the C4 keep / drop pool (`--min-keep 400`). Tests:
        `test_checkpoint_selection_and_stopping_rules`, the dry run (step 0 + best.json), the prompt-pool modes.
- [x] **P8 (optional, ≈ $2–3 of OpenRouter)** early group-variance estimate on hosted Qwen at the training temperature
      (8 samples × 20 train prompts per unit) to size C4 and the prompts per step before the GPU run.
      → skipped (U3): C4 on Oct 1 measures it on the real engine; the keep / drop pool rule covers a low spread share.

## C. October operations

- [x] **P9 Budget and workspace plan:** map G1, C2 (split by unit), C4 and C5 onto workspaces so each stays under $25
      month-to-date, including each workspace's one-time costs (image build, 19 GB weight download to its own volume);
      `gpu/mtd.sh` prints month-to-date per profile and treats any billing error as UNKNOWN (never $0); a pre-launch
      check refuses a job whose projection exceeds the workspace's headroom.
      → done: the plan is `audit/OCT1_RUNBOOK.md` §1 (9 workspaces, $25 usable each; G1 + C5-smoke on W1, C2 in two
        shards on W2/W3, C4 on W4, C5-main in segments on W5–W7, the after-evaluation on W7/W8, W9 in reserve; ≈ $95–150
        of $225 projected, ≈ $0.5 one-time per workspace). `gpu/budget.py` (in Python rather than a shell script):
        `mtd` = daily report of the complete days + hourly report of today (a daily report omits today; hourly reports
        cannot span > 7 days — both found by querying), per workspace; any failed or unreadable query is UNKNOWN.
        `check` refuses when MTD is UNKNOWN, when MTD + projection > $25, or when MTD + the worst case (the `--minutes`
        timeout at the GPU's rate) > $30. `gpu/launch.sh` is the launch path: refuses a command without `--minutes`,
        runs the check, pins the profile, prints `modal profile current`, launches detached. `gpu/project.py` turns a
        measured job (G1) into the dollars and timeout for the next one. Verified live today: every workspace reads
        MTD ≥ $27 (all STOP) and `launch.sh` refused. Tests: `eval/tests/test_gpu_budget.py` (MTD sums days + today's
        hours, the 1st of the month, a failing billing CLI → UNKNOWN → refused, the decision rules, the projection).
- [x] **P10 Results sync and resume:** pull outputs from each workspace's results volume, merge runs of one unit that
      were split across workspaces (resume by `gt_id`), tested locally on protocol-run directories.
      → done: `scripts/sync_runs.py` — `pull` (a job's outputs from one workspace's volume → `results/modal/<profile>/`),
        `push` (a stopped run's partial directory to the same path on another workspace's volume, so the same command
        resumes there by `gt_id`), `merge` (every piece of each run → one directory per run: refuses pieces whose
        validity-determining settings differ — model, limits, sampling, prompt hash, reward fingerprint —, one record per
        `gt_id` preferring one without an error, the union checked against the unit's full sample, sources / duplicates /
        errors / missing in the manifest, exit 1 when incomplete), `merge-gv` (C4 shards → one `prompts.json`).
        `eval.protocol_run --shard i/k` and `--retry-errors`; `scripts/group_variance.py --shard i/k --retry-errors`
        (deduplicated summaries). The `modal volume get/put` path semantics were checked on a throwaway volume (created
        and deleted; no compute). Tests: `eval/tests/test_sync_runs.py` (shards merge to the unsharded run's rewards;
        an errored run resumed elsewhere; mismatched limits refused; a half sample flagged; C4 shards).
- [x] **P11 Oct 1 runbook** (`audit/OCT1_RUNBOOK.md`): the ordered commands (G1 → projection → C2 / C4 → C5) with pinned
      profiles, month-to-date checks before / during / after, timeouts, kill sweep, and the go / no-go after the smoke.
      → done: rules in force; workspace plan; pre-launch sweep; G1 (weights download on CPU, 3 episodes per unit + 16
        per training unit, the go / no-go list incl. "within ±0.15 of hosted Qwen"); C2 + no-tools arms in shards and
        C4 in parallel, monitoring, resume-elsewhere, merge; C5-smoke with the plan's five risks and the P3 adapter
        check on its step-3 checkpoint; C5-main in segments moved between workspaces; the after-evaluation on the
        `best.json` step; the verdict command; clean-up.
- [x] **P13 Observability (added on request):** every RL training run and GPU evaluation inspectable at any time
      while it runs, esp. the losses and key metrics, from a terminal.
      → done: `eval/run_log.py` — one formatted stdout line per event (streamed by `modal app logs -f`) and the same
        event in `events.jsonl`: episode progress with per-unit means and ETA, every training step with ART's losses
        (loss, entropy, KL, grad norm, importance ratio mean / p95, clipped fraction), reward ± sd per unit, groups
        without reward spread, tokens/s, timings; dev evaluations with best ★ and alerts; errors immediately with the
        instance id; STOP / FATAL / end. `gpu/telemetry.py` — GPU, vLLM (running / waiting / KV cache / tokens/s /
        preemptions) and host every 15 s in both Modal apps; a failed vLLM start prints its log tail; `job.json` records
        a failed job's error; volume commits every 30 s. `scripts/rl_watch.py` — a live terminal dashboard after
        PufferLib's (`pufferl.py print_dashboard`: summary, losses, env stats, performance, utilization, log tail),
        reading the volume while the job runs plus the live `modal app logs` stream; red on errors / stop / 5 min of
        silence; `--once` for scripted polls. PufferLib's dashboard is part of its own PPO trainer, so the pattern is
        reused rather than the library. Verified: a dry run rendered from local files and from a throwaway Modal volume.
        Tests: `eval/tests/test_run_logs.py`.
- [x] **P12 Verification:** `scripts/verify_roadmap.py` gains checks for the stage-C / D artifacts; tests, CI green;
      committed and pushed.
      → done: 16 new items (C1–C6, the GPU runs as deferred, D1 pre-registration, P1–P4, P6, P7, P9 incl. the live
        launch-guard refusal, P10, P7+P11 docs, P13). Local run: 109 pass, 0 fail except CI still running on the
        newest commit, 2 deferred (Kimi row by decision; G1–G6 on Oct 1); 323 tests passed, 0 failed. CI green on
        every earlier commit of this batch (e43dabd, 8f0664f, dbb1dd6).

## D. Needs you

- [ ] **U1** Set each Modal workspace's usage limit to $30 at `modal.com/settings/<workspace>/usage` (dashboard only;
      it is what actually prevents out-of-pocket spend).
- [x] **U2** Confirm the private-split path (P5). → local scorer against a Modal endpoint.
- [x] **U3** Approve or skip P8 (≈ $2–3 of OpenRouter) → skip.
