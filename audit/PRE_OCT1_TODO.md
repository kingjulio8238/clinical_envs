# Before the Oct 1 GPU runs — todo

Goal: on Oct 1 the only unknowns left are the ones a GPU must measure (throughput, memory, the base model's real
scores). Everything else — how the result is judged, how the trained checkpoint is served and compared, how the budget
is split, what to do when something breaks — is built, tested and written down first. No GPU; no spend except the
optional P8.

## A. Judging the result (audit/RL_SUCCESS_CRITERIA.md)

- [ ] **P1 Before/after evaluator** (`scripts/rl_before_after.py`): from the base and trained run directories, compute
      criteria 1–6 (paired gains with 95% intervals on public and heldout, the 25%-of-gap bars, named vs coded,
      monitor signals vs base, atypical transfer, heldout/public ratio, untrained-unit regressions, tools − no-tools)
      and print a PASS / FAIL table with the outcome row from the criteria. Validated now on a stand-in pair (hosted
      Qwen as "base", GPT-6 Sol as "trained") so the code path is proven before real checkpoints exist.
- [ ] **P2 Frequency stratification (D2)** inside P1: the gain by tercile of how often the reference diagnosis occurs in
      train; criterion 4's "rarest third" check.
- [ ] **P3 Serving the trained checkpoint:** `gpu/vllm_eval.py` serves base + LoRA (`--enable-lora --lora-modules`) from
      the ART checkpoint directory on the results volume; `qwen3.5-9b-local` targets it by name (`SH_VLLM_MODEL`); the
      same limits and sampling as the base run. Tested for configuration (command line, model name routing).
- [ ] **P4 Checkpoint selection on a train-dev split:** `scripts/train_rl.py` validates on a fixed, seeded slice of
      train-split patients excluded from the training prompts (today it validates on a heldout sample — contrary to the
      criteria doc); test that dev patients never appear in training batches.
- [ ] **P5 Private-split confirmation path (criterion 7):** decide and script where the one private run happens, so the
      private labels never leave the operator's machine (recommended: the client and scorer run locally with the
      overlay, against the Modal vLLM server exposed as a web endpoint for that one run).

## B. Training

- [ ] **P6 ART configuration for Qwen3.5-9B:** ART 0.5.20 validates Qwen3.5-4B / 27B, not 9B, and defaults to 4-bit
      training weights. Set `allow_unvalidated_arch`, bf16 LoRA training (`load_in_4bit=False`, so the trained policy
      matches the bf16 vLLM server), `max_seq_length`, GPU sharing (sleep mode) in `scripts/train_rl.py`; write the
      fallbacks if 9B fails on the GPU (Qwen3.5-4B, validated in ART, with its own hosted baseline; or TRL's GRPO).
- [ ] **P7 Training plan** (`audit/RL_TRAINING_PLAN.md`): hyperparameters (rollouts per group 8, groups per step,
      learning rate, KL / clipping), steps and epochs, dev-evaluation cadence, checkpoint retention, stopping rules
      (monitor alerts, dev reward plateau), and the budget per phase.
- [ ] **P8 (optional, ≈ $2–3 of OpenRouter)** early group-variance estimate on hosted Qwen at the training temperature
      (8 samples × 20 train prompts per unit) to size C4 and the prompts per step before the GPU run.

## C. October operations

- [ ] **P9 Budget and workspace plan:** map G1, C2 (split by unit), C4 and C5 onto workspaces so each stays under $25
      month-to-date, including each workspace's one-time costs (image build, 19 GB weight download to its own volume);
      `gpu/mtd.sh` prints month-to-date per profile and treats any billing error as UNKNOWN (never $0); a pre-launch
      check refuses a job whose projection exceeds the workspace's headroom.
- [ ] **P10 Results sync and resume:** pull outputs from each workspace's results volume, merge runs of one unit that
      were split across workspaces (resume by `gt_id`), tested locally on protocol-run directories.
- [ ] **P11 Oct 1 runbook** (`audit/OCT1_RUNBOOK.md`): the ordered commands (G1 → projection → C2 / C4 → C5) with pinned
      profiles, month-to-date checks before / during / after, timeouts, kill sweep, and the go / no-go after the smoke.
- [ ] **P12 Verification:** `scripts/verify_roadmap.py` gains checks for the stage-C / D artifacts; tests, CI green;
      committed and pushed.

## D. Needs you

- [ ] **U1** Set each Modal workspace's usage limit to $30 at `modal.com/settings/<workspace>/usage` (dashboard only;
      it is what actually prevents out-of-pocket spend).
- [ ] **U2** Confirm the private-split path (P5).
- [ ] **U3** Approve or skip P8 (≈ $2–3 of the $3.32 OpenRouter balance).
