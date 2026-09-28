# RL readiness — todo (close every gap in audit/RL_READINESS.md before any training run)

Scope: everything between "the environment is verified" (audit/ROADMAP_VERIFICATION.md) and "the first RL run can
start with a result we can trust". Done = every box checked, with evidence, and the readiness gate at the end green.
Ordered by dependency; items in the same group are independent. Costs are compute/API spend, not calendar time.

## A. Reward validity (environment; no GPU)

- [x] **A1 Concept-level diagnosis matching (synonym noise).** 9–21% of Qwen's zero-scored answered diagnoses are
      clinically the same diagnosis under another name (Asherman ↔ intrauterine adhesions, HSP ↔ IgA vasculitis,
      arsenic toxicity ↔ poisoning). Map predicted names/codes to the graph's own diagnosis nodes (`diagnoses`
      display names, SNOMED ids where present, merged-node aliases, ICD descriptions from the CMS file) and credit a
      match to the reference node; keep the token rules as the fallback. Gate: judge-audited correct-but-0 rate
      ≤ 5% on every diagnosis unit (re-run `scripts/reward_noise_audit.py` on the stored predictions, both models);
      the kitchen-sink and hedge probes stay at the floor; floors regenerated.
      *Cost: judge re-audit ≈ $0.5 of OpenAI credits.*
      - done: `eval/diagnosis_aliases.json` (6,911 nodes, 10,749 aliases: CMS description, SNOMED description,
        merged duplicates; `scripts/build_diagnosis_aliases.py`, deterministic), the reference's parenthetical gloss,
        a small synonym lexicon (IgA/immunoglobulin A, synechiae/adhesions, toxicity/overdose/poisoning, CSF, arterial,
        -related, periprosthetic), and near-identical multi-word names across blocks (≥ 2 shared words, Jaccard ≥ 2/3).
        Judge-audited correct-but-0 (GPT-6 Sol judge), before → after: Qwen agent patient_diagnosis 9% → 2.8%,
        atypical 21% → 3.7%, differential 0% → 0%, test_selection 13–15% → 3.1%; GPT-6 Sol 0% on all four;
        out-of-sample check on Qwen's no-tools answers (not used to build the lexicon): 2.7% / – / 0% / 0%.
        Residual cases need anatomy or ontology knowledge (popliteal ⊂ lower-extremity embolism, fourth-ventricle vs
        cerebellar hemangioblastoma, intimate-partner violence vs adult abuse). All probes stay at the floor; floors
        regenerated (unchanged). Judge verdicts vary slightly between runs (GPT-6 accepts only its default temperature).
- [x] **A2 Report the coding share.** Add `diagnosis_named` (name/concept credit ignoring the code) and
      `diagnosis_coded` (exact-code share) as secondary metrics for patient_diagnosis, atypical, differential and
      test_selection, in the scorer output, the leaderboard and the RL logs, so a gain can be split into "named the
      diagnosis" vs "coded it". (Today 23–32% of Qwen's reward is name credit, 1–7% of GPT-6 Sol's.)
      - done: scorer outputs for all four units, `protocol_run` records them per episode (`metrics`), the rescore
        script backfills them, the leaderboard shows `named` / `coded` columns. Qwen vs Sol named / coded:
        patient_diagnosis 0.47 / 0.13 vs 0.75 / 0.48; atypical 0.45 / 0.09 vs 0.87 / 0.57; differential 0.63 / 0.27 vs
        0.78 / 0.63; test_selection 0.50 / 0.13 vs 0.65 / 0.43 — Qwen's coding gap is the larger one.
- [x] **A3 Adversarial probe suite for optimization pressure.** Extend the degenerate policies with the strategies a
      policy under RL is most likely to find, each a CI gate at the floor: many diagnoses per answer (precision
      floor), duplicated entries, the train-label frequency prior with generic names, codes without names and names
      without codes, maximal-length differentials, retrieval of every section, order spam in test_selection.
      Gate: all at or below the floor on public and heldout; any that is not gets a scorer fix first.
      - done: 20 probes (`eval/degenerate.py PROBES`: name sink, hedged name, 50 diagnoses, duplicated entries,
        codes-only, names-only, problem-list names, one mega-order, a passage repeated ten times), CI-gated on public
        and heldout (`test_rl_pressure_probes_stay_at_the_floor`). One exploit found and fixed: retrieval counted a
        duplicated passage at every rank (ten copies of one passage: nDCG@10 0.33 → 0.07 after dedup; no stored
        prediction changed). Every probe now scores ≤ 0.014 (diagnosis units) and 0.073 (retrieval, the single passage it repeats), below each floor.
- [x] **A4 Freeze the reward.** Tag the scorer commit used for training (`reward-v1`), record it in every RL and
      evaluation manifest, and fail the trainer at start-up if the working scorer differs from the tag.
      - done: `eval/reward_version.py` + `eval/reward_lock.json` (`reward-v1`, SHA-256 over the 11 reward files,
        fingerprint 3bc9a76270f49c59); the protocol runner records version and fingerprint in every manifest and
        refuses a drifted scorer (`--allow-reward-drift` is recorded); CI fails when a reward file changes without
        re-locking; git tag `reward-v1`. The trainer (C1) must call `eval.reward_version.require_frozen()` at start-up.

## B. Task set (environment; no GPU)

- [ ] **B1 Re-specify lab_triage.** Both models score below the flag-everything floor (0.63): recall of every
      key/supporting result dominates. Redesign the reward so over-flagging costs (e.g. precision-weighted F1 with
      background results as explicit negatives, or ranked top-k with the most-urgent weight), regenerate floors, and
      re-smoke both models. Gate: the flag-everything floor ≤ 0.3 and the oracle 1.0. Until then it stays out of RL.
      *Cost: re-smoke ≈ $0.1.*
- [ ] **B2 error_detection saturation.** Qwen 0.89, GPT-6 Sol 1.00. Either harden it (multiple or subtler errors,
      localize the sentence, not just section and type) or retire it from RL targets and document why. Also cap its
      episodes by turns: 7 of 120 Qwen episodes looped until the deadline.
- [ ] **B3 Summarization headroom.** Echoing the key finding names scores 0.487 (the floor), Qwen 0.525. Decide
      whether to raise the floor-to-ceiling spread (weight precision / grounding more) or to keep summarization as
      an evaluation-only unit. Document the decision.
- [ ] **B4 test_selection as the agentic unit.** Its anchor gap (+0.09, interval above 0) just missed the 0.10 rule.
      Re-measure it on the full public + heldout sets in the local baseline (C2) before deciding whether it joins the
      second RL round.

## C. RL pipeline (needs a GPU)

- [ ] **C1 Trainer integration.** Wrap `LocalEnv` in the chosen trainer's multi-turn tool loop (verl agent loop,
      prime-rl, or ART): Qwen3.5 chat template and tool-call parser, the same intro, tools, submit tool and budget as
      `eval/protocol_run.py`, the reward from the frozen scorer. Tests: an oracle policy reaches 1.0 through the
      trainer's loop; a scripted failure scores 0; the trainer's episode for a fixed seed equals `run_episode`'s.
- [ ] **C2 Local baseline ("before").** Serve the base weights with the trainer's vLLM and fixed sampling (record
      temperature, top-p, max tokens, thinking on/off); run the protocol on every public and heldout instance of
      every unit; commit as `results/local/qwen3.5-9b-base__*`. Compare with the hosted numbers to size the
      provider/quantization effect.
- [ ] **C3 Deterministic episode limits.** Replace the wall-clock deadline in training and in the before/after
      evaluation with token and turn limits (4,096 tokens per turn; a turn cap near the observed p90, 8–16), used
      identically in C2 and the final evaluation.
- [ ] **C4 Group-variance measurement.** k = 8 samples on a few hundred train prompts per training unit: the share
      of zero-variance groups, pass@k, and the reward spread per prompt; build the prompt filter or curriculum from it.
- [ ] **C5 Throughput and cost smoke.** A short training run on the real hardware (a few optimizer steps) measuring
      generated tokens per second, rollout and update time, memory at the p90 context. Project one full pass over
      the training prompts at k = 8 (~46k rollouts, ~230M generated tokens by the hosted averages) from the measured
      rate, and pick the hardware from that projection.
- [ ] **C6 Training monitors.** Per-episode logs (answer length, number of diagnoses, name length, duplicates,
      name-vs-code credit, tool calls, turns, truncations); evaluation hooks every N steps on a fixed heldout subset;
      the A3 probes and the reward-noise judge on a sample of policy answers at each checkpoint; alerts on reward up
      while heldout flat or name length / entries growing.

## D. Evaluation plan fixed in advance

- [ ] **D1 Pre-register the success criteria** (in this file, before training): trained vs base on the training
      units, paired 95% intervals on public and on heldout; judge-confirmed; decomposed named vs coded; no significant
      regression on untrained units; atypical_diagnosis as the transfer test; the tools ablation re-run on the
      trained model; one confirmation on the private split through the operator scorer.
- [ ] **D2 Frequency stratification.** Report the gain by how often each reference diagnosis occurs in train, so a
      gain that is only prior-learning is visible.

## Readiness gate (all green before the first full training run)

- [ ] A1–A4 done; floors current on all four splits; `pytest eval/tests etl/tests` and the simulator suite green;
      `scripts/verify_roadmap.py` 0 failures
- [ ] B1–B4 decided and documented
- [ ] C1–C6 done on the target hardware, with the cost projection inside the available budget
- [ ] D1–D2 written down before training starts
