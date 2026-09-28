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

- [x] **B1 Re-specify lab_triage.** Both models score below the flag-everything floor (0.63): recall of every
      key/supporting result dominates. Redesign the reward so over-flagging costs (e.g. precision-weighted F1 with
      background results as explicit negatives, or ranked top-k with the most-urgent weight), regenerate floors, and
      re-smoke both models. Gate: the flag-everything floor ≤ 0.3 and the oracle 1.0. Until then it stays out of RL.
      *Cost: re-smoke ≈ $0.1.*
      - done. Two faults, not one: (1) the labels name interpretations ("Thrombocytopenia", "Severe hypertension")
        while clinicians name analytes ("Platelet count", "Blood pressure"), so correct answers missed; (2) F1 rewarded
        flagging everything (70% of results are relevant). Each lab_triage row now carries its documented results
        (name, value, relevant, context tokens of the lab/vitals clause documenting it; `build_stage7_tasks.py
        --patch-triage-results`, 1,248 rows, release + overlay); a flagged name reaches a result by its label or its
        analyte (`order_matches` + vital-sign analytes); `triage_score = 0.6 x Youden's J + 0.4 x urgent hit`, so
        flagging everything, nothing or a random subset earns 0 on the flagging term; the prompt says so. The old
        flag-all floor policy read the label names — it is now a privileged gate (≤ 0.12 J), and the floor is the best
        chart-reading policy. Public floor 0.63 → **0.23**, oracle 1.00. Full re-run (new prompt): Qwen 0.551
        (normalized 0.42), GPT-6 Sol 0.607; paired gap 0.035 [−0.05, 0.11] → valid reward with headroom, not an RL
        target yet (gap rule). Old runs kept under results/superseded/b1_lab_triage_f1.
- [x] **B2 error_detection saturation.** Qwen 0.89, GPT-6 Sol 1.00. Either harden it (multiple or subtler errors,
      localize the sentence, not just section and type) or retire it from RL targets and document why. Also cap its
      episodes by turns: 7 of 120 Qwen episodes looped until the deadline.
      - decided: **retired from RL targets, kept as an evaluation-only regression check.** Evidence: GPT-6 Sol 1.00,
        Qwen 0.89 (normalized 0.79); both models already localize the injected error (37/40 descriptions quote the
        changed text), so a localization term would not add headroom; the hardening that would (clean no-error
        controls, subtler error types) risks label noise from the source charts' own inconsistencies and is recorded
        as future work. Turn cap: `UNIT_MAX_TURNS["error_detection"] = 16` (Qwen median 5, p90 9; loops ran 14–42),
        recorded in each manifest; the global deterministic limits come with C3.
- [x] **B3 Summarization headroom.** Echoing the key finding names scores 0.487 (the floor), Qwen 0.525. Decide
      whether to raise the floor-to-ceiling spread (weight precision / grounding more) or to keep summarization as
      an evaluation-only unit. Document the decision.
      - decided: **evaluation-only for RL round 1.** The 0.487 floor came from a policy that echoes the rubric's
        own finding names — information no model is served since Stage 2 — so floors now exclude such privileged
        policies (`eval.floors.PRIVILEGED_POLICIES`, recorded as `privileged_floor`, still CI-gated ≤ 0.5). Real
        floor 0.435 (chart dump); Qwen normalized 0.16, GPT-6 Sol 0.33, paired gap 0.073 [0.03, 0.12] < 0.10 → it
        does not qualify. Widening its headroom would mean a new summarization reward (a separate project); it stays
        a regression check.
- [x] **B4 test_selection as the agentic unit.** Its anchor gap (+0.09, interval above 0) just missed the 0.10 rule.
      Re-measure it on the full public + heldout sets in the local baseline (C2) before deciding whether it joins the
      second RL round.
      - done (full public split, 314 instances each; $1.64 OpenRouter, $10.4 OpenAI credits): Qwen 0.273,
        GPT-6 Sol 0.402, paired gap **0.129 [0.090, 0.167]**, Qwen errors 1.3% → **qualifies; joins RL round 1** as the
        agentic unit (tools vs no tools +0.15 for Qwen). The larger sample exposed more synonym misses; fixed in the
        concept matcher (embolus/embolism, accents, qualifier-only specificity, the prediction's own gloss, reference
        alternatives "A or B" and workup tails): judge-audited correct-but-0 Qwen 8.3% → 1.1% (n=91). GPT-6 Sol's
        residual is 8% (4 of 50 zeros): all is-a cases — a specific answer to a generic reference (Serratia
        bacteremia for "Gram-negative rod infection", Hollenhorst plaque for "Cholesterol embolism") that token rules
        cannot see (see "Open decision" below).

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

- [x] A1–A4 done; floors current on all four splits; `pytest eval/tests etl/tests` and the simulator suite green;
      `scripts/verify_roadmap.py` 0 failures
      → 2026-09-28: `scripts/verify_roadmap.py` 87 pass / 0 fail / 1 deferred (Kimi row); 283 tests passed; CI green at the reward-v1 commit
- [x] B1–B4 decided and documented → RL round-1 units: patient_diagnosis, atypical_diagnosis (evaluation-only transfer
      test per §3.6 of RL_READINESS.md), differential_diagnosis, evidence_retrieval, test_selection; evaluation-only:
      lab_triage, error_detection, context_summarization, specialty_conditioned, imaging_indication
      → 2026-09-28: `scripts/verify_roadmap.py` 94 pass / 0 fail / 1 deferred (Kimi row); 292 tests passed; CI green; reward-v3

## Open decision (from B4) — closed 2026-09-28 ("close all gaps before we proceed")

- [x] **Is-a matches.** 4 of GPT-6 Sol's 50 zero-scored test_selection answers were correct, more specific forms of a
      generic reference. Built `eval/diagnosis_isa_aliases.json` (`scripts/build_isa_aliases.py`): GPT-6 Sol listed the
      specific forms (is-a only: subtypes, organisms, sites; never causes, complications or differentials) of **every**
      surviving diagnosis node, so private references are not singled out — 3,221 nodes, 9,839 forms, $1.6 of OpenAI
      credits, cached per batch, frozen into the reward (`reward-v3`). A match earns the related credit (0.5), never
      more. Over-credit audit (`scripts/isa_overcredit_audit.py`, the judge on every stored answer the table newly
      credits): 33 answers, 18 same / 14 related / 1 different → **3.0%** (≤ 10% gate). Two more lexical fixes from the
      same audit ("secondary malignant neoplasm" = metastatic; a name inside a specific form). Correct-but-0 now ≤ 5%
      on every unit for both models: Qwen 2.9 / 3.7 / 0 / 2.4%, GPT-6 Sol 0 / 0 / 0 / 4.2%, Qwen no-tools 2.7 / – / 0 / 0%.
      Residual cases need knowledge no table here holds (a Hollenhorst plaque is a retinal cholesterol embolus).

