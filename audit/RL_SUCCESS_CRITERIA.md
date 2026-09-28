# RL success criteria — pre-registered (D1)

Recorded 2026-09-28, **before any training run**. The first trained checkpoint of Qwen3.5-9B is judged against these
criteria; they are not changed after results are seen. If a criterion turns out to be ill-posed, the change and its
reason are recorded below the line, dated, and the original stays in place.

## What is compared

- **Before:** the base weights `Qwen/Qwen3.5-9B`; **after:** the trained checkpoint (base + LoRA).
- **Same everything else:** one local vLLM setup, identical sampling (Qwen3.5 thinking defaults, per-episode seed),
  identical deterministic limits (`eval/episode.py`), the frozen reward **`reward-v3`**
  (`eval/reward_lock.json`), the same instances.
- **Instances:** every public **and** every heldout instance of every unit (C2 covers the before). Comparisons are
  paired per instance, with 95% bootstrap intervals (`eval/protocol.py`).
- **The hosted numbers below are for planning only.** The thresholds in criterion 2 are recomputed from the local
  base (C2) once it has run, with the same rule (25% of the gap to GPT-6 Sol), and entered in the table then.

Planning baseline (hosted Qwen3.5-9B, public split, Stage 8, rescored under reward-v3):

| unit | Qwen3.5-9B | GPT-6 Sol | Sol − Qwen [95% CI] | Qwen tools − no tools |
|---|---|---|---|---|
| patient_diagnosis | 0.334 | 0.520 | +0.185 [0.083, 0.289] | +0.088 |
| differential_diagnosis | 0.419 | 0.533 | +0.108 [0.037, 0.181] | +0.071 |
| evidence_retrieval | 0.563 | 0.700 | +0.150 [0.097, 0.203] | — |
| test_selection (n = 314) | 0.280 | 0.405 | +0.125 [0.088, 0.163] | +0.178 |
| atypical_diagnosis (held-out transfer test) | 0.351 | 0.647 | +0.272 [0.141, 0.418] | — |

## Primary criteria (all must hold)

1. **Significant gain on the training units.** Trained − base, paired 95% interval above 0, on **at least 3 of the 4
   training units** (patient_diagnosis, differential_diagnosis, evidence_retrieval, test_selection), **on public and
   on heldout**.

2. **A meaningful gain: at least 25% of the base-to-GPT-6-Sol gap** on each of those units.

   | unit | gap (planning, hosted) | minimum gain (planning) | gap (local C2) | minimum gain (final) |
   |---|---|---|---|---|
   | patient_diagnosis | 0.185 | +0.046 | _after C2_ | _after C2_ |
   | differential_diagnosis | 0.108 | +0.027 | _after C2_ | _after C2_ |
   | evidence_retrieval | 0.150 | +0.038 | _after C2_ | _after C2_ |
   | test_selection | 0.125 | +0.031 | _after C2_ | _after C2_ |

   (The local gap uses the local base vs GPT-6 Sol on the Sol sample: 40 per unit, 314 for test_selection.)

3. **The gain is diagnostic, not coding and not gaming.**
   - `diagnosis_named` rises significantly (paired interval above 0) on patient_diagnosis and test_selection. If only
     `diagnosis_coded` rises, the result is reported as "RL taught ICD coding", not as diagnostic improvement.
   - Judge-audited correct-but-scored-0 of the trained model's answers stays **≤ 5%** on every diagnosis unit
     (`scripts/reward_noise_audit.py`).
   - Monitor signals (`eval/rl_monitor.py`) on the trained model's heldout answers: probe patterns (> 10 entries,
     names > 20 words, duplicated entries) in **< 5%** of answers; mean entries, answer size and name length **< 2×**
     the base model's; no "reward up while naming fell" alert.
   - All 20 RL-pressure probes and the Stage-1 exploits stay at their floors (CI).

4. **It generalizes.**
   - **atypical_diagnosis** (never trained on): trained − base **≥ 0** (the lower bound of the paired interval above
     −0.03); a significant gain is the stretch.
   - **Heldout vs public:** the heldout gain is **≥ 70%** of the public gain on each unit counted in criterion 1.
   - **Label frequency (D2):** the gain is **> 0 in the rarest third** of training diagnoses (by how often the
     reference diagnosis occurs in the train split), not only in the common ones.

5. **No regression on the untrained units.** context_summarization, specialty_conditioned, imaging_indication,
   lab_triage and error_detection: the lower bound of the paired interval (trained − base) **> −0.03** on each.

6. **Tool use holds or improves.**
   - The tools − no-tools gap (patient_diagnosis, differential_diagnosis, test_selection) for the trained model is at
     least the base model's (paired intervals reported).
   - test_selection: orders per episode neither collapse to 0 nor exceed 2× the base model's; the parsimony component
     and `order_log` are reported.

7. **Private-split confirmation, once.** The final checkpoint, scored once through the operator's private-split scorer
   (labels in `private/labels_v1.3.db`): the same direction of change as public/heldout, and a significant gain on
   patient_diagnosis. Run once; never used for model selection.

## Reading the outcome

| outcome | verdict | next step |
|---|---|---|
| 1–7 all hold | **RL on this environment boosts the model** | the same recipe on the next candidates (qwen3.5-27b, muse-glimmer-30b); the deferred Kimi row; write-up |
| 1–2 hold, 3 shows mostly coding | partial: **RL taught ICD coding** | report as such; consider weighting naming above coding in the reward (a new reward version) |
| 1 holds on public, 4 fails (heldout or rare labels) | **overfitting / prior-learning** | larger or filtered prompt set (C4), regularization (KL), before scaling |
| gains below the criterion-2 bars | **weak signal** | check group variance first: many all-zero prompts explain it; then steps, learning rate, rollouts per group |
| any criterion-3 alert | **reward hack** | fix the scorer, re-lock the reward, rescore both models, retrain |

**Stretch:** the trained 9B matches or beats GPT-6 Sol on at least one training unit (smallest gaps:
differential_diagnosis, test_selection).

## Model selection during training

Checkpoints are chosen on a **dev set carved from train-split patients that are never trained on** (a fixed, seeded
slice per unit, excluded from the training prompts), never on public, heldout or private. Public and heldout stay
untouched until the final before/after evaluation (criteria 1–6) of the chosen checkpoint; private is used once (7).

---
Amendments (dated, with reasons): none.
