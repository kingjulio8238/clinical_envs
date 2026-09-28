# RL readiness: Qwen3.5-9B on the Synthetic Hospital environment (2026-09-27)

**Verdict.** The *environment* is ready: its rewards, splits, speed and headroom support RL on four units.
The *RL pipeline* is not: no trainer is connected to the environment yet, and the baseline has not been
measured under the serving you will train with. Those two are the blockers; everything else below is
validation or monitoring.

## 1. What was validated (evidence, re-run for this note)

| Requirement | Status | Evidence |
|---|---|---|
| Rewards resist cheap strategies on the target units | yes | every degenerate policy stays at its floor: patient_diagnosis 0.036, differential 0.027, atypical 0.035, retrieval 0.420 (content-blind ranking); CI-gated (`eval/tests/test_reward_hacking.py`) |
| …including under optimization pressure | **fixed today** | an RL-style probe found a new exploit: one "diagnosis name" built from the 3,000 most common label words scored differential **0.27** (floor 0.03) and patient_diagnosis 0.06, through the name-equivalence credit added in Stage 8. Closed with a specificity cap (a more specific name may add qualifiers, not a catalogue of diseases); both probes are now floor policies and a test (`test_kitchen_sink_and_hedged_names_score_the_floor`). Degenerate policies had not found it. Expect RL to find more (§3.8) |
| Train/eval separation | yes | 0 patients and 0 source questions shared between train and public/heldout/private |
| Enough training data | yes | train instances: patient_diagnosis 2,115, retrieval 2,173, differential 1,500, atypical 1,314, test_selection 965 |
| Learning signal (partial credit, not all-or-nothing) | yes | Qwen mean / SD / share scoring 0: patient_diagnosis 0.31 / 0.32 / 38%; atypical 0.27 / 0.32 / 51%; retrieval 0.56 / 0.26 / 5%; differential 0.39 / 0.23 / 12% |
| Headroom a stronger policy reaches | yes | GPT-6 Sol − Qwen, paired: +0.21, +0.39, +0.15, +0.13, all intervals above 0 (`results/rl_decision.md`) |
| Environment cost per rollout | yes | in-process `LocalEnv`: ~1 ms per action including scoring; one shared read-only DB across threads; oracle 1.0; identical to the HTTP env. Generation, not the environment, sets RL throughput |
| Reward determinism | yes | re-scoring every stored prediction reproduces its reward exactly (`scripts/rescore_protocol_runs.py`) |

## 2. What the reward will teach (decide before training)

Rescoring with the name credit switched off:

| unit | Qwen reward | code-only | share from naming the right diagnosis with a wrong code | GPT-6 Sol share |
|---|---|---|---|---|
| patient_diagnosis | 0.305 | 0.229 | 25% | 7% |
| differential_diagnosis | 0.390 | 0.300 | 23% | 1% |
| atypical_diagnosis | 0.271 | 0.183 | 32% | 3% |

A large part of the Qwen–Sol gap is **ICD-10 coding**, not diagnosis. RL will very likely raise reward by learning
codes (name credit 0.75 → exact code 1.0). That is part of the task as specified, but it is not the claim "the model
reasons better clinically". Keep the reward, and report every result decomposed into *named the diagnosis
correctly* (name credit or the LLM judge) and *coded it exactly*.

## 3. Before training (in dependency order)

1. **Connect a trainer.** No rollout adapter exists (only the Harbor export, one Docker stack per task, too heavy for
   RL). Wrap `LocalEnv` (`reset` / `step` / tool schemas / submit tool) in the trainer's multi-turn tool loop (verl
   agent loop, prime-rl, or ART), with Qwen3.5's chat template and tool-call parser. Reuse `eval/protocol_run.py`'s
   episode logic so training and evaluation see identical prompts, tools and rules. Smoke it end to end on a GPU
   with the oracle and a few real rollouts before anything long.
2. **Re-measure the baseline locally.** The hosted Qwen numbers came from a mix of OpenRouter providers (Venice,
   SiliconFlow, DeepInfra, Together) whose quantization and settings may differ from the weights you will train.
   Run the protocol with the base weights under the trainer's vLLM, fixed sampling settings, on **all** public and
   heldout instances of every unit (patient_diagnosis 693 + 905 …). No API cost; this is the "before".
3. **Measure group variance.** GRPO-style methods learn nothing from prompts where all k samples get the same
   reward. With k = 8 on a few hundred train prompts, measure the zero-variance share; drop or down-weight prompts
   that are always 0 or always 1. The hosted run suggests many always-0 prompts on atypical (51% zeros).
4. **Freeze the scorer.** Pin the scorer commit as the reward. Any scorer change after training starts is applied to
   base and trained predictions alike with the rescore script, never to one side.
5. **Deterministic episode limits.** The protocol's 15-minute wall-clock deadline is not reproducible across hardware.
   For RL use token and turn limits: 4,096 tokens per turn, and a turn cap near the observed p90 (8–16 turns) instead
   of the 40-action budget. Thinking loops (seen in error_detection) end at the cap and score 0.
6. **Pick the training units.** Train on patient_diagnosis, differential_diagnosis and evidence_retrieval. Keep
   atypical_diagnosis **evaluation-only**: its variants are masked versions of patient_diagnosis charts, so it is a
   clean transfer and robustness test. test_selection (the most agentic unit, where tools matter most, +0.15) is a
   second-round candidate: its anchor gap did not reach significance.
7. **Budget the compute from a smoke, not from this note.** The hosted episodes averaged about 5k output and 20–30k
   total tokens over 5–8 turns. One pass over ~5,800 train prompts at k = 8 is ~46k rollouts, i.e. roughly 230M
   generated tokens. Measure tokens per second on the actual GPUs with the smoke before committing hardware.
8. **Monitor for new hacks while training.** Log per episode: answer length, number of diagnoses, name length,
   duplicate entries, share of name-credit vs code credit, and number of tool calls. At every evaluation checkpoint,
   re-run the degenerate floors and the reward-noise judge on a sample of the policy's answers, and compare the
   heldout reward curve with the train reward curve.

## 4. After training

1. **Paired evaluation.** Trained vs base, same local serving and sampling, every public and heldout instance of
   every unit, 95% paired bootstrap intervals, normalized against the floors (`eval/protocol.py`).
2. **Is the gain real?** Decompose it into named-correctly vs coded-exactly; run the LLM-judge audit on both
   models' answers; scan for degenerate patterns; stratify the gain by how often each diagnosis appears in train
   (a gain only on frequent labels is prior-learning).
3. **Does it transfer?** atypical_diagnosis (robustness), the untrained units (summarization, imaging, specialty) for
   regressions, and the tools-vs-no-tools ablation on the trained model (did it learn to use the chart?).
4. **Confirm once on the private split** through the operator scorer.
5. **Decide.**
   - Success: a significant paired gain on at least two trained units, on both public and heldout, confirmed by
     the judge, with no significant regression elsewhere. Then run the same recipe on the next candidates
     (qwen3.5-27b, muse-glimmer-30b), add the deferred Kimi row, and write it up.
   - A gain that is only coding, or only on frequent labels, means fixing the reward or the tasks before scaling.

## 5. Known environment limits (not blockers for the chosen units)

- lab_triage: both models score below the flag-everything floor; needs re-specifying before any RL use.
- error_detection: saturated (Qwen 0.89, GPT-6 Sol 1.00).
- Synonym noise: 9–21% of Qwen's zero-scored answered diagnoses are clinically the same diagnosis (e.g.
  intrauterine adhesions for Asherman syndrome); a SNOMED-concept matcher would remove most of it.
- Whole-patient summarization: echoing the key finding names scores 0.487 (floor), close to Qwen's 0.525, so there
  is little headroom there.

## 6. Update after the reward-validity items (A1–A4, `audit/RL_READINESS_TODO.md`)

- **Synonym noise closed to ≤ 5%.** Concept aliases from the release (CMS / SNOMED descriptions, merged duplicates,
  parenthetical glosses) plus a small synonym lexicon: judge-audited correct-but-0 for Qwen 2.8% / 3.7% / 0% / 3.1%
  (patient_diagnosis / atypical / differential / test_selection), 0% for GPT-6 Sol, and 2.7% / – / 0% / 0% on the
  no-tools answers that were not used to build it. Rewards after rescoring: Qwen patient_diagnosis 0.332,
  atypical 0.347, differential 0.408, test_selection 0.273; GPT-6 Sol 0.520 / 0.647 / 0.526 / 0.325. Still GO on
  the same four units (`results/rl_decision.md`).
- **Named vs coded is now a first-class metric** (`diagnosis_named`, `diagnosis_coded`; leaderboard columns).
  Qwen names the reference diagnosis in 47% of patient_diagnosis references but codes 13% exactly (Sol 75% / 48%):
  the coding gap is larger than the naming gap, as §2 predicted.
- **Probe suite:** 20 RL-pressure probes are CI floor gates on public and heldout; one more exploit was found and
  fixed (retrieval counted a duplicated passage at every rank).
- **Frozen reward:** `reward-v1` (`eval/reward_lock.json`, git tag), recorded in every manifest; the runner refuses a
  drifted scorer; CI fails on an un-locked scorer change.
