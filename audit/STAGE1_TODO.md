# Stage 1: reward-hacking regression suite — todo

Scope: ROADMAP Stage 1 only. **Status: DONE 2026-09-27.** Every box checked; validation below.

## Build
- [x] `eval/degenerate.py`: SQLite loaders for the released DB + one function per degenerate policy + one oracle per task
- [x] `eval/floors.py`: floor (best chart-only degenerate) and ceiling (oracle) per unit/metric/split, agentic floor recorded separately, `normalize()`, CLI (`--write/--check/--json`), committed `eval/floors.json` (public, heldout, train)
- [x] `eval/tests/test_reward_hacking.py`: 31 tests; gates in one `GATES` dict; 15 `xfail(strict=True)` each naming the roadmap stage that fixes it
- [x] tool-echo test runs the real `epic_service.get_problem_list` over the SQLite file (aiosqlite), not a replica
- [x] normalized scores surfaced in `eval/report.py` (`<metric>_normalized`; specialty runs routed to their unit; no column when a unit has no headroom)
- [x] CI workflow `.github/workflows/tests.yml`: `eval.floors --check` + `pytest eval/tests etl/tests`
- [x] `pytest.ini` marker `reward_hacking`; `requirements-dev.txt` (aiosqlite)

## Exploits covered
- [x] empty / malformed submission scores 0 on every unit (absent unit: documented as free abstention, no headroom)
- [x] dx: empty predictions counted, not skipped (#19) — xfail S3
- [x] retrieval: one HPI section (#1) — xfail S3
- [x] retrieval: content-blind section-type ranking vs random, P@5 and nDCG@10 (#22) — xfail S3
- [x] retrieval: random-shuffle floor recorded (5 seeds)
- [x] summarization: paste the chart — xfail S3
- [x] summarization: structured-hint names ∩ must-include (#12) — xfail S2
- [x] summarization: negation ("denies fever") — xfail S3
- [x] specialty: stock phrase (#4) — xfail S3; phrase is free on involved items — pass (documented)
- [x] specialty: absent-ness predictable from the name — xfail S3
- [x] specialty: oracle = 1.0 on every involved item (#23) — xfail S3
- [x] dx: regex-copy the problem list (#2) — xfail S4
- [x] dx: echo `view_problem_list` (#10) — xfail S2
- [x] dx: reference + documented secondary conditions (#13) — xfail S3
- [x] dx: bare 3-char categories — xfail S3
- [x] imaging: restate-the-order floor recorded; oracle = ceiling
- [x] hallucination metric flags invented sentences (#14) — xfail S3
- [x] ceilings: oracle reaches 1.0 on dx / summarization / absent / imaging; retrieval ≥ 0.99

## Validation (all run 2026-09-27 on the released benchmark_v1.3.db)
- [x] suite: `16 passed, 15 xfailed in 36.7s` (public split)
- [x] CI command `pytest eval/tests etl/tests -q`: `66 passed, 1 skipped, 15 xfailed in 52.9s`; `eval.floors --check`: current
- [x] xfail flips verified: monkeypatching a fixed-k P@5 → `test_single_passage_cannot_saturate_p5` FAILED [XPASS(strict)] **and** `test_committed_floors_are_current` FAILED; monkeypatching explicit-only abstention → `test_stock_phrase_in_full_summary_is_not_abstention` FAILED [XPASS(strict)]
- [x] `eval/floors.json` public block regenerates byte-identically; heldout and train blocks written
- [x] pre-existing suites unaffected: eval (non-suite) + etl `50 passed, 1 skipped`; epic_sim tests still collect (92)
- [x] report column unit-checked: copy baseline → ≈0.00, best paper model (0.732) → −0.64, oracle → 1.0; no column on the no-headroom unit
- [x] docs: README section, ROADMAP Stage 1 status, STATE.md next move

## Floors (public / heldout / train), chart-only policies
| unit | metric | ceiling | floor | policy |
|---|---|---|---|---|
| patient_diagnosis | w-F1 neutral | 1.000 | 0.836 / 0.841 / 0.851 | copy problem list + profile (agentic floor 1.000 via the tool API) |
| evidence_retrieval | P@5 | 0.999 / 1.000 / 1.000 | 0.995 / 0.993 / 0.998 | one HPI section (negligible headroom) |
| evidence_retrieval | nDCG@10 | 1.000 | 0.808 / 0.799 / 0.807 | section-type prior |
| context_summarization | clinical_f1 | 1.000 | 0.668 / 0.685 / 0.688 | chart dump |
| specialty_involved | conditioned_f1 | 0.991 / 0.992 / 0.990 | 0.478 / 0.469 / 0.464 | abstain by name |
| specialty_absent | abstention_acc | 1.000 | 1.000 | empty answer (no headroom) |
| imaging_indication | concept F1 | 1.000 | 0.217 / 0.176 / 0.182 | restate the order |
