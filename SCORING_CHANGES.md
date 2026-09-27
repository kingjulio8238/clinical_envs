# Scoring changes (Stage 3 of `audit/ROADMAP.md`)

Metric **names** are unchanged so `/score`, `/env`, `eval.report`, `eval/floors.json` and the test suites keep
working; their **definitions** changed as below. Every batch metric is now the mean of per-item scores, so the
reward `/score` returns for one item equals the benchmark aggregate over items. The defects each change closes are
cited from `audit/FINDINGS.md`.

## All tasks
- Missing or malformed answers score 0. Patient diagnosis used to skip empty predictions ([perfect, empty] scored
  1.0, #19); the single-turn runner used to drop API failures from the mean. Failures are now stored as `{}`.

## Patient diagnosis: `weighted_problem_list_f1_neutral`
| Term | Before | After |
|---|---|---|
| code match | binary at the 3-character category (bare `K35` = full credit) | graded: exact 1.0, subcategory 0.75, category 0.5; an unspecified reference (3 characters or `.9`) is fully credited by any code in its category |
| acuity | not scored | matched credit × 0.5 when the reference acuity is known and the predicted acuity is missing or different |
| recall | severity-weighted, binary | severity-weighted sum of credits |
| precision | unmatched predictions in the chart-neutral set dropped from the denominator; neutral set = profile conditions only | same rule; neutral set also covers the conditions the vignettes document (graph `secondary` diagnoses, #13) |
| aggregate | HM(mean weighted recall, mean precision) | mean over items of HM(item weighted recall, item precision) |

## Evidence retrieval: reward `ndcg_10` (was `precision_5`)
- `precision_at_k` divides by k; one submitted passage can fill one slot (was `min(k, len)`, #1).
- Judgments are graded from **section content** (`scripts/regrade_retrieval_by_content.py`): a section's grade is
  the release rule (key 3/2, supporting 2/1, background 1 by edge type) applied to the query diagnosis's findings
  that the section text states with the right polarity (present findings stated, absent findings negated).
  Grades used to be a function of section *type* (#22). Relevant (≥2) share of sections fell from 69% to 20%.
- nDCG@10 is the reward; P@5 is still returned.

## Whole-patient summarization: `clinical_f1`
| Term | Before | After |
|---|---|---|
| recall | substring / 70%-token match, no negation ("denies fever" credited "Fever") | lexical-not-negated, value polarity, or ontology-concept match (`eval/concept_match.py`) |
| precision | none (recall-only; a chart dump scored 0.67) | share of the summary's clinical concepts the record supports (chart text or the patient's annotated finding/diagnosis names) |
| length | none | factor 1 up to 350 words, then 350 / words |
| `clinical_f1` | recall | HM(recall, precision) × length |
| must-include labels | first 20 key findings chronologically (latest visit often absent) | round-robin across encounters, critical first, cap 20 (`scripts/rebalance_must_include.py`) |
| `hallucination_rate` | 15% token overlap with the whole chart (invented sentences scored 0, #14) | sentences whose clinical concepts are all unsupported |

## Specialty-conditioned summarization: `conditioned_f1` / `abstention_accuracy`
- Abstention is the explicit submission field `abstain: true`; text never abstains (a stock phrase appended to a
  full summary used to score 1.0 on every absent item, #4). Wrong abstention on an involved item scores 0.
- Items with no critical finding fall back to complete recall (540 items used to score 0 for any answer, #23).
- Excluded findings implied by the target findings themselves are not counted as leakage.
- Length factor as for whole-patient summaries.
- Absent specialties are resampled per patient in proportion to involvement frequency
  (`scripts/resample_absent_specialties.py`), so the specialty name no longer predicts the label.

## Imaging indication: `clinical_question_concept_f1`
- Reference = graph terms of the ordering encounter (correct diagnosis, distractor differential, key
  symptoms/signs/imaging findings; `scripts/build_imaging_reference.py`), extracted to concepts. The Kimi-authored
  question, written from a profile that carried future history, is scored as `clinical_question_concept_f1_llm`.

## Floors and ceilings
`eval/floors.json` was regenerated for public, heldout, train and private; see `audit/STAGE3_TODO.md` for the
before/after table. The specialty *absent* unit remains binary (a constant `abstain: true` scores 1.0) and is not
normalized.
