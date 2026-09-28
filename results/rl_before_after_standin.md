# RL before / after (audit/RL_SUCCESS_CRITERIA.md)

base `qwen3.5-9b` (results) vs trained `gpt-6-sol` (results); splits public

| criterion | check | status | evidence |
|---|---|---|---|
| 1 | significant gain on ≥ 3 of 4 training units, on public | **PASS** | patient_diagnosis public: +0.185 [+0.083, +0.289] (n=40); differential_diagnosis public: +0.108 [+0.037, +0.181] (n=40); evidence_retrieval public: +0.150 [+0.097, +0.203] (n=40); test_selection public: +0.125 [+0.088, +0.163] (n=314) |
| 2 | public gain ≥ 25% of the base-to-anchor gap on the units counted in 1 | **PASS** | patient_diagnosis: gap +0.185 → bar +0.046, gain +0.185; differential_diagnosis: gap +0.108 → bar +0.027, gain +0.108; evidence_retrieval: gap +0.150 → bar +0.038, gain +0.150; test_selection: gap +0.125 → bar +0.031, gain +0.125 |
| 3a | diagnosis_named rises significantly on patient_diagnosis and test_selection | **PASS** | patient_diagnosis: named +0.250 [+0.075, +0.425] (n=40), coded +0.300 [+0.125, +0.475] (n=40); test_selection: named +0.182 [+0.127, +0.242] (n=314), coded +0.382 [+0.328, +0.439] (n=314) |
| 3b | judge-audited correct-but-0 ≤ 5% on every diagnosis unit (trained) | **PASS** | patient_diagnosis 0.0%; atypical_diagnosis 0.0%; differential_diagnosis 0.0%; test_selection 4.2% |
| 3c | probe patterns < 5%, answer size < 2.0× base | **PASS** | patient_diagnosis: clean; differential_diagnosis: clean; evidence_retrieval: clean; test_selection: clean; atypical_diagnosis: clean |
| 4a | atypical transfer: lower bound > -0.03 | **PASS** | public: +0.272 [+0.141, +0.418] (n=23) |
| 4b | heldout vs public | **N/A** | needs both splits |
| 4c | gain > 0 in the rarest third of training diagnoses (D2) | **PASS** | patient_diagnosis: rarest +0.217 [+0.018, +0.449] (n=14) (T1 +0.217, T2 +0.163, T3 +0.173); differential_diagnosis: rarest +0.191 [+0.081, +0.313] (n=14) (T1 +0.191, T2 +0.108, T3 +0.017); test_selection: rarest +0.076 [+0.019, +0.135] (n=105) (T1 +0.076, T2 +0.155, T3 +0.144) |
| 5 | no regression on untrained units: lower bound > -0.03 | **FAIL** | context_summarization public: +0.073 [+0.026, +0.121] (n=40); specialty_conditioned public: +0.091 [-0.030, +0.207] (n=40); imaging_indication public: +0.031 [-0.021, +0.079] (n=40); lab_triage public: +0.035 [-0.046, +0.114] (n=40); error_detection public: +0.050 [+0.000, +0.125] (n=40) — every mean ≥ 0: the interval is too wide (sample size), not a measured regression |
| 6a | tools − no-tools gap holds (trained gap − base gap, lower bound > −0.03) | **N/A** | needs single-arm runs for base and trained |
| 6b | test_selection orders per episode in (0, 2× base] | **PASS** | base 4.44, trained 4.40 |
| 7 | private split once | **N/A** | run once on the final checkpoint (audit/PRE_OCT1_TODO.md P5) |

**Outcome: inconclusive on criterion 5: untrained units need more instances (every mean ≥ 0, intervals too wide)**
