# Stage 8 RL go/no-go — candidate `qwen3.5-9b`, anchor `gpt-6-sol`, split `public`

A unit qualifies when the candidate's normalized score is in [0.05, 0.80], the anchor − candidate paired difference is ≥ 0.10 with a 95% interval above 0, and the unit's error rate is ≤ 2%. GO needs ≥ 3 units.

| unit | n | candidate raw [95% CI] | normalized | paired n | anchor − candidate [95% CI] | errors | correct-but-0 (audit) | headroom / gap / errors | qualifies |
|---|---|---|---|---|---|---|---|---|---|
| atypical_diagnosis | 77 | 0.271 [0.203, 0.344] | 0.244 | 23 | 0.391 [0.239, 0.560] | 0.0% | 21% | yes / yes / yes | **yes** |
| context_summarization | 120 | 0.525 [0.489, 0.559] | 0.074 | 40 | 0.073 [0.026, 0.121] | 0.0% | — | yes / no / yes | no |
| differential_diagnosis | 120 | 0.390 [0.346, 0.430] | 0.373 | 40 | 0.131 [0.056, 0.208] | 0.0% | 0% | yes / yes / yes | **yes** |
| error_detection | 120 | 0.892 [0.833, 0.942] | 0.786 | 40 | 0.050 [0.000, 0.125] | 5.8% | — | yes / no / no | no |
| evidence_retrieval | 120 | 0.563 [0.517, 0.610] | 0.246 | 40 | 0.150 [0.097, 0.203] | 0.0% | — | yes / yes / yes | **yes** |
| imaging_indication | 120 | 0.249 [0.221, 0.276] | 0.039 | 40 | 0.031 [-0.021, 0.079] | 0.0% | — | no / no / yes | no |
| lab_triage | 120 | 0.425 [0.370, 0.480] | -0.559 | 40 | 0.008 [-0.054, 0.066] | 0.0% | — | no / no / yes | no |
| patient_diagnosis | 120 | 0.305 [0.252, 0.362] | 0.280 | 40 | 0.212 [0.109, 0.316] | 1.7% | 9% | yes / yes / yes | **yes** |
| specialty_conditioned | 120 | 0.595 [0.518, 0.674] | 0.318 | 40 | 0.091 [-0.030, 0.207] | 0.0% | — | yes / no / yes | no |
| test_selection | 120 | 0.247 [0.197, 0.298] | 0.246 | 40 | 0.093 [0.002, 0.178] | 0.0% | 13% | yes / no / yes | no |

**GO**: 4 unit(s) qualify — atypical_diagnosis, differential_diagnosis, evidence_retrieval, patient_diagnosis.
