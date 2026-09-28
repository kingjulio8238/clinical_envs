# Stage 8 RL go/no-go — candidate `qwen3.5-9b`, anchor `gpt-6-sol`, split `public`

A unit qualifies when the candidate's normalized score is in [0.05, 0.80], the anchor − candidate paired difference is ≥ 0.10 with a 95% interval above 0, and the unit's error rate is ≤ 2%. GO needs ≥ 3 units.

| unit | n | candidate raw [95% CI] | normalized | paired n | anchor − candidate [95% CI] | errors | correct-but-0 (audit) | headroom / gap / errors | qualifies |
|---|---|---|---|---|---|---|---|---|---|
| atypical_diagnosis | 77 | 0.351 [0.282, 0.425] | 0.327 | 23 | 0.272 [0.141, 0.418] | 0.0% | 4% | yes / yes / yes | **yes** |
| context_summarization | 120 | 0.525 [0.489, 0.559] | 0.159 | 40 | 0.073 [0.026, 0.121] | 0.0% | — | yes / no / yes | no |
| differential_diagnosis | 120 | 0.419 [0.378, 0.458] | 0.403 | 40 | 0.108 [0.037, 0.181] | 0.0% | 0% | yes / yes / yes | **yes** |
| error_detection | 120 | 0.892 [0.833, 0.942] | 0.786 | 40 | 0.050 [0.000, 0.125] | 5.8% | — | yes / no / no | no |
| evidence_retrieval | 120 | 0.563 [0.517, 0.610] | 0.246 | 40 | 0.150 [0.097, 0.203] | 0.0% | — | yes / yes / yes | **yes** |
| imaging_indication | 120 | 0.249 [0.221, 0.276] | 0.039 | 40 | 0.031 [-0.021, 0.079] | 0.0% | — | no / no / yes | no |
| lab_triage | 120 | 0.551 [0.497, 0.607] | 0.417 | 40 | 0.035 [-0.046, 0.114] | 0.0% | — | yes / no / yes | no |
| patient_diagnosis | 120 | 0.334 [0.279, 0.390] | 0.309 | 40 | 0.185 [0.083, 0.289] | 1.7% | 3% | yes / yes / yes | **yes** |
| specialty_conditioned | 120 | 0.595 [0.518, 0.674] | 0.318 | 40 | 0.091 [-0.030, 0.207] | 0.0% | — | yes / no / yes | no |
| test_selection | 314 | 0.280 [0.247, 0.313] | 0.279 | 314 | 0.125 [0.088, 0.163] | 1.3% | 2% | yes / yes / yes | **yes** |

**GO**: 5 unit(s) qualify — atypical_diagnosis, differential_diagnosis, evidence_retrieval, patient_diagnosis, test_selection.
