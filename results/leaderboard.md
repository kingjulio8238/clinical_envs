# Leaderboard — split `public` (EVAL_PROTOCOL.md)

Rankings use the normalized score `(raw − floor) / (ceiling − floor)`; intervals are 95% bootstrap over instances; Δ vs best is a paired difference on the instances both models scored. Failures score 0 and are counted in `n`. The data generator (Kimi) is shown below the line and is not ranked.


## atypical_diagnosis — arm `agent`  (floor 0.035, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 23 | 0.647 [0.533, 0.761] | 0.634 [0.515, 0.752] | best | 0.870 | 0.565 | 0 | 0 | 6.3 | 0.35 |
| qwen3.5-9b | 77 | 0.347 [0.279, 0.422] | 0.323 [0.253, 0.401] | -0.272 [-0.418, -0.141] * | 0.455 | 0.091 | 0 | 2 | 10.4 | 0.37 |

## context_summarization — arm `agent`  (floor 0.487, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.619 [0.576, 0.666] | 0.258 [0.175, 0.349] | best | — | — | 0 | 0 | 11.1 | 1.22 |
| qwen3.5-9b | 120 | 0.525 [0.489, 0.559] | 0.074 [0.004, 0.141] | -0.073 [-0.121, -0.026] * | — | — | 0 | 0 | 7.4 | 0.30 |

## differential_diagnosis — arm `agent`  (floor 0.027, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.526 [0.460, 0.596] | 0.513 [0.445, 0.584] | best | 0.775 | 0.625 | 0 | 0 | 6.2 | 0.57 |
| qwen3.5-9b | 120 | 0.408 [0.365, 0.449] | 0.392 [0.348, 0.434] | -0.109 [-0.183, -0.039] * | 0.633 | 0.267 | 0 | 0 | 6.6 | 0.31 |

## differential_diagnosis — arm `single`  (floor 0.027, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 120 | 0.340 [0.298, 0.383] | 0.322 [0.278, 0.366] | best | 0.542 | 0.250 | 0 | 8 | 1.0 | 0.06 |

## error_detection — arm `agent`  (floor 0.494, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | best | — | — | 0 | 0 | 5.2 | 0.50 |
| qwen3.5-9b | 120 | 0.892 [0.833, 0.942] | 0.786 [0.671, 0.885] | -0.050 [-0.125, 0.000] | — | — | 7 | 2 | 5.0 | 0.32 |

## evidence_retrieval — arm `agent`  (floor 0.420, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.700 [0.626, 0.764] | 0.483 [0.355, 0.593] | best | — | — | 0 | 0 | 7.4 | 0.90 |
| qwen3.5-9b | 120 | 0.563 [0.517, 0.610] | 0.246 [0.166, 0.327] | -0.150 [-0.203, -0.097] * | — | — | 0 | 0 | 7.5 | 0.34 |

## imaging_indication — arm `agent`  (floor 0.219, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.289 [0.240, 0.344] | 0.090 [0.027, 0.161] | best | — | — | 0 | 0 | 5.7 | 0.68 |
| qwen3.5-9b | 120 | 0.249 [0.221, 0.276] | 0.039 [0.003, 0.074] | -0.031 [-0.079, 0.021] | — | — | 0 | 0 | 6.6 | 0.36 |

## lab_triage — arm `agent`  (floor 0.631, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.487 [0.392, 0.582] | -0.390 [-0.648, -0.134] | best | — | — | 0 | 0 | 3.9 | 0.50 |
| qwen3.5-9b | 120 | 0.425 [0.370, 0.480] | -0.559 [-0.707, -0.410] | -0.008 [-0.066, 0.054] | — | — | 0 | 4 | 7.7 | 0.48 |

## patient_diagnosis — arm `agent`  (floor 0.036, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.520 [0.408, 0.620] | 0.502 [0.386, 0.606] | best | 0.750 | 0.475 | 0 | 0 | 6.3 | 0.61 |
| qwen3.5-9b | 120 | 0.332 [0.278, 0.389] | 0.308 [0.251, 0.366] | -0.185 [-0.289, -0.083] * | 0.467 | 0.133 | 2 | 0 | 7.0 | 0.34 |

## patient_diagnosis — arm `single`  (floor 0.036, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 120 | 0.246 [0.196, 0.296] | 0.218 [0.166, 0.270] | best | 0.358 | 0.092 | 1 | 17 | 1.0 | 0.08 |

## specialty_conditioned — arm `agent`  (floor 0.407, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 120 | 0.595 [0.518, 0.674] | 0.318 [0.187, 0.450] | best | — | — | 0 | 0 | 9.3 | 0.31 |
| gpt-6-sol | 40 | 0.589 [0.465, 0.711] | 0.307 [0.098, 0.513] | 0.091 [-0.030, 0.207] | — | — | 0 | 0 | 9.1 | 0.95 |

## test_selection — arm `agent`  (floor 0.001, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-sol | 40 | 0.325 [0.240, 0.416] | 0.324 [0.239, 0.415] | best | 0.650 | 0.425 | 0 | 0 | 8.1 | 1.68 |
| qwen3.5-9b | 120 | 0.273 [0.222, 0.325] | 0.273 [0.221, 0.325] | -0.084 [-0.167, 0.004] | 0.500 | 0.133 | 0 | 1 | 8.8 | 0.60 |

## test_selection — arm `single`  (floor 0.001, ceiling 1.000)

| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 120 | 0.106 [0.090, 0.123] | 0.105 [0.089, 0.122] | best | 0.417 | 0.117 | 0 | 4 | 1.0 | 0.07 |

# Ablations (paired, same instances)

| model | ablation | n | Δ mean [95% CI] | significant |
|---|---|---|---|---|
| qwen3.5-9b | differential_diagnosis: tools − no tools | 120 | 0.068 [0.038, 0.100] | yes |
| qwen3.5-9b | patient_diagnosis: tools − no tools | 120 | 0.087 [0.044, 0.131] | yes |
| qwen3.5-9b | test_selection: tools − no tools | 120 | 0.167 [0.121, 0.219] | yes |
| gpt-6-sol | atypical − typical diagnosis (`agent`) | 23 | 0.091 [-0.014, 0.210] | no |
| qwen3.5-9b | atypical − typical diagnosis (`agent`) | 77 | 0.014 [-0.044, 0.074] | no |
