"""Verify every item of the fork's roadmap checklist (audit/ROADMAP.md, stages 1–8) against executable evidence:
the named tests (run now, not trusted from a TODO checkbox), live checks on the release DB, the private overlay
and the repository, the git history for the dependency order, and the latest CI run. Writes
audit/ROADMAP_VERIFICATION.md.

    python scripts/verify_roadmap.py [--skip-tests]   # --skip-tests reuses the last JUnit files in results/verify/

Needs the compose stack up (the simulator suite runs inside the app container, with the private overlay).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "verify"
DB = ROOT / "benchmark_v1.3.db"
OVERLAY = ROOT / "private" / "labels_v1.3.db"

T = "test"   # evidence: tests that must all pass (and at least one must exist)
C = "check"  # evidence: a live check function

# ---------------------------------------------------------------------------
# live checks
# ---------------------------------------------------------------------------

def q(sql, db=DB):
    with sqlite3.connect(db) as c:
        return c.execute(sql).fetchall()


def chk_floors_reported():
    doc = json.loads((ROOT / "eval" / "floors.json").read_text())
    splits = sorted(doc["splits"])
    units = sorted(doc["splits"]["public"])
    lb = (ROOT / "results" / "leaderboard.md").read_text()
    ok = splits == ["heldout", "private", "public", "train"] and "normalized" in lb and "floor" in lb
    return ok, f"eval/floors.json: {len(units)} units x splits {splits}; leaderboard reports raw + normalized vs floor"


def chk_ci():
    r = subprocess.run(["gh", "run", "list", "--limit", "1", "--branch", "main", "--json", "status,conclusion,headSha,displayTitle"],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        run = json.loads(r.stdout)[0]
    except Exception:  # noqa: BLE001
        return None, f"could not read CI: {r.stderr.strip()[:200]}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ok = run["conclusion"] == "success"
    return ok, f"latest CI run on main: {run['status']}/{run['conclusion']} at {run['headSha'][:7]} (HEAD {head[:7]})"


def chk_score_reward_only():
    src = (ROOT / "epic_sim" / "app" / "routers" / "score.py").read_text()
    return ("redact_metrics" in src and "verbose_score_splits" in (ROOT / "epic_sim/app/config.py").read_text(),
            "private-split /score and /env responses carry the reward only (`redact_metrics`); public/heldout/train keep "
            "the metric breakdown for analysis (EPIC_SIM_VERBOSE_SCORE_SPLITS) — tests below")


def chk_index_instances():
    n = q("select count(*) from benchmark_ground_truth where task='patient_diagnosis' and granularity='encounter'")[0][0]
    enc = q("select count(*) from longitudinal_encounters")[0][0]
    return n > 0 and n <= 5602, f"{n:,} index-encounter diagnosis instances (target: up to 5,602; {enc:,} encounters in the release)"


def chk_cms():
    rows = dict(q("select icd10_status, count(*) from diagnoses group by 1"))
    bad = q("select count(*) from diagnoses where icd10_status='billable' and (icd10_code is null or icd10_code='')")[0][0]
    return bad == 0 and "billable" in rows, f"icd10_status per node {rows}; billable nodes without a code: {bad}"


def chk_semantic():
    n = q("select count(*) from diagnoses where icd10_flag is not null")[0][0]
    return n > 0, f"semantic flag stored on {n:,} nodes (`diagnoses.icd10_flag`)"


def chk_merge_provenance():
    merged = q("select count(*) from diagnoses where merged_into is not null")[0][0]
    table = q("select count(*) from diagnosis_merges")[0][0]
    cols = {r[1] for r in q("pragma table_info(diagnoses)")}
    prov = {"icd10_repaired_from", "icd10_repair_reason"} <= cols
    return merged > 0 and table > 0 and prov, f"{merged} nodes merged, {table} rows in `diagnosis_merges`, provenance columns {prov}"


def chk_distractors():
    total = q("select count(*) from question_diagnoses where role='distractor'")[0][0]
    rel = q("select count(*), sum(json_array_length(json_extract(ground_truth,'$.distractors'))) from benchmark_ground_truth "
            "where task='differential_diagnosis' and json_extract(ground_truth,'$.distractors') is not null")[0]
    ov = q("select count(*), sum(json_array_length(json_extract(ground_truth,'$.distractors'))) from benchmark_ground_truth "
           "where task='differential_diagnosis'", OVERLAY)[0] if OVERLAY.exists() else (0, 0)
    return total > 27000 and (rel[1] or 0) > 0, (f"{total:,} distractor rows in the graph; differential_diagnosis uses them in "
                                                  f"{rel[0] + ov[0]:,} instances ({(rel[1] or 0) + (ov[1] or 0):,} distractor slots)")


def chk_stage5_measured():
    b = ROOT / "audit" / "bench"
    files = sorted(p.name for p in b.glob("*.json")) if b.exists() else []
    return any("http" in f for f in files) and any("local" in f for f in files), f"benchmarks in audit/bench: {files}"


def chk_protocol_applied():
    readme = (ROOT / "README.md").read_text()
    return ("## Protocol results (Stage 8)" in readme and "non-protocol" in readme,
            "README 'Protocol results (Stage 8)' cites EVAL_PROTOCOL.md and results/leaderboard.md; paper Tables 2–3 / "
            "Appendix A labeled non-protocol; DATA_CARD reports no model scores")


def chk_ceilings():
    doc = json.loads((ROOT / "eval" / "floors.json").read_text())["splits"]["public"]
    ceil = {u: round(list(b["metrics"].values())[0]["ceiling"], 3) for u, b in doc.items()}
    return all(v >= 0.9 for v in ceil.values()), f"oracle ceilings (public): {ceil}"


def chk_paired():
    lb = (ROOT / "results" / "leaderboard.md").read_text()
    return "paired" in lb and "[" in lb, "every leaderboard cell: 95% bootstrap CI; Δ vs best = paired bootstrap on shared instances"


def chk_kimi():
    runs = sorted(p.name for p in (ROOT / "results").glob("kimi-k2.5__*__public__s0"))
    smoke = sorted(p.name for p in (ROOT / "results" / "smoke").glob("kimi-k2.5__*"))
    return (bool(runs) or None), (f"full Kimi runs: {len(runs)} units; smoke: {len(smoke)} units (clean). Deferred by decision "
                                  "until after RL; separation machinery tested")


def chk_ablations():
    lb = (ROOT / "results" / "leaderboard.md").read_text()
    ok = "tools − no tools" in lb and "atypical − typical" in lb
    return ok, "paired ablations on identical instances: tools − no tools (3 units) and atypical − typical (paired parents)"


def chk_random_samples():
    mf = [json.loads(p.read_text()) for p in (ROOT / "results").glob("*__public__s0/manifest.json")]
    return bool(mf) and all(m.get("seed") == 0 for m in mf), f"{len(mf)} runs, seeded random samples (seed 0, one instance per patient first)"


def chk_order():
    log = subprocess.run(["git", "log", "--reverse", "--format=%ad %s", "--date=iso"], capture_output=True, text=True, cwd=ROOT).stdout
    first = {}
    for line in log.splitlines():
        m = re.search(r"Stage (4b|\d)\b", line)
        if m and m.group(1) not in first:
            first[m.group(1)] = line[:19]
    order = ["1", "2", "3", "4", "4b", "5", "6", "7", "8"]
    seen = [s for s in order if s in first]
    times = [first[s] for s in seen]
    ok = seen == order and times == sorted(times)
    return ok, "first commit per stage: " + ", ".join(f"{s} {first[s][:16]}" for s in seen)


def chk_gate():
    ok, ev = chk_order()
    return ok, "Stage 7 work began only after Stages 1–4 were committed with their suites green (" + ev.split(": ", 1)[1] + ")"


def chk_a1_audit():
    out, ok = [], True
    for name, f in (("Qwen agent", "reward_noise_audit_qwen.json"), ("GPT-6 Sol agent", "reward_noise_audit_sol.json"),
                    ("Qwen no-tools (out of sample)", "reward_noise_audit_qwen_single.json")):
        p = ROOT / "results" / f
        if not p.exists():
            return False, f"missing results/{f}"
        d = json.loads(p.read_text())
        rates = {u: d[u]["correct_zero_rate"] for u in ("patient_diagnosis", "atypical_diagnosis", "differential_diagnosis", "test_selection")}
        ok &= all(r is None or r <= 0.05 for r in rates.values())
        out.append(name + " " + " / ".join("—" if r is None else f"{r:.1%}" for r in rates.values()))
    return ok, "judge-audited correct-but-0 (≤ 5% required; patient_dx / atypical / differential / test_selection): " + "; ".join(out)


def chk_a1_aliases():
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/build_diagnosis_aliases.py", "--check"], capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0, "eval/diagnosis_aliases.json reproducible from the release DB: " + r.stdout.strip()


def chk_a2_reported():
    lb = (ROOT / "results" / "leaderboard.md").read_text()
    preds = [json.loads(l) for p in (ROOT / "results").glob("*__patient_diagnosis__*__public__s0/predictions.jsonl")
             for l in p.read_text().splitlines() if l.strip()]
    ok = "| named | coded |" in lb and preds and all("diagnosis_named" in (p.get("metrics") or {}) for p in preds)
    return ok, f"leaderboard named/coded columns; {len(preds)} patient_diagnosis predictions carry diagnosis_named/coded"


def chk_a4_lock():
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "eval.reward_version", "--check"], capture_output=True, text=True, cwd=ROOT)
    info = json.loads(r.stdout)
    tag = subprocess.run(["git", "rev-list", "-n", "1", "reward-v1"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ok = r.returncode == 0 and info["reward_version"] == "reward-v1" and bool(tag)
    return ok, (f"lock {info['reward_version']} fingerprint {info['reward_locked_fingerprint']}, working tree "
                f"{info['reward_fingerprint']}, drift {info['reward_drift']}; git tag reward-v1 -> {tag[:7] or 'MISSING'}")


# ---------------------------------------------------------------------------
# the checklist
# ---------------------------------------------------------------------------

RH = "eval/tests/test_reward_hacking.py::"
LL = "eval/tests/test_label_leaks.py::"
DR = "eval/tests/test_data_repairs.py::"
LE = "eval/tests/test_local_env.py::"
S7 = "eval/tests/test_stage7_tasks.py::"
PR = "eval/tests/test_protocol.py::"
FH = "epic_sim/tests/test_fhir.py::"
ITEMS = [
    ("1", "Turn every exploit into a failing CI test (CI runs the suite on every push)", C, chk_ci),
    ("1", "Submitting only one HPI section", T, [RH + "test_single_passage_cannot_saturate_p5"]),
    ("1", "Ranking sections by type", T, [RH + "test_content_blind_ranking_captures_little_headroom", RH + "test_random_ranking_floor_is_recorded"]),
    ("1", "Pasting the chart", T, [RH + "test_chart_dump_is_not_a_good_summary"]),
    ("1", "Echoing finding names", T, [RH + "test_echoing_finding_names_is_not_a_good_summary", RH + "test_structured_prompts_carry_no_graph_concepts"]),
    ("1", "Using the abstention phrase", T, [RH + "test_stock_phrase_in_full_summary_is_not_abstention", RH + "test_stock_phrase_does_not_help_involved_items", RH + "test_empty_submission_is_not_an_abstention"]),
    ("1", "Copying the problem list", T, [RH + "test_copying_the_problem_list_does_not_solve_diagnosis", LE + "test_copying_the_problem_list_does_not_solve_diagnosis_through_the_env"]),
    ("1", "Echoing the problem-list tool", T, [RH + "test_problem_list_tool_reveals_nothing_beyond_the_profile"]),
    ("1", "Listing documented secondary conditions", T, [RH + "test_documented_comorbidities_are_not_penalized"]),
    ("1", "Inventing sentences", T, [RH + "test_invented_sentences_are_flagged"]),
    ("1", "Report every score relative to the defined floors", T, [RH + "test_committed_floors_are_current", RH + "test_normalized_score_is_zero_at_floor_and_one_at_ceiling", PR + "test_normalization_uses_floors_file"]),
    ("1", "(floors file + normalized reporting)", C, chk_floors_reported),
    ("2", "Fix get_problem_list at the source", T, [LL + "test_problem_list_service_returns_documented_history_without_codes", LE + "test_problem_list_is_documented_history_without_codes", "epic_sim/tests/test_epic.py::test_problem_list"]),
    ("2", "Apply the time cutoff to every consumer", T, [LL + "test_allowed_encounters_for_imaging_instance", LL + "test_filter_future_encounters_covers_tool_and_fhir_shapes", FH + "test_observation_honors_session_cutoff", "epic_sim/tests/test_env.py::test_imaging_episode_hides_future_encounters"]),
    ("2", "Hide assessment/plan from every consumer", T, [LL + "test_encounter_detail_and_section_hide_outcome_sections", LL + "test_strip_outcome_sections_shapes", "epic_sim/tests/test_env.py::test_step_hides_outcome_sections_and_counts_budget"]),
    ("2", "Retrieval query cannot act as the diagnosis answer key", T, [LL + "test_retrieval_instances_are_per_diagnosis"]),
    ("2", "Remove label-bearing prompt hints", T, [RH + "test_structured_prompts_carry_no_graph_concepts"]),
    ("2", "Few-shot examples from train only", T, [LL + "test_frozen_few_shot_examples_are_train_only"]),
    ("2", "Private held-out split", T, [LL + "test_private_split_has_no_labels_in_release", LL + "test_private_overlay_restores_labels_when_present", LL + "test_strip_labels_keeps_inputs"]),
    ("2", "/score returns only the reward", T, [LL + "test_score_response_is_redacted_for_private", "epic_sim/tests/test_score.py::test_private_split_scoring_is_rate_limited", "epic_sim/tests/test_env.py::test_agent_token_can_act_but_not_see_reward"]),
    ("2", "(scope of the redaction)", C, chk_score_reward_only),
    ("3", "Score missing answers as 0 on every task", T, [RH + "test_empty_submission_scores_zero", RH + "test_empty_diagnosis_prediction_is_counted", LE + "test_empty_and_malformed_submissions_score_zero", S7 + "test_oracle_scores_one_and_empty_scores_zero"]),
    ("3", "Diagnosis: do not penalize conditions documented in the notes", T, [RH + "test_documented_comorbidities_are_not_penalized", "epic_sim/tests/test_score.py::test_patient_diagnosis_chart_neutral_not_penalized"]),
    ("3", "Diagnosis: graded credit for ICD codes", T, [RH + "test_icd_credit_is_graded", RH + "test_bare_categories_do_not_earn_full_credit"]),
    ("3", "Diagnosis: score acuity", T, [RH + "test_acuity_is_scored"]),
    ("3", "Retrieval: fixed-k denominator", T, [RH + "test_single_passage_cannot_saturate_p5"]),
    ("3", "Retrieval: grade sections by content, not labels", T, [RH + "test_retrieval_grades_depend_on_content", LL + "test_judgments_follow_the_content_grading_rule"]),
    ("3", "Summarization: precision term", T, [RH + "test_summary_oracle_is_fully_grounded", RH + "test_long_summary_is_discounted", "eval/tests/test_current_visit.py::test_off_target_penalizes_verbosity"]),
    ("3", "Summarization: per-encounter findings quota", T, [RH + "test_must_include_covers_every_encounter_when_possible"]),
    ("3", "Summarization: concept-level matching", T, ["eval/tests/test_imaging_concepts.py::test_extractor_matches_synonyms_and_abbreviations", "eval/tests/test_current_visit.py::test_similarity_synonym_vs_unrelated"]),
    ("3", "Summarization: negation handled", T, [RH + "test_negated_finding_is_not_credited", "eval/tests/test_specialty_conditioned.py::test_negation_guard"]),
    ("3", "Specialty: explicit abstain field", T, [RH + "test_abstention_requires_the_field", "eval/tests/test_specialty_conditioned.py::test_sc_abstention"]),
    ("3", "Specialty: randomize absent specialties", T, [RH + "test_specialty_name_does_not_predict_absence"]),
    ("3", "Specialty: fix the 540 always-zero items", T, [RH + "test_specialty_oracle_scores_one_on_every_item"]),
    ("3", "Imaging: deterministic reference", T, ["epic_sim/tests/test_score.py::test_imaging_reference_terms_score_one", RH + "test_imaging_floor_is_recorded"]),
    ("4", "Medical history reflects the visit date", T, [DR + "test_index_chart_has_no_dated_line_for_its_label", DR + "test_hpi_never_names_its_own_new_diagnosis"]),
    ("4", "Surgical history reflects the visit date", T, [DR + "test_procedures_follow_their_diagnosis"]),
    ("4", "Strip tested diagnoses from the 500 profiles", T, [DR + "test_profiles_do_not_list_tested_diagnoses", DR + "test_profile_problem_lines_do_not_name_keyed_diagnoses", DR + "test_primary_diagnoses_column_is_not_a_label_list"]),
    ("4", "Diagnosis = one visit, chart up to that visit", T, [DR + "test_index_encounter_instances_cover_first_occurrences", "eval/tests/test_current_visit.py::test_interior_index_holds_out_future", DR + "test_neutral_extra_is_earlier_keyed_codes"]),
    ("4", "Remove copyability", T, [DR + "test_copy_policy_is_near_zero_on_index_instances", RH + "test_copying_the_problem_list_does_not_solve_diagnosis"]),
    ("4", "Target up to 5,602 valid instances", C, chk_index_instances),
    ("4", "ICD: validate against CMS", T, [DR + "test_icd_codes_are_billable_with_provenance"]),
    ("4", "(CMS status per node)", C, chk_cms),
    ("4", "ICD: semantic validation check", C, chk_semantic),
    ("4", "ICD: merge duplicates + store provenance", T, [DR + "test_merged_nodes_have_no_edges_and_labels_use_node_codes", DR + "test_release_info_records_every_repair"]),
    ("4", "(merge table + provenance columns)", C, chk_merge_provenance),
    ("4b", "Presence flags", T, [FH + "test_absent_findings_are_negative_not_positive", "epic_sim/tests/test_units.py::test_no_text_no_number_gives_nothing_so_the_resource_carries_a_presence_concept"]),
    ("4b", "Dates", T, [FH + "test_observation_honors_session_cutoff", FH + "test_observation_search_params"]),
    ("4b", "Units", T, ["epic_sim/tests/test_units.py::test_quantity_with_ucum_unit_and_reference_range", FH + "test_observation_units_components_and_reference_range"]),
    ("4b", "One resource per measurement", T, [FH + "test_observation_is_one_resource_per_measurement"]),
    ("4b", "Write support (optional)", T, [FH + "test_write_round_trip_is_session_scoped", FH + "test_write_validation_and_scopes", FH + "test_capability_statement_declares_create_and_search_params"]),
    ("5", "Measure /env step latency; identify the bottleneck", C, chk_stage5_measured),
    ("5", "In-process environment over SQLite", T, [LE + "test_observations_and_rewards_match_the_http_env", LE + "test_point_in_time_cutoff_and_hidden_sections", LE + "test_budget_forces_submission_and_errors_are_observations"]),
    ("5", "In-process env passes the Stage-1 reward-hacking tests", T, [LE + "test_stage1_policies_score_identically_through_the_env", LE + "test_oracle_reaches_the_ceiling", LE + "test_copying_the_problem_list_does_not_solve_diagnosis_through_the_env"]),
    ("6", "Wire all stages into main.py", T, ["etl/tests/test_pipeline_wiring.py::test_every_stage_1_to_12_resolves_to_a_callable", "etl/tests/test_pipeline_wiring.py::test_selected_stages_from_flags"]),
    ("6", "Fail loudly when required files are missing", T, ["etl/tests/test_pipeline_wiring.py::test_preflight_lists_missing_inputs_with_fixes", "etl/tests/test_pipeline_wiring.py::test_main_exits_2_with_the_missing_list", "etl/tests/test_pipeline_wiring.py::test_stage_11_raises_without_curated_files"]),
    ("6", "Fix retry behavior", T, ["etl/tests/test_llm_client.py::test_retries_5xx_then_succeeds", "etl/tests/test_llm_client.py::test_client_error_is_not_retried", "etl/tests/test_llm_client.py::test_validation_failure_is_retried_inside_the_loop", "etl/tests/test_llm_client.py::test_attempts_exhausted_raises"]),
    ("6", "Fix the cache key", T, ["etl/tests/test_llm_client.py::test_cache_key_covers_the_whole_prompt_and_settings"]),
    ("6", "Open LLM endpoint", T, ["etl/tests/test_llm_client.py::test_openai_request_carries_seed_temperature_and_parses_reply", "etl/tests/test_llm_client.py::test_settings_from_env_legacy_gateway"]),
    ("6", "Seeded LLM generation", T, ["etl/tests/test_llm_client.py::test_openai_request_carries_seed_temperature_and_parses_reply", "etl/tests/test_llm_client.py::test_log_call_records_sampling_settings_and_cache_lookup_round_trips"]),
    ("6", "Bring your own source corpus (contamination, reproducibility)", T, ["etl/tests/test_pipeline_wiring.py::test_source_fingerprints_roundtrip", "etl/tests/test_pipeline_smoke.py::test_stage_5_and_7_run_end_to_end_on_the_mock", "etl/tests/test_deck_registry.py::test_resolve_by_glob"]),
    ("7", "Differential diagnosis", T, [S7 + "test_differential_label_is_correct_plus_distractors", S7 + "test_differential_rewards_rank_and_penalizes_the_problem_list"]),
    ("7", "Test selection (the agentic task)", T, [S7 + "test_test_selection_labels_are_documented_discriminating_tests", S7 + "test_test_selection_episode_hides_results_and_orders_reveal_documented_findings", S7 + "test_workup_score_components", "eval/tests/test_stage8_audit.py::test_order_by_test_name_reveals_the_result_named_finding"]),
    ("7", "Use the 27k unused distractor rows", C, chk_distractors),
    ("7", "Injected errors with labels", T, [S7 + "test_error_injection_is_a_real_change_of_the_stated_type", S7 + "test_error_detection_half_credit_and_aliases"]),
    ("7", "Lab-triage tasks", T, [S7 + "test_lab_triage_labels_partition_the_results", S7 + "test_triage_all_results_is_penalized"]),
    ("7", "Atypical variants", T, [S7 + "test_atypical_masks_mention_of_a_strong_finding_and_keeps_the_label", S7 + "test_error_and_atypical_episodes_serve_the_altered_text"]),
    ("7", "(every new family: oracle 1.0 in both environments)", T, [S7 + "test_oracle_episodes_reach_the_ceiling_in_the_env", "eval/tests/test_stage8_audit.py::test_stage7_and_specialty_episodes_match_the_http_env"]),
    ("8", "Apply to anything the fork reports", C, chk_protocol_applied),
    ("8", "Define performance floors", T, [RH + "test_committed_floors_are_current"]),
    ("8", "Define performance ceilings", C, chk_ceilings),
    ("8", "Report paired intervals", T, [PR + "test_bootstrap_and_paired_difference_are_deterministic"]),
    ("8", "(paired intervals in the committed leaderboard)", C, chk_paired),
    ("8", "Evaluate Kimi independently from the ranked panel", T, [PR + "test_leaderboard_ranks_normalized_and_separates_the_generator"]),
    ("8", "(Kimi row run)", C, chk_kimi),
    ("8", "Run clean ablations", C, chk_ablations),
    ("8", "Random agentic patient samples", T, ["eval/tests/test_stage8_audit.py::test_atypical_sample_is_paired_with_the_typical_sample", PR + "test_runner_scores_oracle_agent_one_and_records_cost"]),
    ("8", "(runs use seeded random samples)", C, chk_random_samples),
    ("A1", "Concept-level diagnosis matching: correct-but-0 ≤ 5%", C, chk_a1_audit),
    ("A1", "(aliases deterministic; concept credit and probes tested)", T, ["eval/tests/test_stage8_audit.py::test_concept_aliases_credit_clinically_identical_names", "eval/tests/test_stage8_audit.py::test_name_credit", "eval/tests/test_stage8_audit.py::test_kitchen_sink_and_hedged_names_score_the_floor"]),
    ("A1", "(alias file reproducible)", C, chk_a1_aliases),
    ("A2", "Named vs coded reported", T, ["eval/tests/test_stage8_audit.py::test_diagnosis_named_and_coded_split_the_reward"]),
    ("A2", "(in the leaderboard and every stored prediction)", C, chk_a2_reported),
    ("A3", "RL-pressure probe suite at the floor (public + heldout)", T, ["eval/tests/test_stage8_audit.py::test_rl_pressure_probes_stay_at_the_floor", "eval/tests/test_stage8_audit.py::test_kitchen_sink_and_hedged_names_score_the_floor"]),
    ("A4", "Reward frozen (lock, manifest, refusal, CI)", T, ["eval/tests/test_stage8_audit.py::test_reward_files_match_the_lock", "eval/tests/test_stage8_audit.py::test_runner_refuses_a_drifted_reward_and_records_the_version"]),
    ("A4", "(lock intact, git tag reward-v1)", C, chk_a4_lock),
    ("order", "Dependency order 1 → 2 → 3 → 4 → 4b → 5 → 6 → 7 → 8", C, chk_order),
    ("gate", "Stage 7 only after Stages 1–4 made rewards exploit-resistant and data leak-free", C, chk_gate),
]


def run_tests() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "eval/tests", "etl/tests", f"--junitxml={OUT}/local.xml"],
                   cwd=ROOT, check=False)
    subprocess.run(["docker", "compose", "exec", "-T", "app", "pytest", "epic_sim/tests", "-q", "-p", "no:cacheprovider",
                    "--junitxml=/tmp/sim.xml"], cwd=ROOT, check=False)
    subprocess.run(f"docker compose cp app:/tmp/sim.xml {OUT}/sim.xml", shell=True, cwd=ROOT, check=False)


def load_results() -> dict[str, str]:
    res: dict[str, str] = {}
    for f in ("local.xml", "sim.xml"):
        p = OUT / f
        if not p.exists():
            continue
        for tc in ET.parse(p).getroot().iter("testcase"):
            cls = tc.get("classname", "")
            path = cls.replace(".", "/") + ".py" if "." in cls else f"epic_sim/tests/{cls}.py"   # container rootdir
            name = re.sub(r"\[.*\]$", "", tc.get("name", ""))
            key = f"{path}::{name}"
            status = "passed"
            for child in tc:
                if child.tag in ("failure", "error"):
                    status = "failed"
                elif child.tag == "skipped":
                    status = "skipped"
            prev = res.get(key)
            if prev != "failed":                       # any failing parametrization fails the test
                res[key] = status if prev in (None, "passed") or status == "failed" else prev
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    if not a.skip_tests:
        run_tests()
    res = load_results()
    lines = ["# Roadmap verification (stages 1–8, RL readiness A1–A4)", "",
             "Generated by `scripts/verify_roadmap.py`: every item is checked against tests run for this report or a live "
             "check on the release DB / overlay / repository / git history / CI — not against the TODO checkboxes.", "",
             "| stage | item | status | evidence |", "|---|---|---|---|"]
    counts = {"PASS": 0, "FAIL": 0, "DEFERRED": 0}
    for stage, item, kind, spec in ITEMS:
        if kind == T:
            st = [(t, res.get(t, "missing")) for t in spec]
            ok = all(s == "passed" for _, s in st)
            ev = "; ".join(f"`{t.split('::')[1]}` {s}" for t, s in st)
            status = "PASS" if ok else "FAIL"
        else:
            ok, ev = spec()
            status = "PASS" if ok else ("DEFERRED" if ok is None else "FAIL")
        counts[status] += 1
        lines.append(f"| {stage} | {item} | **{status}** | {ev} |")
    lines += ["", f"**{counts['PASS']} pass, {counts['FAIL']} fail, {counts['DEFERRED']} deferred** "
                  f"({sum(1 for s in res.values() if s == 'passed')} tests passed, "
                  f"{sum(1 for s in res.values() if s == 'failed')} failed, {sum(1 for s in res.values() if s == 'skipped')} skipped in this run)."]
    (ROOT / "audit" / "ROADMAP_VERIFICATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
