"""Agent system prompts for Phase F — 5 tasks × 2 arms.

Each task has a shared preamble + task-specific goal + arm-specific interface description.
"""

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------

PREAMBLE = """You are a clinical AI agent tasked with {task_description}.

You will interact with a patient's medical record through {interface_description}.
After gathering sufficient information, submit your answer using the submission
format specified below.

IMPORTANT CONSTRAINTS:
- You have a budget of {budget} {action_type}. Each query or tool call counts as 1.
- Plan your information gathering strategy before acting.
- Start with high-level overview, then drill into specifics as needed.
- Submit your answer when you have sufficient evidence — do not exhaust your
  budget on marginal queries.

{task_specific_goal}

SUBMISSION SCHEMA:
{output_json_schema}"""


# ---------------------------------------------------------------------------
# Interface descriptions
# ---------------------------------------------------------------------------

STRUCTURED_INTERFACE = """You have access to clinical tools for reviewing \
patient charts, viewing results, searching records, and submitting your \
clinical assessment. Tools are provided via function calling — call them \
directly when you need information.

Start by opening the patient's chart for an overview, then drill into \
specific encounters and results as needed. When you have gathered enough \
evidence, use the appropriate submit tool to record your answer.

Do NOT output JSON tool calls in your text. Simply call the tools directly \
using the function calling interface."""


BASH_INTERFACE = """You have access to a PostgreSQL database via the psql command and basic Unix
utilities (grep, jq, cat, head, wc). Write SQL queries to explore the
patient's medical record.

{orientation}"""


# ---------------------------------------------------------------------------
# Bash orientation prompt (schema + examples + tips)
# ---------------------------------------------------------------------------

BASH_ORIENTATION = """DATABASE SCHEMA (9 accessible tables):

─── longitudinal_patients ───
  patient_id          INTEGER PRIMARY KEY
  profile             JSONB
      → chronic_conditions: [{{icd10, name, onset_date}}]
      → home_medications: [{{name, dose, frequency, route}}]
      → allergies: [{{allergen, reaction, severity}}]
      → family_history: [{{relation, condition}}]
      → social_history: {{smoking_status, alcohol_use, occupation, ...}}
  age                 INTEGER
  sex                 TEXT          -- 'M' | 'F'
  num_encounters      INTEGER
  created_at          TIMESTAMPTZ

─── longitudinal_encounters ───
  encounter_id        INTEGER PRIMARY KEY
  patient_id          INTEGER       → FK longitudinal_patients
  encounter_date      DATE
  encounter_type      TEXT          -- outpatient | ed | inpatient | icu | telehealth
                                    --   | procedure | follow_up
  chief_complaint     TEXT
  attending_name      TEXT
  department          TEXT
  note_text           TEXT          -- full encounter note (concatenated sections)
  source_question_ids JSONB         -- internal reference, not clinically useful
  created_at          TIMESTAMPTZ

─── encounter_ehr_sections ───
  id                  SERIAL PRIMARY KEY
  encounter_id        INTEGER       → FK longitudinal_encounters
  section_type        TEXT          -- demographics | chief_complaint | hpi | pmh | psh
                                    --   | medications | allergies | family_history
                                    --   | social_history | ros | vitals | physical_exam
                                    --   | labs | imaging | pathology | other_studies
                                    -- (assessment/plan sections are not available)
  section_text        TEXT
  section_order       INTEGER       -- display order within encounter

─── diagnoses ───
  diagnosis_id        INTEGER PRIMARY KEY
  icd10_code          TEXT          -- e.g., 'I21.01'
  icd10_desc          TEXT
  snomed_id           BIGINT
  display_name        TEXT
  category            TEXT          -- e.g., 'cardiovascular', 'pulmonary', 'infectious'
  acuity              TEXT          -- acute | chronic | acute_on_chronic | subacute | recurrent

─── clinical_findings ───
  finding_id          INTEGER PRIMARY KEY
  snomed_id           BIGINT
  display_name        TEXT
  finding_type        TEXT          -- symptom | sign | lab_result | imaging_finding | history

─── diagnosis_findings ───
  id                  SERIAL PRIMARY KEY
  diagnosis_id        INTEGER       → FK diagnoses
  finding_id          INTEGER       → FK clinical_findings
  association_type    TEXT          -- e.g., 'pathognomonic', 'supportive', 'common', 'risk_factor'
  specificity         FLOAT         -- 0.0–1.0

─── fact_cards ───
  fact_id             INTEGER PRIMARY KEY
  fact_text           TEXT
  subject             TEXT
  organ_system        TEXT
  topic               TEXT
  created_at          TIMESTAMPTZ

─── imaging_orders ───
  order_id            INTEGER PRIMARY KEY
  encounter_id        INTEGER       → FK longitudinal_encounters
  modality            TEXT          -- CT | MRI | XR | US | NM | PET | Fluoro
  body_region         TEXT
  clinical_indication TEXT
  order_date          DATE
  created_at          TIMESTAMPTZ

─── terminology_codes ───
  code_id             SERIAL PRIMARY KEY
  code_system         TEXT          -- 'ICD10' | 'SNOMED' | 'LOINC' | 'CPT'
  code_value          TEXT
  display_name        TEXT
  parent_code         TEXT
  category            TEXT


EXAMPLE QUERIES:

1. Get patient profile:
   psql -At -c "SELECT profile FROM longitudinal_patients WHERE patient_id = {{pid}}" | jq '.'

2. List encounters chronologically:
   psql -c "SELECT encounter_id, encounter_date, encounter_type, chief_complaint, department
             FROM longitudinal_encounters
             WHERE patient_id = {{pid}}
             ORDER BY encounter_date"

3. Get EHR sections for an encounter:
   psql -c "SELECT section_type, section_text
             FROM encounter_ehr_sections
             WHERE encounter_id = {{eid}}
             ORDER BY section_order"

TIPS:
- Use -At flags for clean output without headers/borders
- Use LIMIT to avoid huge result sets (output is truncated at 8,000 characters)
- Pipe to head/tail/grep to filter long output
- Use jq to navigate JSONB columns (profile, source_question_ids)
- For text search: WHERE section_text ILIKE '%term%'
  or: to_tsvector('english', section_text) @@ plainto_tsquery('english', 'term')
- fact_cards has 53,999 rows — always filter by subject, organ_system, or topic

SUBMISSION FORMAT:
When ready to submit your answer, use curl:
  curl -s -X POST {api_base}/agent/tools \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer {{token}}" \\
    -d '{{submission_json}}'"""


# ---------------------------------------------------------------------------
# Task-specific goals + output schemas
# ---------------------------------------------------------------------------

TASK_GOALS = {
    "patient_diagnosis": {
        "description": "identifying all active diagnoses and chronic conditions for a patient",
        "goal": (
            "Review this patient's complete longitudinal medical record across all encounters. "
            "Identify all active diagnoses and chronic conditions with ICD-10-CM codes and "
            "acuity classifications (acute, chronic, acute_on_chronic)."
        ),
        "schema": """{
  "active_diagnoses": [
    {"icd10": "I21.01", "name": "STEMI involving LAD", "acuity": "acute"},
    ...
  ],
  "chronic_conditions": [
    {"icd10": "I10", "name": "Essential hypertension", "acuity": "chronic"},
    ...
  ]
}""",
    },
    "context_summarization": {
        "description": "producing a comprehensive clinical summary for a patient",
        "goal": (
            "Review this patient's medical record and produce a comprehensive clinical summary "
            "(5–10 sentences) addressing the clinical question: {clinical_question}"
        ),
        "schema": """{
  "summary": "A 65-year-old male with history of ... presented to the ED with ..."
}""",
    },
    "evidence_retrieval": {
        "description": "retrieving and ranking evidence passages for given diagnoses",
        "goal": (
            "For the given diagnoses ({diagnosis_names}), review the patient's chart sections and assign "
            "relevance grades (0–3), ranking the most relevant first. A passage is one EHR section; its "
            "passage_id is \"ees_<section_id>\" (the section_id the chart tools return).\n"
            "Grade 0: Not relevant. Grade 1: Marginally relevant. "
            "Grade 2: Clearly relevant. Grade 3: Highly specific / defining."
        ),
        "schema": """{
  "rankings": [
    {"passage_id": "ees_1234", "grade": 3},
    {"passage_id": "ees_5678", "grade": 2},
    ...
  ]
}""",
    },
    "imaging_indication": {
        "description": "generating a radiology pre-read assessment for an imaging order",
        "goal": (
            "Given this imaging order ({modality} of {body_region}, indication: '{clinical_indication}'), "
            "review the patient's record to infer the underlying clinical question, produce a "
            "pre-read summary, and generate a differential diagnosis. "
            "You may review encounters up to and including encounter_id = {encounter_id} "
            "(do not access future encounters)."
        ),
        "schema": """{
  "clinical_question": "Is there evidence of pulmonary embolism?",
  "pre_read_summary": "65M with acute-onset dyspnea and pleuritic chest pain...",
  "must_include_findings": ["filling defect in pulmonary artery", ...],
  "differential": [
    {"diagnosis": "Pulmonary embolism", "icd10": "I26.99"},
    ...
  ]
}""",
    },
    # ---- Stage 7 families (all bound to an index encounter; the chart after it is not observable) ----
    "differential_diagnosis": {
        "description": "ranking the differential diagnosis for a patient's index visit",
        "goal": (
            "Review the chart up to and including encounter_id = {encounter_id} (the index visit). Rank the "
            "diagnoses this presentation should make a clinician consider, most likely first: the leading "
            "diagnosis and the alternatives that must be distinguished from it. At most 5 entries, each with an "
            "ICD-10-CM code and a name. Conditions already documented before the index visit are not the target."
        ),
        "schema": """{
  "differential": [
    {"icd10": "K35.80", "name": "Acute appendicitis"},
    {"icd10": "N10", "name": "Acute pyelonephritis"},
    ...
  ]
}""",
    },
    "test_selection": {
        "description": "choosing the tests that establish the diagnosis at a patient's index visit",
        "goal": (
            "The results of the index visit (encounter_id = {encounter_id}) are hidden: its laboratory, imaging, "
            "pathology and other study sections are not shown. Review the history and examination, then use the "
            "order_test tool to obtain the results you need, by test or panel name (e.g. 'CBC', 'lipase', "
            "'CT abdomen'); each order costs one action and returns the result as documented at that visit, or "
            "'not performed' if it was not. Order what discriminates the diagnosis and no more, then submit the "
            "diagnosis you established. Reward = diagnosis credit x evidence (a discriminating test ordered) x "
            "parsimony (orders beyond those needed reduce it)."
        ),
        "schema": """{"icd10": "K85.90", "name": "Acute pancreatitis"}""",
    },
    "error_detection": {
        "description": "finding the documentation error in a patient's index visit note",
        "goal": (
            "Exactly one section of the index visit (encounter_id = {encounter_id}) contains an injected "
            "documentation error: an implausible laboratory value, a left/right (laterality) swap, an age that "
            "contradicts the demographics, or a sex/pronoun contradiction. Read the visit's sections and the "
            "earlier chart, identify the section and the error type, and describe the error."
        ),
        "schema": """{
  "section_type": "labs",
  "error_type": "implausible_value",
  "description": "White blood cell count listed as 90/uL; incompatible with the rest of the CBC and the presentation."
}""",
    },
    "lab_triage": {
        "description": "triaging the results of a patient's index visit",
        "goal": (
            "Review the index visit (encounter_id = {encounter_id}). From its laboratory and vital-sign results, "
            "list the findings that bear on the diagnosis of this presentation (key or supporting), leaving out "
            "the incidental ones, and name the single most urgent finding. Use the finding names as documented."
        ),
        "schema": """{
  "relevant": ["Serum lipase", "Serum calcium", "Heart rate"],
  "most_urgent": "Serum lipase"
}""",
    },
    "atypical_diagnosis": {
        "description": "identifying the diagnosis established at a patient's index visit from an atypical presentation",
        "goal": (
            "Review the chart up to and including encounter_id = {encounter_id}. Report the diagnosis established "
            "at that visit (ICD-10-CM code, name, acuity) under active_diagnoses or chronic_conditions. Some of the "
            "classic findings are not documented at this visit; reason from what is present. Conditions documented "
            "before the index visit are not the target."
        ),
        "schema": """{
  "active_diagnoses": [{"icd10": "I21.01", "name": "STEMI involving LAD", "acuity": "acute"}],
  "chronic_conditions": []
}""",
    },
}


# ---------------------------------------------------------------------------
# Task-specific submit tool schemas (override Epic API defaults for Phase F)
# ---------------------------------------------------------------------------

SUBMIT_TOOL_SCHEMAS = {
    "patient_diagnosis": {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": (
                "Submit your complete diagnosis list for this patient. Include ALL active "
                "diagnoses and chronic conditions you identified, each with an ICD-10-CM code, "
                "a display name, and acuity classification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "active_diagnoses": {
                        "type": "array",
                        "description": "Active diagnoses (acute or acute_on_chronic)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "icd10": {"type": "string", "description": "ICD-10-CM code (e.g., I21.01)"},
                                "name": {"type": "string", "description": "Diagnosis display name"},
                                "acuity": {"type": "string", "enum": ["acute", "acute_on_chronic"]},
                            },
                            "required": ["icd10", "name", "acuity"],
                        },
                    },
                    "chronic_conditions": {
                        "type": "array",
                        "description": "Chronic conditions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "icd10": {"type": "string", "description": "ICD-10-CM code"},
                                "name": {"type": "string", "description": "Condition display name"},
                                "acuity": {"type": "string", "enum": ["chronic"]},
                            },
                            "required": ["icd10", "name", "acuity"],
                        },
                    },
                },
                "required": ["active_diagnoses", "chronic_conditions"],
            },
        },
    },
    "context_summarization": {
        "type": "function",
        "function": {
            "name": "submit_summary",
            "description": (
                "Submit your comprehensive clinical summary (5-10 sentences) "
                "addressing the clinical question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Clinical summary text (5-10 sentences)",
                    },
                    "abstain": {
                        "type": "boolean",
                        "description": "Specialty-conditioned items only: true when the patient has no active "
                                       "problem in the requested specialty (then leave summary empty). "
                                       "Abstention is scored from this field, never from the text.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    "evidence_retrieval": {
        "type": "function",
        "function": {
            "name": "submit_rankings",
            "description": (
                "Submit relevance grades (0-3) for clinical passages. "
                "Grade 0=not relevant, 1=marginal, 2=clearly relevant, 3=highly specific."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rankings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "passage_id": {"type": "string"},
                                "grade": {"type": "integer", "minimum": 0, "maximum": 3},
                            },
                            "required": ["passage_id", "grade"],
                        },
                    },
                },
                "required": ["rankings"],
            },
        },
    },
    "imaging_indication": {
        "type": "function",
        "function": {
            "name": "submit_pre_read",
            "description": (
                "Submit your radiology pre-read assessment including clinical question, "
                "summary, must-include findings, and differential diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clinical_question": {
                        "type": "string",
                        "description": "The inferred clinical question driving this imaging order",
                    },
                    "pre_read_summary": {
                        "type": "string",
                        "description": "Brief clinical context for the radiologist",
                    },
                    "must_include_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Findings the radiologist must comment on",
                    },
                    "differential": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "diagnosis": {"type": "string"},
                                "icd10": {"type": "string"},
                            },
                            "required": ["diagnosis", "icd10"],
                        },
                    },
                },
                "required": ["clinical_question", "pre_read_summary", "must_include_findings", "differential"],
            },
        },
    },
    # ---- Stage 7 families ----
    "differential_diagnosis": {
        "type": "function",
        "function": {
            "name": "submit_differential",
            "description": "Submit the ranked differential for the index visit: at most 5 diagnoses, most likely first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "differential": {
                        "type": "array", "maxItems": 5,
                        "items": {"type": "object",
                                  "properties": {"icd10": {"type": "string", "description": "ICD-10-CM code"},
                                                 "name": {"type": "string"}},
                                  "required": ["icd10", "name"]},
                    },
                },
                "required": ["differential"],
            },
        },
    },
    "test_selection": {
        "type": "function",
        "function": {
            "name": "submit_workup",
            "description": "Submit the diagnosis you established from the tests you ordered. Ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {"icd10": {"type": "string", "description": "ICD-10-CM code"}, "name": {"type": "string"}},
                "required": ["icd10", "name"],
            },
        },
    },
    "error_detection": {
        "type": "function",
        "function": {
            "name": "submit_error",
            "description": "Report the injected documentation error: which section, which type, and what is wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_type": {"type": "string", "description": "the section that carries the error (e.g. labs, hpi, physical_exam, imaging, vitals)"},
                    "error_type": {"type": "string", "enum": ["implausible_value", "laterality", "age_contradiction", "sex_contradiction"]},
                    "description": {"type": "string"},
                },
                "required": ["section_type", "error_type", "description"],
            },
        },
    },
    "lab_triage": {
        "type": "function",
        "function": {
            "name": "submit_triage",
            "description": "Submit the results that bear on this visit's diagnosis and the single most urgent one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relevant": {"type": "array", "items": {"type": "string"}, "description": "finding names as documented"},
                    "most_urgent": {"type": "string"},
                },
                "required": ["relevant", "most_urgent"],
            },
        },
    },
}
SUBMIT_TOOL_SCHEMAS["atypical_diagnosis"] = SUBMIT_TOOL_SCHEMAS["patient_diagnosis"]   # same answer, same scorer


# Bash submission examples per task
BASH_SUBMISSION_EXAMPLES = {
    "patient_diagnosis": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_diagnosis", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"active_diagnoses": [{{"icd10": "...", "name": "...", "acuity": "acute"}}], "chronic_conditions": [{{"icd10": "...", "name": "...", "acuity": "chronic"}}]}}}}}}'""",

    "context_summarization": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_summary", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"summary": "..."}}}}}}'""",

    "evidence_retrieval": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_rankings", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"rankings": [{{"passage_id": "...", "grade": 3}}]}}}}}}'""",

    "imaging_indication": """curl -s -X POST {api_base}/agent/tools \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {token}" \\
  -d '{{"tool": "submit_pre_read", "session_id": "{session_id}", "arguments": {{"gt_id": {gt_id}, "payload": {{"clinical_question": "...", "pre_read_summary": "...", "must_include_findings": [...], "differential": [...]}}}}}}'""",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(
    task: str,
    arm: str,
    budget: int = 40,
    api_base: str = "http://api:8000",
    token: str = "",
    session_id: str = "",
    **task_kwargs,
) -> str:
    """Build the full system prompt for an agent session.

    Args:
        task: One of the 4 agent eval tasks (no diagnosis_accuracy).
        arm: 'structured' or 'bash'.
        budget: Action budget (default 40).
        api_base: API base URL for bash submissions.
        token: JWT token for bash submissions.
        session_id: Session ID for bash submissions.
        **task_kwargs: Task-specific placeholders (encounter_id, clinical_question, etc.)
    """
    task_info = TASK_GOALS[task]

    if arm in ("structured", "context"):
        interface_desc = STRUCTURED_INTERFACE
        action_type = "tool calls"
    else:
        orientation = BASH_ORIENTATION.format(
            api_base=api_base, pid="{pid}", eid="{eid}",
        )
        interface_desc = BASH_INTERFACE.format(orientation=orientation)
        action_type = "commands"

    goal = task_info["goal"].format(**task_kwargs)
    schema = task_info["schema"]

    prompt = PREAMBLE.format(
        task_description=task_info["description"],
        interface_description=interface_desc,
        budget=budget,
        action_type=action_type,
        task_specific_goal=goal,
        output_json_schema=schema,
    )

    # For bash arm, append the submission example
    if arm == "bash" and task in BASH_SUBMISSION_EXAMPLES:
        example = BASH_SUBMISSION_EXAMPLES[task].format(
            api_base=api_base,
            token=token,
            session_id=session_id,
            gt_id=task_kwargs.get("gt_id", "{gt_id}"),
        )
        prompt += f"\n\nSUBMISSION EXAMPLE:\n{example}"

    return prompt


def build_patient_intro(
    patient_id: int,
    task: str,
    encounter_id: int | None = None,
    diagnosis_names: str | None = None,
    modality: str | None = None,
    body_region: str | None = None,
    clinical_indication: str | None = None,
    **_extra,
) -> str:
    """Build the initial user message introducing the patient assignment."""
    parts = [f"Your assigned patient is patient_id = {patient_id}."]

    if task in ("patient_diagnosis", "atypical_diagnosis") and encounter_id:
        parts.append(
            f"Index encounter_id = {encounter_id}. Diagnose THIS visit: report the diagnosis established at the index "
            "encounter (ICD-10-CM code, name, acuity), using only the chart up to and including it. Earlier encounters "
            "are context; conditions documented before the index visit are not the target.")
    elif task == "differential_diagnosis" and encounter_id:
        parts.append(f"Index encounter_id = {encounter_id}. Rank the differential for THIS visit (at most 5, most likely first).")
    elif task == "test_selection" and encounter_id:
        parts.append(f"Index encounter_id = {encounter_id}. Its results are hidden: use order_test(name) to obtain the ones "
                     "you need, then submit_workup with the diagnosis you established.")
    elif task == "error_detection" and encounter_id:
        parts.append(f"Index encounter_id = {encounter_id}. One of its sections carries an injected documentation error; find it.")
    elif task == "lab_triage" and encounter_id:
        parts.append(f"Index encounter_id = {encounter_id}. Triage its laboratory and vital-sign results.")
    elif task == "evidence_retrieval" and diagnosis_names:
        parts.append(f"Target diagnoses: {diagnosis_names}")
    elif task == "context_summarization" and _extra.get("specialty_conditioned"):
        # Stage 8 smoke: without this the only mention of `abstain` was a field description, and agents that
        # correctly found no problem in the specialty wrote it in prose and scored 0 (abstention is read from the
        # field, never the text, by design: Stage 3)
        parts.append("This is a specialty-conditioned summary. If the patient has no active problem in the requested "
                     "specialty, call submit_summary with abstain=true and an empty summary; otherwise set abstain=false "
                     "and write the summary.")
    elif task == "imaging_indication" and encounter_id:
        parts.append(f"Imaging order encounter_id = {encounter_id}")
        if modality:
            parts.append(f"Modality: {modality}")
        if body_region:
            parts.append(f"Body region: {body_region}")
        if clinical_indication:
            parts.append(f"Clinical indication: {clinical_indication}")

    parts.append("Begin your investigation now.")
    return "\n".join(parts)
