"""Task-specific data loaders, prompt formatters, and output parsers."""

from eval.tasks import diagnosis, imaging, patient_diagnosis, retrieval, stage7, summarization

TASK_LOADERS = {
    "patient_diagnosis": patient_diagnosis.load_inputs,
    "context_summarization": summarization.load_inputs,
    "evidence_retrieval": retrieval.load_inputs,
    "imaging_indication": imaging.load_inputs,
    "differential_diagnosis": stage7.make_loader('differential_diagnosis'),
    "test_selection": stage7.make_loader('test_selection'),
    "error_detection": stage7.make_loader('error_detection'),
    "lab_triage": stage7.make_loader('lab_triage'),
    "atypical_diagnosis": stage7.make_loader('atypical_diagnosis'),
}

TASK_PARSERS = {
    "patient_diagnosis": patient_diagnosis.parse_output,
    "context_summarization": summarization.parse_output,
    "evidence_retrieval": retrieval.parse_output,
    "imaging_indication": imaging.parse_output,
    "differential_diagnosis": stage7.parse_output,
    "test_selection": stage7.parse_output,
    "error_detection": stage7.parse_output,
    "lab_triage": stage7.parse_output,
    "atypical_diagnosis": stage7.parse_output,
}

TASK_FORMATTERS = {
    "patient_diagnosis": patient_diagnosis.format_prompt,
    "context_summarization": summarization.format_prompt,
    "evidence_retrieval": retrieval.format_prompt,
    "imaging_indication": imaging.format_prompt,
    "differential_diagnosis": stage7.format_prompt,
    "test_selection": stage7.format_prompt,
    "error_detection": stage7.format_prompt,
    "lab_triage": stage7.format_prompt,
    "atypical_diagnosis": stage7.format_prompt,
}
