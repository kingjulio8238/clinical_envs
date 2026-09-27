"""A deterministic stand-in for the LLM endpoint, for dry-running the ETL without a model (Stage 6).

Speaks the OpenAI chat-completions format at /v1/chat/completions (and the anthropic format at
/v1/messages). It recognizes the pipeline's prompts by their wording and returns a structurally valid
reply for each: a Stage-5a extraction, a Stage-7 section list, a Stage-6b typing list, a Stage-6c / 4c
results object, a Stage-8b profile, a Stage-9a timeline, a Stage-9b polished HPI, a Stage-10 summary.
`--fail-first N` makes the first N replies structurally invalid, to exercise the validation retry.

    python scripts/mock_llm_server.py --port 8765 [--fail-first 1]
    SH_LLM_BASE_URL=http://127.0.0.1:8765 SH_LLM_MODEL=mock python -m etl.main --stage 5 --pilot 3 --db /tmp/x.db

Also importable: `serve_in_thread(port)` returns (server, state) for tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply_for(prompt: str) -> object:
    p = prompt
    if "PRIMARY DIAGNOSIS" in p and "clinical_findings" in p:                      # stage 5a
        m = re.search(r"(\d{1,3})-year-old", p)
        return {"primary_diagnosis": {"name": "Acute appendicitis", "icd10_suggestion": "K35.80", "acuity": "acute",
                                      "category": "Gastrointestinal", "snomed_suggestion": "74400008"},
                "differential_diagnoses": [{"name": "Mesenteric adenitis", "icd10_suggestion": "I88.0", "acuity": "acute", "category": "Gastrointestinal"}],
                "secondary_diagnoses": [],
                "clinical_findings": [{"name": "Right lower quadrant pain", "finding_type": "symptom", "present": True, "relevance": "key"},
                                      {"name": "Fever", "finding_type": "vital_sign", "present": True, "relevance": "supporting", "value_text": "38.5°C", "value_numeric": 38.5},
                                      {"name": "Age", "finding_type": "demographic", "present": True, "relevance": "background", "value_text": f"{m.group(1) if m else '40'}-year-old"}]}
    if "segment it into standard EHR sections" in p or "EHR sections" in p:          # stage 7
        return [{"section_type": "chief_complaint", "text": "Abdominal pain.", "order": 0},
                {"section_type": "hpi", "text": "Right lower quadrant pain for one day with fever.", "order": 1},
                {"section_type": "vitals", "text": "T 38.5°C, HR 102.", "order": 2},
                {"section_type": "physical_exam", "text": "Guarding in the right lower quadrant.", "order": 3}]
    if "polished_hpi" in p:                                                            # stage 9b
        return {"polished_hpi": "The patient returns with one day of right lower quadrant pain and fever, unchanged from the prior visit's course."}
    if '"encounters"' in p or "timeline" in p.lower():                                 # stage 9a
        ids = [int(x) for x in re.findall(r"question_id[^0-9]*(\d+)", p)][:8] or [1]
        return {"encounters": [{"source_question_ids": [q], "encounter_date": f"2020-0{(i % 9) + 1}-15", "encounter_type": "ed",
                                "chief_complaint": "Abdominal pain", "department": "Emergency"} for i, q in enumerate(ids)]}
    if "home_medications" in p and "chronic_conditions" in p:                          # stage 8b
        return {"age_at_first_encounter": 40, "sex": "M", "race_ethnicity": "White", "occupation": "teacher", "insurance": "commercial",
                "smoking_status": "never", "alcohol_use": "social", "chronic_conditions": ["Hypertension"], "surgical_history": [],
                "family_history": ["Father: type 2 diabetes"], "allergies": ["NKDA"], "home_medications": [{"name": "lisinopril", "dose": "10 mg daily"}],
                "name": "Synthetic Patient"}
    if "results" in p and "[id:" in p:                                                 # stages 4c / 1d / 6c
        ids = re.findall(r"\[id:\s*(\d+)\]", p)
        return {"results": {i: {"topic": "Appendicitis", "organ_system": "Gastrointestinal", "subject": "Surgery"} for i in ids}}
    if "relationship" in p.lower() and "finding" in p.lower():                        # stage 6b
        ids = [int(x) for x in re.findall(r"finding_id[^0-9]*(\d+)", p)][:40]
        return [{"finding_id": i, "relationship": "commonly_seen", "frequency": "common"} for i in ids] or [{"finding_id": 1, "relationship": "commonly_seen"}]
    return {"summary": "Synthetic reference summary.", "must_include_findings": [], "clinical_question": "Evaluate for appendicitis."}


class _State:
    def __init__(self, fail_first: int = 0):
        self.requests = 0
        self.fail_first = fail_first
        self.lock = threading.Lock()


def make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = "".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in body.get("messages", []))
            with state.lock:
                state.requests += 1
                invalid = state.requests <= state.fail_first
            payload = {"garbled": True} if invalid else reply_for(prompt)
            text = json.dumps(payload)
            if self.path.endswith("/v1/messages"):
                self._send({"id": f"mock-{state.requests}", "type": "message", "content": [{"type": "text", "text": text}],
                            "usage": {"input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4}, "stop_reason": "end_turn"})
            else:
                self._send({"id": f"mock-{state.requests}", "object": "chat.completion", "model": body.get("model", "mock"),
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}})

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def serve_in_thread(port: int = 0, fail_first: int = 0) -> tuple[ThreadingHTTPServer, _State]:
    state = _State(fail_first)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fail-first", type=int, default=0)
    a = ap.parse_args(argv)
    server, _ = serve_in_thread(a.port, a.fail_first)
    print(f"mock LLM server on http://127.0.0.1:{server.server_address[1]} (fail-first {a.fail_first}); Ctrl-C to stop", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
