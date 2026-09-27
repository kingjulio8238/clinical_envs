"""Reward-noise audit (Stage-8 success criteria): of the diagnosis-scored episodes a model answered but scored 0,
how many named the reference diagnosis anyway? An LLM judge (GPT-6 Sol by default; audit only, never the reward)
compares the reference diagnosis names with the model's names and says same / related / different.

    python scripts/reward_noise_audit.py [--model qwen3.5-9b] [--judge gpt-6-sol] [--results results] [--max 400]

Writes results/reward_noise_audit.json: per unit, n zero-scored answered episodes, the judge's counts, and
correct_zero_rate = same / judged (the criterion is <= 20%), plus every judged case for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402
from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402

UNITS = ("patient_diagnosis", "atypical_diagnosis", "differential_diagnosis", "test_selection")
SYSTEM = ("You audit a clinical benchmark's scorer. Given the reference diagnosis (or diagnoses) and a model's answer, "
          "decide whether the model's answer names the reference diagnosis in a way a clinician would accept as the "
          "same diagnosis (possibly with a different code or wording, or more specific), a related but different or "
          "less specific diagnosis, or a different diagnosis. Reply with JSON only: "
          '{"verdict": "same" | "related" | "different", "reason": "<one short sentence>"}')


def _names(task: str, gt: dict, sub: dict) -> tuple[list[str], list[str]]:
    if task in ("patient_diagnosis", "atypical_diagnosis"):
        ref = [d.get("display_name") or "" for d in gt.get("active_diagnoses", []) + gt.get("chronic_conditions", [])
               if not d.get("excluded_nondiagnostic")]
        ans = [f"{d.get('name') or ''} ({d.get('icd10') or ''})" for d in (sub.get("active_diagnoses") or []) + (sub.get("chronic_conditions") or [])
               if isinstance(d, dict)]
    elif task == "differential_diagnosis":
        ref = [d.get("display_name") or "" for d in gt.get("correct", [])]
        ans = [f"{d.get('name') or ''} ({d.get('icd10') or ''})" for d in (sub.get("differential") or [])[:1] if isinstance(d, dict)]
    else:
        ref = [(gt.get("diagnosis") or {}).get("name") or ""]
        dx = sub.get("diagnosis") if isinstance(sub.get("diagnosis"), dict) else sub
        ans = [f"{dx.get('name') or ''} ({dx.get('icd10') or ''})"]
    return [r for r in ref if r], [a for a in ans if a.strip(" ()")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--judge", default="gpt-6-sol")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--split", default="public")
    ap.add_argument("--max", type=int, default=400, help="cap on judged cases per model (cost)")
    a = ap.parse_args()
    db = D.ReleaseDB()
    judge = create_adapter(MODEL_REGISTRY[a.judge])
    cases = []
    for u in UNITS:
        f = Path(a.results) / f"{a.model}__{u}__agent__{a.split}__s0" / "predictions.jsonl"
        if not f.exists():
            continue
        insts = {i["gt_id"]: i for i in db.instances(u, a.split)}
        for l in f.read_text().splitlines():
            p = json.loads(l)
            if float(p["reward"]) > 0 or not p.get("submission") or p.get("error"):
                continue
            ref, ans = _names(u, insts[p["gt_id"]]["gt"], p["submission"])
            if ref and ans:
                cases.append({"unit": u, "gt_id": p["gt_id"], "reference": ref, "answer": ans})
    cases = cases[: a.max]

    def ask(c):
        user = f"Reference diagnosis: {'; '.join(c['reference'])}\nModel answer: {'; '.join(c['answer'])}"
        try:
            r = judge.call(SYSTEM, user)
            t = r.text.strip().strip("`")
            t = t[t.find("{"): t.rfind("}") + 1]
            v = json.loads(t)
            return {**c, "verdict": v.get("verdict"), "reason": v.get("reason"), "cost_input_tokens": r.input_tokens}
        except Exception as exc:  # noqa: BLE001
            return {**c, "verdict": None, "reason": f"judge error: {exc}"[:200]}

    with ThreadPoolExecutor(8) as ex:
        judged = list(ex.map(ask, cases))
    out: dict = {"model": a.model, "judge": a.judge, "split": a.split}
    for u in UNITS:
        js = [j for j in judged if j["unit"] == u]
        ok = [j for j in js if j["verdict"] in ("same", "related", "different")]
        counts = {k: sum(1 for j in ok if j["verdict"] == k) for k in ("same", "related", "different")}
        out[u] = {"zero_scored_answered": len(js), "judged": len(ok), **counts,
                  "correct_zero_rate": (counts["same"] / len(ok)) if ok else None}
        print(f"{u:24s} zero-scored answered {len(js):4d}  same {counts['same']:3d}  related {counts['related']:3d}  "
              f"different {counts['different']:3d}  correct-but-0 {out[u]['correct_zero_rate'] if ok else '—'}")
    out["cases"] = judged
    (Path(a.results) / "reward_noise_audit.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
