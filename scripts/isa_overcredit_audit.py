"""Over-crediting audit for the is-a table (B4 open decision): every stored diagnosis answer whose credit rises only
because of an is-a alias is judged by the LLM judge; "different" means the table credited a wrong answer.

    python scripts/isa_overcredit_audit.py [--results results] [--judge gpt-6-sol]

Writes results/isa_overcredit_audit.json. Gate: over-credit rate ≤ 10% of the gained episodes.
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
from eval import scoring as S  # noqa: E402
from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from reward_noise_audit import SYSTEM, UNITS, _names  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--judge", default="gpt-6-sol")
    a = ap.parse_args()
    db = D.ReleaseDB()
    gained = []
    for run in sorted(Path(a.results).glob("*__*__*__public__s0")):
        mf = json.loads((run / "manifest.json").read_text())
        task = mf["task"]
        if task not in UNITS:
            continue
        insts = {i["gt_id"]: i for i in db.instances(task, "public")}
        for line in (run / "predictions.jsonl").read_text().splitlines():
            p = json.loads(line)
            if p.get("error") or not p.get("submission"):
                continue
            sub = dict(p["submission"])
            if task == "test_selection" and mf["arm"] == "agent":
                sub["tests_ordered"] = list(p.get("order_log") or [])
            S.ISA_ENABLED = False
            off = float(D.score(db, task, [sub], [insts[p["gt_id"]]])[D.PRIMARY_METRIC[task]])
            S.ISA_ENABLED = True
            on = float(D.score(db, task, [sub], [insts[p["gt_id"]]])[D.PRIMARY_METRIC[task]])
            if on > off + 1e-9:
                gt = insts[p["gt_id"]]["gt"]
                ref, ans = _names(task, gt, p["submission"])
                if task == "differential_diagnosis":     # a differential earns credit for the distractors too
                    ref = ref + [d.get("display_name") or "" for d in gt.get("distractors", [])]
                    ans = [f"{d.get('name') or ''} ({d.get('icd10') or ''})" for d in (p["submission"].get("differential") or [])[:5]
                           if isinstance(d, dict)]
                gained.append({"run": run.name, "unit": task, "gt_id": p["gt_id"], "gain": on - off, "reference": ref, "answer": ans})
    judge = create_adapter(MODEL_REGISTRY[a.judge])

    def ask(c):
        try:
            refs = "; ".join(c["reference"])
            head = ("Reference diagnoses (credit for naming any of them): " if c["unit"] == "differential_diagnosis" else "Reference diagnosis: ")
            r = judge.call(SYSTEM, f"{head}{refs}\nModel answer: {'; '.join(c['answer'])}")
            t = r.text.strip().strip("`"); t = t[t.find("{"): t.rfind("}") + 1]
            v = json.loads(t)
            return {**c, "verdict": v.get("verdict"), "reason": v.get("reason")}
        except Exception as exc:  # noqa: BLE001
            return {**c, "verdict": None, "reason": f"judge error: {exc}"[:200]}

    with ThreadPoolExecutor(8) as ex:
        judged = list(ex.map(ask, gained))
    ok = [j for j in judged if j["verdict"]]
    wrong = [j for j in ok if j["verdict"] == "different"]
    out = {"gained_episodes": len(judged), "judged": len(ok), "same": sum(j["verdict"] == "same" for j in ok),
           "related": sum(j["verdict"] == "related" for j in ok), "different": len(wrong),
           "over_credit_rate": (len(wrong) / len(ok)) if ok else 0.0, "cases": judged}
    (Path(a.results) / "isa_overcredit_audit.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "cases"})
    for j in wrong:
        print("  OVER-CREDIT", j["unit"], j["reference"], "|", j["answer"][:2], "|", j["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
