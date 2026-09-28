"""Numbers for the project write-up on the personal site (kjv2), straight from the stored runs and audits.

    python scripts/export_site.py [--out ../kjv2/react-app/src/data/clinical-rl/numbers.json]

Everything the site plots comes from here: the floors file, the stored protocol runs (results/<run>/), the judge
audits (results/reward_noise_audit_*.json, results/isa_overcredit_audit.json), the environment benchmarks
(audit/bench/) and the run manifests. The only typed-in numbers are the *before* values of the original release, which
come from the paper audit's scripts (audit/FINDINGS.md, cited per value) — the release's scorer no longer exists in
this repo. Re-run after every result that changes a number (the RL run adds the before/after section).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402
from eval import protocol as P  # noqa: E402

DEFAULT_OUT = ROOT.parent / "kjv2" / "react-app" / "src" / "data" / "clinical-rl" / "numbers.json"
UNITS = ["patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection", "atypical_diagnosis",
         "lab_triage", "context_summarization", "specialty_conditioned", "imaging_indication", "error_detection"]
TRAIN_UNITS = ["patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection"]
LABEL = {"patient_diagnosis": "Diagnosis", "differential_diagnosis": "Differential", "evidence_retrieval": "Evidence retrieval",
         "test_selection": "Test selection", "atypical_diagnosis": "Atypical presentation", "lab_triage": "Lab triage",
         "context_summarization": "Summarization", "specialty_conditioned": "Specialty summary",
         "imaging_indication": "Imaging indication", "error_detection": "Error detection"}

# The original release's scores for the zero-model exploits (audit/FINDINGS.md; audit/scripts/ against b047385).
RELEASE_EXPLOITS = [
    {"id": "copy", "label": "Copy the chart's problem list", "unit": "patient_diagnosis", "before": 0.843,
     "before_metric": "severity-weighted F1", "source": "FINDINGS #2", "after_policy": "copy_problem_list_plus_chronic",
     "after_metric": "weighted_problem_list_f1_neutral"},
    {"id": "hpi", "label": "Submit one HPI section", "unit": "evidence_retrieval", "before": 0.995,
     "before_metric": "P@5", "source": "FINDINGS #1", "after_policy": "single_hpi", "after_metric": "ndcg_10"},
    {"id": "type", "label": "Rank sections by type alone", "unit": "evidence_retrieval", "before": 0.972,
     "before_metric": "P@5", "source": "FINDINGS #11", "after_policy": "section_type_prior", "after_metric": "ndcg_10"},
    {"id": "dump", "label": "Paste the whole chart", "unit": "context_summarization", "before": 0.676,
     "before_metric": "finding F1", "source": "FINDINGS #11", "after_policy": "chart_dump", "after_metric": "clinical_f1"},
    {"id": "phrase", "label": "Append “No significant distress.”", "unit": "specialty_absent", "before": 1.0,
     "before_metric": "abstention accuracy", "source": "FINDINGS #4", "after_policy": "phrase_only",
     "after_metric": "abstention_accuracy"},
]
RELEASE_BEST_MODEL = {"patient_diagnosis": 0.732, "context_summarization": 0.550, "evidence_retrieval": 0.833}
"""Best model in the paper's Table 2 on the same column (FINDINGS #11)."""

# Judge-audited correct-but-scored-0 before the concept matcher (audit/RL_READINESS_TODO.md A1; same judge and runs).
NOISE_BEFORE = {"patient_diagnosis": 0.09, "atypical_diagnosis": 0.21, "differential_diagnosis": 0.0, "test_selection": 0.14}
# The optimisation-pressure exploit found before training (audit/RL_READINESS.md §1): one "diagnosis name" built from
# the 3,000 most common label words, scored under the Stage-8 name credit before the specificity cap.
NAME_SINK_BEFORE = {"differential_diagnosis": 0.27, "patient_diagnosis": 0.06}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def runs_by_key() -> dict:
    out = {}
    for r in P.load_runs(ROOT / "results"):
        mf = r["manifest"]
        out[(mf["model"], mf["task"], mf.get("arm", "agent"), mf.get("split", "public"))] = r
    return out


def summarize_run(r: dict) -> dict:
    vals = [float(p.get("reward") or 0) for p in r["predictions"]]
    lo, hi = P.bootstrap_ci(vals)
    nc = P._named_coded(r["predictions"])
    mu = sum(vals) / len(vals)
    return {"n": len(vals), "mean": mu, "lo": lo, "hi": hi, "named": nc["named"], "coded": nc["coded"],
            "errors": sum(1 for p in r["predictions"] if p.get("error")),
            "sd": (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5,
            "zero_share": sum(1 for v in vals if v <= 1e-9) / len(vals),
            "partial_share": sum(1 for v in vals if 1e-9 < v < 1 - 1e-9) / len(vals)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    floors = json.loads((ROOT / "eval" / "floors.json").read_text())["splits"]["public"]
    db = D.ReleaseDB(shared=True)

    corpus = {"patients": db.conn.execute("select count(*) from longitudinal_patients").fetchone()[0],
              "encounters": db.conn.execute("select count(*) from longitudinal_encounters").fetchone()[0],
              "units": {u: {s: len(db.instances(u, s)) for s in ("train", "public", "heldout")} for u in UNITS}}

    exploits = []
    for e in RELEASE_EXPLOITS:
        m = floors[e["unit"]]["metrics"][e["after_metric"]]
        exploits.append({**e, "after": m["policies"][e["after_policy"]], "best_model": RELEASE_BEST_MODEL.get(e["unit"])})

    probes = {u: {"n": len(ps), "max": max(floors[u]["metrics"][next(iter(floors[u]["metrics"]))]["policies"].get(p, 0)
                                          for p in ps),
                  "floor": floors[u]["metrics"][next(iter(floors[u]["metrics"]))]["floor"]}
              for u, ps in D.PROBES.items()}
    name_sink = {u: {"before": v, "after": floors[u]["metrics"][next(iter(floors[u]["metrics"]))]["policies"]["name_sink"]}
                 for u, v in NAME_SINK_BEFORE.items()}

    q = json.loads((ROOT / "results" / "reward_noise_audit_qwen.json").read_text())
    s = json.loads((ROOT / "results" / "reward_noise_audit_sol.json").read_text())
    isa = json.loads((ROOT / "results" / "isa_overcredit_audit.json").read_text())
    noise = {u: {"before": NOISE_BEFORE[u], "after": q[u]["correct_zero_rate"], "sol_after": s[u]["correct_zero_rate"],
                 "judged": q[u]["judged"]} for u in NOISE_BEFORE}

    runs = runs_by_key()
    baseline = {}
    for u in UNITS:
        qr, sr = runs.get(("qwen3.5-9b", u, "agent", "public")), runs.get(("gpt-6-sol", u, "agent", "public"))
        if not (qr and sr):
            continue
        qs, ss = summarize_run(qr), summarize_run(sr)
        gap = P.paired_diff(sr["rewards"], qr["rewards"])
        fl = floors[u]["metrics"][next(iter(floors[u]["metrics"]))]
        norm = (qs["mean"] - fl["floor"]) / (fl["ceiling"] - fl["floor"])
        single = runs.get(("qwen3.5-9b", u, "single", "public"))
        baseline[u] = {"label": LABEL[u], "qwen": qs, "sol": ss, "floor": fl["floor"], "ceiling": fl["ceiling"],
                       "gap": {"mean": gap["mean"], "lo": gap["ci"][0], "hi": gap["ci"][1], "n": gap["n"]},
                       "qwen_normalized": norm,
                       "go": 0.05 <= norm <= 0.80 and gap["mean"] >= 0.10 and gap["ci"][0] > 0 and qs["errors"] / qs["n"] <= 0.02,
                       "qwen_single": summarize_run(single) if single else None,
                       "min_gain": 0.25 * gap["mean"] if u in TRAIN_UNITS else None}

    bench = {}
    for name, f in (("http", "http_baseline.json"), ("local", "local_64_after.json")):
        d = json.loads((ROOT / "audit" / "bench" / f).read_text())
        bench[name] = d["runs"][0]["steps_per_s"]

    cost = {"openrouter": 0.0, "openai": 0.0, "episodes": 0}
    for m in glob.glob(str(ROOT / "results" / "**" / "manifest.json"), recursive=True):
        mf = json.loads(Path(m).read_text())
        key = "openai" if "openai.com" in (mf.get("provider_base_url") or "") else "openrouter"
        cost[key] += float(mf.get("cost_usd") or 0)
        cost["episodes"] += int(mf.get("n_recorded") or 0)

    ver = (ROOT / "audit" / "ROADMAP_VERIFICATION.md").read_text()
    m = re.search(r"\*\*(\d+) pass, (\d+) fail, (\d+) deferred\*\* \((\d+) tests passed", ver)
    lock = json.loads((ROOT / "eval" / "reward_lock.json").read_text())
    first = git("log", "--reverse", "--author=kingjulio8238", "--format=%as").splitlines()

    out = {
        "generated": time.strftime("%Y-%m-%d"), "commit": git("rev-parse", "--short", "HEAD"),
        "reward_version": lock.get("version"), "started": first[0] if first else None,
        "commits": len(first), "corpus": corpus, "exploits": exploits, "probes": probes,
        "probe_total": sum(v["n"] for v in probes.values()), "name_sink": name_sink, "noise": noise,
        "isa": {"judged": isa["judged"], "same": isa["same"], "related": isa["related"], "different": isa["different"],
                "over_credit_rate": isa["over_credit_rate"]},
        "baseline": baseline, "go_units": [u for u, v in baseline.items() if v["go"]], "bench": bench, "cost": cost,
        "verification": {"pass": int(m[1]), "fail": int(m[2]), "deferred": int(m[3]), "tests": int(m[4])} if m else None,
        "rl": None,   # filled by the before/after evaluation (scripts/rl_before_after.py) once the run exists
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {len(baseline)} units, go {out['go_units']}, cost ${cost['openrouter']:.2f} OR + ${cost['openai']:.2f} OpenAI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
