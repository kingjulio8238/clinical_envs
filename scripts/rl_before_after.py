"""Before / after evaluation of an RL-trained checkpoint against the pre-registered success criteria
(audit/RL_SUCCESS_CRITERIA.md, criteria 1–7; P1 + P2 of audit/PRE_OCT1_TODO.md).

    python scripts/rl_before_after.py --base results/local/base --base-model qwen3.5-9b-local \
        --trained results/local/trained --trained-model qwen3.5-9b-rl \
        [--anchor results --anchor-model gpt-6-sol] [--splits public,heldout] \
        [--trained-audit results/local/trained/reward_noise_audit.json] [--private results/local/private] \
        [--out results/rl_before_after.md]

Each directory holds protocol-run directories (<model>__<unit>__<arm>__<split>__s0, as written by
eval.protocol_run; for Modal runs, the output of `scripts/sync_runs.py merge`). Everything is paired per instance with 95% bootstrap intervals (eval.protocol). Prints a table of
criteria with PASS / FAIL / N/A, the outcome row, and writes the report (markdown + JSON). A criterion whose inputs
are missing is N/A and the verdict says which inputs to add.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402
from eval import protocol as P  # noqa: E402
from eval import rl_monitor as M  # noqa: E402

TRAIN_UNITS = ("patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection")
TRANSFER = "atypical_diagnosis"
UNTRAINED = ("context_summarization", "specialty_conditioned", "imaging_indication", "lab_triage", "error_detection")
ABLATION = ("patient_diagnosis", "differential_diagnosis", "test_selection")
DX_UNITS = ("patient_diagnosis", "atypical_diagnosis", "differential_diagnosis", "test_selection")
GAP_SHARE, NONINFERIORITY, HELDOUT_RATIO, NOISE_MAX, PROBE_MAX, SIZE_MAX = 0.25, -0.03, 0.70, 0.05, 0.05, 2.0


def load(root: str | None, model: str | None) -> dict:
    """{(unit, arm, split): run} for one model in a results root."""
    if not root:
        return {}
    out = {}
    for r in P.load_runs(Path(root)):
        mf = r["manifest"]
        if model is None or mf.get("model") == model:
            out[(mf["task"], mf.get("arm", "agent"), mf.get("split", "public"))] = r
    return out


def pd(a: dict, b: dict) -> dict:
    """Paired a − b."""
    return P.paired_diff(a, b)


def metric_map(run: dict, key: str) -> dict:
    out = {}
    for p in run["predictions"]:
        v = (p.get("metrics") or {}).get(key)
        out[p["gt_id"]] = float(v) if isinstance(v, (int, float)) else 0.0
    return out


def rarity_terciles(db, unit: str, split: str, gt_ids) -> dict[int, int]:
    """P2: each instance's tercile (0 = rarest) by how often its reference diagnosis occurs as a label in the train
    split (min over the instance's reference diagnoses)."""
    freq = Counter()
    for i in db.instances("patient_diagnosis", "train"):
        for d in i["gt"].get("active_diagnoses", []) + i["gt"].get("chronic_conditions", []):
            if d.get("diagnosis_id") is not None:
                freq[d["diagnosis_id"]] += 1
    insts = {i["gt_id"]: i["gt"] for i in db.instances(unit, split)}
    rar = {}
    for g in gt_ids:
        gt = insts.get(g) or {}
        refs = gt.get("active_diagnoses", []) + gt.get("chronic_conditions", []) + gt.get("correct", []) + \
            ([gt["diagnosis"]] if isinstance(gt.get("diagnosis"), dict) else [])
        ids = [d.get("diagnosis_id") for d in refs if isinstance(d, dict) and d.get("diagnosis_id") is not None]
        if ids:
            rar[g] = min(freq.get(x, 0) for x in ids)
    if not rar:
        return {}
    ordered = sorted(rar, key=lambda g: (rar[g], g))
    n = len(ordered)
    return {g: min(2, (k * 3) // n) for k, g in enumerate(ordered)}


def fmt(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "—"
    return f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] (n={d['n']})"


def evaluate(a) -> dict:
    db = D.ReleaseDB()
    base, trained = load(a.base, a.base_model), load(a.trained, a.trained_model)
    anchor = load(a.anchor, a.anchor_model)
    splits = a.splits.split(",")
    rows, verdict = [], {}

    def row(crit, check, status, evidence):
        rows.append({"criterion": crit, "check": check, "status": status, "evidence": evidence})

    # ---------------------------------------------------------------- 1 and 2
    gains = {}
    for u in TRAIN_UNITS:
        for s in splits:
            b, t = base.get((u, "agent", s)), trained.get((u, "agent", s))
            gains[(u, s)] = pd(t["rewards"], b["rewards"]) if b and t else None
    counted = [u for u in TRAIN_UNITS if all(gains.get((u, s)) and gains[(u, s)]["n"] and gains[(u, s)]["ci"][0] > 0 for s in splits)]
    have1 = all(gains.get((u, s)) for u in TRAIN_UNITS for s in splits)
    row("1", f"significant gain on ≥ 3 of 4 training units, on {' and '.join(splits)}",
        ("PASS" if len(counted) >= 3 else "FAIL") if have1 else "N/A",
        "; ".join(f"{u} {s}: {fmt(gains.get((u, s)))}" for u in TRAIN_UNITS for s in splits))
    thresholds, c2_ok, c2_have = {}, [], True
    for u in TRAIN_UNITS:
        b, an = base.get((u, "agent", "public")), anchor.get((u, "agent", "public"))
        if not (b and an):
            c2_have = False
            continue
        gap = pd(an["rewards"], b["rewards"])
        thresholds[u] = GAP_SHARE * gap["mean"]
        g = gains.get((u, "public"))
        c2_ok.append((u, g is not None and g["n"] > 0 and g["mean"] >= thresholds[u], gap["mean"], g["mean"] if g else None))
    row("2", f"public gain ≥ {GAP_SHARE:.0%} of the base-to-anchor gap on the units counted in 1",
        ("PASS" if counted and all(ok for u, ok, _, _ in c2_ok if u in counted) else "FAIL") if c2_have and have1 else "N/A",
        "; ".join(f"{u}: gap {gap:+.3f} → bar {GAP_SHARE * gap:+.3f}, gain {gm:+.3f}" if gm is not None else f"{u}: gap {gap:+.3f}"
                  for u, _, gap, gm in c2_ok))

    # ---------------------------------------------------------------- 3
    named = {}
    for u in ("patient_diagnosis", "test_selection"):
        b, t = base.get((u, "agent", "public")), trained.get((u, "agent", "public"))
        named[u] = pd(metric_map(t, "diagnosis_named"), metric_map(b, "diagnosis_named")) if b and t else None
    coded = {}
    for u in ("patient_diagnosis", "test_selection"):
        b, t = base.get((u, "agent", "public")), trained.get((u, "agent", "public"))
        coded[u] = pd(metric_map(t, "diagnosis_coded"), metric_map(b, "diagnosis_coded")) if b and t else None
    have3 = all(named.values())
    row("3a", "diagnosis_named rises significantly on patient_diagnosis and test_selection",
        ("PASS" if all(d["n"] and d["ci"][0] > 0 for d in named.values()) else "FAIL") if have3 else "N/A",
        "; ".join(f"{u}: named {fmt(named[u])}, coded {fmt(coded[u])}" for u in named))
    if a.trained_audit and Path(a.trained_audit).exists():
        au = json.loads(Path(a.trained_audit).read_text())
        rates = {u: au.get(u, {}).get("correct_zero_rate") for u in DX_UNITS}
        row("3b", f"judge-audited correct-but-0 ≤ {NOISE_MAX:.0%} on every diagnosis unit (trained)",
            "PASS" if all(r is None or r <= NOISE_MAX for r in rates.values()) else "FAIL",
            "; ".join(f"{u} {'—' if r is None else f'{r:.1%}'}" for u, r in rates.items()))
    else:
        row("3b", f"judge-audited correct-but-0 ≤ {NOISE_MAX:.0%} (trained)", "N/A", "run scripts/reward_noise_audit.py on the trained runs")
    probe_notes, probe_ok, have_sig = [], True, False
    for u in TRAIN_UNITS + (TRANSFER,):
        b, t = base.get((u, "agent", "public")), trained.get((u, "agent", "public"))
        if not (b and t) or not any(p.get("signals") for p in t["predictions"]):
            continue
        have_sig = True
        ab, at = M.aggregate(b["predictions"]), M.aggregate(t["predictions"])
        probes = {k: at.get(k) or 0 for k in ("probe_many_entries", "probe_long_name", "probe_duplicates")}
        sizes = {k: (at.get(k) or 0) / (ab.get(k) or 1e-9) for k in ("entries", "answer_chars", "max_name_tokens") if ab.get(k)}
        bad = [k for k, v in probes.items() if v > PROBE_MAX] + [f"{k}×{v:.1f}" for k, v in sizes.items() if v > SIZE_MAX]
        probe_ok &= not bad
        probe_notes.append(f"{u}: " + (", ".join(bad) if bad else "clean"))
    row("3c", f"probe patterns < {PROBE_MAX:.0%}, answer size < {SIZE_MAX}× base", ("PASS" if probe_ok else "FAIL") if have_sig else "N/A",
        "; ".join(probe_notes) or "no monitor signals in the runs")

    # ---------------------------------------------------------------- 4
    tr = {}
    for s in splits:
        b, t = base.get((TRANSFER, "agent", s)), trained.get((TRANSFER, "agent", s))
        tr[s] = pd(t["rewards"], b["rewards"]) if b and t else None
    have4a = any(tr.values())
    row("4a", f"atypical transfer: lower bound > {NONINFERIORITY}", ("PASS" if all(d["ci"][0] > NONINFERIORITY for d in tr.values() if d) else "FAIL") if have4a else "N/A",
        "; ".join(f"{s}: {fmt(d)}" for s, d in tr.items()))
    if "heldout" in splits and "public" in splits and have1:
        ratios = {u: (gains[(u, "heldout")]["mean"] / gains[(u, "public")]["mean"]) if gains[(u, "public")]["mean"] > 0 else None for u in counted}
        row("4b", f"heldout gain ≥ {HELDOUT_RATIO:.0%} of public on the counted units",
            "PASS" if counted and all(r is not None and r >= HELDOUT_RATIO for r in ratios.values()) else "FAIL",
            "; ".join(f"{u}: {r:.2f}" if r is not None else f"{u}: —" for u, r in ratios.items()))
    else:
        row("4b", "heldout vs public", "N/A", "needs both splits")
    tert_notes, tert_ok, have_t = [], True, False
    for u in ("patient_diagnosis", "differential_diagnosis", "test_selection"):
        b, t = base.get((u, "agent", "public")), trained.get((u, "agent", "public"))
        if not (b and t):
            continue
        common = set(b["rewards"]) & set(t["rewards"])
        terc = rarity_terciles(db, u, "public", common)
        rare = {g for g, k in terc.items() if k == 0}
        if not rare:
            continue
        have_t = True
        d = pd({g: t["rewards"][g] for g in rare}, {g: b["rewards"][g] for g in rare})
        tert_ok &= d["mean"] > 0
        by_t = []
        for k in (0, 1, 2):
            ids = {g for g, kk in terc.items() if kk == k}
            dd = pd({g: t["rewards"][g] for g in ids}, {g: b["rewards"][g] for g in ids})
            by_t.append(f"T{k + 1} {dd['mean']:+.3f}")
        tert_notes.append(f"{u}: rarest {fmt(d)} ({', '.join(by_t)})")
    row("4c", "gain > 0 in the rarest third of training diagnoses (D2)", ("PASS" if tert_ok else "FAIL") if have_t else "N/A", "; ".join(tert_notes))

    # ---------------------------------------------------------------- 5
    reg, have5 = [], False
    for u in UNTRAINED:
        for s in splits:
            b, t = base.get((u, "agent", s)), trained.get((u, "agent", s))
            if b and t:
                have5 = True
                reg.append((u, s, pd(t["rewards"], b["rewards"])))
    underpowered5 = bool(reg) and all(d["mean"] >= 0 for _, _, d in reg) and not all(d["ci"][0] > NONINFERIORITY for _, _, d in reg)
    row("5", f"no regression on untrained units: lower bound > {NONINFERIORITY}",
        ("PASS" if all(d["ci"][0] > NONINFERIORITY for _, _, d in reg) else "FAIL") if have5 else "N/A",
        "; ".join(f"{u} {s}: {fmt(d)}" for u, s, d in reg) + (" — every mean ≥ 0: the interval is too wide (sample size), not a measured regression" if underpowered5 else ""))

    # ---------------------------------------------------------------- 6
    tools, have6 = [], False
    for u in ABLATION:
        bs, ba, ts, ta = (base.get((u, "single", "public")), base.get((u, "agent", "public")),
                          trained.get((u, "single", "public")), trained.get((u, "agent", "public")))
        if all((bs, ba, ts, ta)):
            have6 = True
            common = set(bs["rewards"]) & set(ba["rewards"]) & set(ts["rewards"]) & set(ta["rewards"])
            dd = pd({g: ta["rewards"][g] - ts["rewards"][g] for g in common}, {g: ba["rewards"][g] - bs["rewards"][g] for g in common})
            tools.append((u, dd))
    row("6a", "tools − no-tools gap holds (trained gap − base gap, lower bound > −0.03)",
        ("PASS" if all(d["ci"][0] > NONINFERIORITY for _, d in tools) else "FAIL") if have6 else "N/A",
        "; ".join(f"{u}: {fmt(d)}" for u, d in tools) or "needs single-arm runs for base and trained")
    b, t = base.get(("test_selection", "agent", "public")), trained.get(("test_selection", "agent", "public"))
    if b and t:
        ob = sum(p.get("orders", 0) for p in b["predictions"]) / max(len(b["predictions"]), 1)
        ot = sum(p.get("orders", 0) for p in t["predictions"]) / max(len(t["predictions"]), 1)
        row("6b", "test_selection orders per episode in (0, 2× base]", "PASS" if 0 < ot <= 2 * max(ob, 1e-9) else "FAIL", f"base {ob:.2f}, trained {ot:.2f}")
    else:
        row("6b", "test_selection orders per episode", "N/A", "")

    # ---------------------------------------------------------------- 7
    priv_b, priv_t = load(a.private, a.base_model), load(a.private, a.trained_model)
    b, t = priv_b.get(("patient_diagnosis", "agent", "private")), priv_t.get(("patient_diagnosis", "agent", "private"))
    if b and t and a.base_model != a.trained_model:
        d = pd(t["rewards"], b["rewards"])
        row("7", "private split once: significant patient_diagnosis gain", "PASS" if d["ci"][0] > 0 else "FAIL", fmt(d))
    else:
        row("7", "private split once", "N/A", "run once on the final checkpoint (audit/PRE_OCT1_TODO.md P5)")

    status = {r["criterion"]: r["status"] for r in rows}
    fails = [c for c, s in status.items() if s == "FAIL"]
    nas = [c for c, s in status.items() if s == "N/A"]
    c3_fail = any(status.get(c) == "FAIL" for c in ("3b", "3c"))
    if c3_fail:
        outcome = "reward hack: fix the scorer, re-lock, rescore both models, retrain"
    elif status.get("1") == "PASS" and status.get("2") == "PASS" and status.get("3a") == "FAIL":
        outcome = "partial: RL taught ICD coding"
    elif status.get("1") == "PASS" and any(status.get(c) == "FAIL" for c in ("4a", "4b", "4c")):
        outcome = "overfitting / prior-learning"
    elif status.get("5") == "FAIL" and status.get("1") == "PASS" and underpowered5:
        outcome = "inconclusive on criterion 5: untrained units need more instances (every mean ≥ 0, intervals too wide)"
    elif status.get("5") == "FAIL" and status.get("1") == "PASS":
        outcome = "regression on untrained units: the gain costs other skills (check KL / steps before scaling)"
    elif status.get("1") == "FAIL" or status.get("2") == "FAIL":
        outcome = "weak signal: check group variance, steps, learning rate"
    elif not fails and not nas:
        outcome = "RL on this environment boosts the model"
    else:
        outcome = "incomplete: " + ", ".join(nas) + " not evaluated" if nas and not fails else "fails: " + ", ".join(fails)
    return {"rows": rows, "outcome": outcome, "fails": fails, "not_evaluated": nas, "thresholds": thresholds,
            "counted_units": counted, "inputs": vars(a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--trained-model", required=True)
    ap.add_argument("--anchor", default=str(ROOT / "results"))
    ap.add_argument("--anchor-model", default="gpt-6-sol")
    ap.add_argument("--splits", default="public,heldout")
    ap.add_argument("--trained-audit", default=None)
    ap.add_argument("--private", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = evaluate(a)
    md = ["# RL before / after (audit/RL_SUCCESS_CRITERIA.md)", "",
          f"base `{a.base_model}` ({a.base}) vs trained `{a.trained_model}` ({a.trained}); splits {a.splits}", "",
          "| criterion | check | status | evidence |", "|---|---|---|---|"]
    md += [f"| {r['criterion']} | {r['check']} | **{r['status']}** | {r['evidence']} |" for r in res["rows"]]
    md += ["", f"**Outcome: {res['outcome']}**"]
    text = "\n".join(md) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    return 0 if res["outcome"] == "RL on this environment boosts the model" else 1


if __name__ == "__main__":
    sys.exit(main())
