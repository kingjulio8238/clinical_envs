"""Results sync and merge for runs split across Modal workspaces (P10 of audit/PRE_OCT1_TODO.md).

    # outputs of one job, from one workspace's results volume → results/modal/<profile>/<run-name>/
    python scripts/sync_runs.py pull --profile sales-32662 --run c2-public
    # resume a stopped run on another workspace: its partial run directory goes to the same path on that volume,
    # then the same command reruns there and skips every recorded gt_id
    python scripts/sync_runs.py push --profile founders-78536 --run c2-public --part 0 \
        --src results/modal/sales-32662/c2-public/part0/qwen3.5-9b-local__patient_diagnosis__agent__public__s0
    # merge every copy / shard of each protocol run found under the inputs into one directory per run
    python scripts/sync_runs.py merge --out results/local/base results/modal/*/c2-*
    # merge C4 (group variance) shards and recompute the prompt filter
    python scripts/sync_runs.py merge-gv --out results/local/c4 results/modal/*/c4-*

`merge` refuses runs whose validity-determining settings differ (model, limits, sampling, prompt, reward
fingerprint, ...), keeps one record per gt_id (a record without an error beats one with; else the later one), checks
the union against the unit's full sample, and records the sources, duplicates, errors and any missing gt_ids in the
merged manifest. A merged run with missing gt_ids is written but marked `complete: false` and the command exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOLUME = "clinical-envs-results"
MUST_MATCH = ("model", "model_id", "task", "arm", "split", "seed", "n_requested", "budget", "max_turns", "limits",
              "local", "extra", "max_output_tokens_per_turn", "temperature", "prompt_hash", "reward_version",
              "reward_fingerprint")
"""Manifest fields that decide what a run measures: two pieces of one run must agree on all of them."""


def _modal(profile: str, *args: str) -> None:
    env = {**os.environ, "MODAL_PROFILE": profile}
    print(f"$ MODAL_PROFILE={profile} modal {' '.join(args)}", flush=True)
    r = subprocess.run(["modal", *args], env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"modal failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")


def pull(profile: str, run: str, dest: Path) -> Path:
    target = dest / profile
    target.mkdir(parents=True, exist_ok=True)
    _modal(profile, "volume", "get", "--force", VOLUME, run, str(target))
    out = target / run
    if not out.exists():
        raise SystemExit(f"pulled nothing: {out} does not exist")
    runs = find_runs([out])
    print(f"{out}: {len(runs)} protocol run(s)" + "".join(f"\n  {d.relative_to(out)}: {_count(d)} records" for d in runs))
    return out


def push(profile: str, src: Path, run: str, part: int) -> None:
    if not (src / "predictions.jsonl").exists():
        raise SystemExit(f"{src} is not a protocol-run directory (no predictions.jsonl)")
    _modal(profile, "volume", "put", "--force", VOLUME, str(src), f"{run}/part{part}/{src.name}")


def _count(d: Path) -> int:
    return sum(1 for line in (d / "predictions.jsonl").read_text().splitlines() if line.strip())


def find_runs(roots: list[Path]) -> list[Path]:
    """Every protocol-run directory (manifest.json + predictions.jsonl) at or under the roots."""
    out = []
    for r in roots:
        for m in sorted(Path(r).rglob("manifest.json")):
            if (m.parent / "predictions.jsonl").exists():
                out.append(m.parent)
    return out


def _preds(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "predictions.jsonl").read_text().splitlines() if line.strip()]


def merge_run(dirs: list[Path], out: Path, db=None) -> dict:
    """Merge the pieces of one protocol run (same run_id) into `out`; returns the merged manifest."""
    from eval import protocol_run as R
    mans = [json.loads((d / "manifest.json").read_text()) for d in dirs]
    ref = mans[0]
    for d, m in zip(dirs[1:], mans[1:]):
        diff = [k for k in MUST_MATCH if m.get(k) != ref.get(k)]
        if diff:
            raise ValueError(f"{d} differs from {dirs[0]} in {diff}: not the same run")
    if any(m.get("reward_drift_allowed") for m in mans):
        raise ValueError(f"{ref['run_id']}: a piece ran with reward drift allowed; rescore it under the frozen reward first")
    db = db or R.shared_db()
    expected = [i["gt_id"] for i in R.sample_instances(db, ref["task"], ref["split"], ref["n_requested"], ref["seed"])]
    keep: dict = {}
    dups = 0
    for d in dirs:
        for p in _preds(d):
            g = p["gt_id"]
            if g in keep:
                dups += 1
                if p.get("error") and not keep[g].get("error"):
                    continue
            keep[g] = p
    stray = sorted(set(keep) - set(expected), key=str)
    if stray:
        raise ValueError(f"{ref['run_id']}: gt_ids outside the unit's sample {stray[:5]}: not the same sample")
    missing = [g for g in expected if g not in keep]
    merged = [keep[g] for g in expected if g in keep]
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(p, default=str) + "\n" for p in merged))
    man = {k: v for k, v in ref.items() if k not in ("gt_ids", "shard")}
    man.update(n_sampled=len(expected), n_recorded=len(merged), complete=not missing, missing=missing,
               cost_usd=round(sum(float(m.get("cost_usd") or 0) for m in mans), 4),
               duration_s=round(sum(float(m.get("duration_s") or 0) for m in mans), 1),
               errors=sum(1 for p in merged if p.get("error")), duplicates_dropped=dups,
               stopped=[m.get("stopped") for m in mans if m.get("stopped")] or None,
               merged_from=[{"path": str(d), "shard": m.get("shard"), "n": _count(d), "git_commit": m.get("git_commit"),
                             "started": m.get("started")} for d, m in zip(dirs, mans)])
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


def merge(roots: list[Path], out_root: Path) -> int:
    from eval import protocol_run as R
    groups: dict[str, list[Path]] = {}
    for d in find_runs(roots):
        if out_root.resolve() in d.resolve().parents:
            continue                                   # never re-read an earlier merge output
        rid = json.loads((d / "manifest.json").read_text())["run_id"]
        groups.setdefault(rid, []).append(d)
    if not groups:
        raise SystemExit(f"no protocol runs under {', '.join(map(str, roots))}")
    db = R.shared_db()
    rc = 0
    for rid, dirs in sorted(groups.items()):
        man = merge_run(dirs, out_root / rid, db)
        flag = "complete" if man["complete"] else f"INCOMPLETE, {len(man['missing'])} missing"
        print(f"{rid}: {len(dirs)} piece(s), {man['n_recorded']}/{man['n_sampled']} ({flag}), "
              f"{man['duplicates_dropped']} duplicates dropped, {man['errors']} errors")
        rc |= not man["complete"]
    return rc


def merge_gv(roots: list[Path], out: Path) -> int:
    """Group-variance (C4) shards: concatenate samples.jsonl, dedupe, recompute summary.json and prompts.json."""
    from eval import reward_version as RV
    from scripts import group_variance as GV
    files = [f for r in roots for f in sorted(Path(r).rglob("samples.jsonl")) if out.resolve() not in f.resolve().parents]
    if not files:
        raise SystemExit("no samples.jsonl under the inputs")
    samples = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    models = {json.loads(f.with_name("summary.json").read_text()).get("model") for f in files if f.with_name("summary.json").exists()}
    ks = {json.loads(f.with_name("prompts.json").read_text()).get("k") for f in files if f.with_name("prompts.json").exists()}
    if len(models) > 1 or len(ks) > 1:
        raise ValueError(f"shards disagree: models {models}, k {ks}")
    k = ks.pop() if ks else max(s["sample"] for s in samples) + 1
    out.mkdir(parents=True, exist_ok=True)
    rows = GV.dedupe(samples)
    (out / "samples.jsonl").write_text("".join(json.dumps(s, default=str) + "\n" for s in rows))
    summ = GV.write_summary(out, rows, k, models.pop() if models else None, RV.current()["reward_version"])
    print(f"{len(files)} shard file(s), {len(samples)} rows → {len(rows)} samples; {summ['n_keep']} prompts kept")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("--profile", required=True); p.add_argument("--run", required=True)
    p.add_argument("--dest", default=str(ROOT / "results" / "modal"))
    p = sub.add_parser("push"); p.add_argument("--profile", required=True); p.add_argument("--run", required=True)
    p.add_argument("--part", type=int, default=0); p.add_argument("--src", required=True)
    p = sub.add_parser("merge"); p.add_argument("--out", required=True); p.add_argument("roots", nargs="+")
    p = sub.add_parser("merge-gv"); p.add_argument("--out", required=True); p.add_argument("roots", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "pull":
        pull(a.profile, a.run, Path(a.dest))
        return 0
    if a.cmd == "push":
        push(a.profile, Path(a.src), a.run, a.part)
        return 0
    if a.cmd == "merge":
        return merge([Path(r) for r in a.roots], Path(a.out))
    return merge_gv([Path(r) for r in a.roots], Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
