"""Stage 8 RL go/no-go (audit/STAGE8_TODO.md "success criteria"): per scoring unit, is there learnable headroom for
the RL candidate on this environment?

A unit qualifies when (1) the candidate's normalized score is in [0.05, 0.80] (above the zero-model floor, below
saturation), (2) the anchor beats it by >= 0.10 raw on the paired instances with a 95% interval excluding 0
(prompting alone reaches higher, so the environment rewards a reachable skill), and (3) the unit's episode error
rate is <= 2%. GO when at least three units qualify. The reward-noise audit (clinically correct answers scored 0)
is reported beside it from `results/reward_noise_audit.json` when present.

    python scripts/stage8_rl_decision.py [--candidate qwen3.5-9b] [--anchor gpt-6-sol] [--results results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import protocol as P  # noqa: E402
from eval.floors import load as load_floors  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="qwen3.5-9b")
    ap.add_argument("--anchor", default="gpt-6-sol")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--split", default="public")
    a = ap.parse_args()
    runs = P.load_runs(Path(a.results))
    floors = load_floors()
    by = {(r["manifest"]["model"], r["manifest"]["task"], r["manifest"]["arm"]): r for r in runs if r["manifest"].get("split") == a.split}
    noise_path = Path(a.results) / "reward_noise_audit.json"
    noise = json.loads(noise_path.read_text()) if noise_path.exists() else {}
    units = sorted({t for (m, t, arm) in by if m == a.candidate and arm == "agent"})
    rows, qualified = [], []
    for u in units:
        c = by[(a.candidate, u, "agent")]
        vals = list(c["rewards"].values())
        raw = sum(vals) / len(vals)
        norm = P.normalize(u, a.split, raw, floors)
        err = sum(1 for p in c["predictions"] if p.get("error")) / len(vals)
        s = by.get((a.anchor, u, "agent"))
        d = P.paired_diff(s["rewards"], c["rewards"]) if s else {"n": 0, "mean": float("nan"), "ci": (float("nan"),) * 2, "significant": False}
        ok_head = norm is not None and 0.05 <= norm <= 0.80
        ok_gap = d["n"] > 0 and d["mean"] >= 0.10 and d["ci"][0] > 0
        ok_err = err <= 0.02
        q = ok_head and ok_gap and ok_err
        if q:
            qualified.append(u)
        nz = noise.get(u, {}).get("correct_zero_rate")
        nz_s = f"{nz:.0%}" if isinstance(nz, (int, float)) else "—"
        lo, hi = P.bootstrap_ci(vals)
        rows.append(f"| {u} | {len(vals)} | {raw:.3f} [{P._fmt(lo)}, {P._fmt(hi)}] | {P._fmt(norm)} | {d['n']} | "
                    f"{P._fmt(d['mean'])} {P._ci(d['ci'])} | {err:.1%} | {nz_s} | "
                    f"{'yes' if ok_head else 'no'} / {'yes' if ok_gap else 'no'} / {'yes' if ok_err else 'no'} | "
                    f"{'**yes**' if q else 'no'} |")
    verdict = "GO" if len(qualified) >= 3 else "NO-GO"
    md = [f"# Stage 8 RL go/no-go — candidate `{a.candidate}`, anchor `{a.anchor}`, split `{a.split}`", "",
          "A unit qualifies when the candidate's normalized score is in [0.05, 0.80], the anchor − candidate paired "
          "difference is ≥ 0.10 with a 95% interval above 0, and the unit's error rate is ≤ 2%. GO needs ≥ 3 units.", "",
          "| unit | n | candidate raw [95% CI] | normalized | paired n | anchor − candidate [95% CI] | errors | "
          "correct-but-0 (audit) | headroom / gap / errors | qualifies |",
          "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          f"**{verdict}**: {len(qualified)} unit(s) qualify — {', '.join(qualified) or 'none'}."]
    out = Path(a.results) / "rl_decision.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
