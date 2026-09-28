"""The frozen reward (RL readiness A4): a version name plus the SHA-256 of every file the reward depends on.

    python -m eval.reward_version            # print the version and whether the working tree matches the lock
    python -m eval.reward_version --check    # exit 1 if any reward file differs from eval/reward_lock.json
    python -m eval.reward_version --lock reward-v2   # deliberately re-lock after a reviewed scorer change

Training and before/after evaluation must run on one reward: the protocol runner records the version and
fingerprint in every manifest and refuses to start when the working scorer differs from the lock (override only
with --allow-reward-drift, which the manifest records). CI fails when a reward file changes without re-locking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = Path(__file__).with_name("reward_lock.json")
REWARD_FILES = (
    "eval/scoring.py", "eval/scoring_tasks7.py", "eval/score_one.py", "eval/stage7.py", "eval/concept_match.py",
    "eval/semantic_match.py", "eval/value_match.py", "eval/imaging_concepts.py", "eval/chart_neutral.py",
    "eval/diagnosis_aliases.json", "eval/private_labels.py",
)
"""Files whose content determines a reward: the scorers, the matchers, order matching (stage7), the concept aliases."""


class RewardDrift(RuntimeError):
    pass


def file_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in REWARD_FILES if (ROOT / f).exists()}


def fingerprint(hashes: dict[str, str] | None = None) -> str:
    h = hashes or file_hashes()
    return hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()[:16]


def lock() -> dict:
    return json.loads(LOCK.read_text()) if LOCK.exists() else {}


def drift() -> list[str]:
    """Reward files whose content differs from the lock (empty = frozen reward intact)."""
    locked = lock().get("files", {})
    now = file_hashes()
    return sorted(f for f in set(locked) | set(now) if locked.get(f) != now.get(f))


def current() -> dict:
    lk = lock()
    return {"reward_version": lk.get("version"), "reward_fingerprint": fingerprint(), "reward_locked_fingerprint": lk.get("fingerprint"),
            "reward_drift": drift()}


def require_frozen() -> dict:
    info = current()
    if info["reward_drift"]:
        raise RewardDrift(f"reward files differ from {LOCK.name} ({info['reward_version']}): {info['reward_drift']}; "
                          "re-lock deliberately with `python -m eval.reward_version --lock <version>`")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lock", metavar="VERSION")
    a = ap.parse_args(argv)
    if a.lock:
        h = file_hashes()
        LOCK.write_text(json.dumps({"version": a.lock, "fingerprint": fingerprint(h), "files": h}, indent=1, sort_keys=True) + "\n")
        print(f"locked {a.lock}: fingerprint {fingerprint(h)} over {len(h)} files")
        return 0
    info = current()
    print(json.dumps(info, indent=1))
    return 1 if (a.check and info["reward_drift"]) else 0


if __name__ == "__main__":
    sys.exit(main())
