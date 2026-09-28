"""P10: a unit split across workspaces (shards) or stopped on one and resumed on another merges into one run — one
record per instance, the full sample, the settings checked — and C4 shards merge into one prompt filter."""

from __future__ import annotations

import json

import pytest

from eval import degenerate as D
from eval import protocol_run as R
from eval.tests.test_protocol import ScriptedAdapter
from scripts import group_variance as GV
from scripts import sync_runs as S

pytestmark = pytest.mark.reward_hacking


@pytest.fixture(scope="module")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    return R.shared_db()


def _run(root, shard=None, n=4, adapter=None, **kw):
    return R.run_units("glm-5.3-flash", ["patient_diagnosis"], n, seed=21, workers=1, out_root=root, quiet=True,
                       adapter=adapter or ScriptedAdapter(R._env), prices=(0.0, 0.0), local=True, shard=shard, **kw)["patient_diagnosis"]


def _preds(d):
    return [json.loads(line) for line in (d / "predictions.jsonl").read_text().splitlines() if line.strip()]


def test_shards_merge_into_the_full_run(db, tmp_path):
    full = _run(tmp_path / "full")
    a = _run(tmp_path / "ws_a" / "part0", shard=(0, 2))
    b = _run(tmp_path / "ws_b" / "part0", shard=(1, 2))
    assert len(_preds(a)) == 2 and len(_preds(b)) == 2
    assert not {p["gt_id"] for p in _preds(a)} & {p["gt_id"] for p in _preds(b)}
    assert S.merge([tmp_path / "ws_a", tmp_path / "ws_b"], tmp_path / "merged") == 0
    m = tmp_path / "merged" / full.name
    man = json.loads((m / "manifest.json").read_text())
    assert man["complete"] and man["n_recorded"] == man["n_sampled"] == 4 and len(man["merged_from"]) == 2
    assert [p["gt_id"] for p in _preds(m)] == [p["gt_id"] for p in sorted(_preds(full), key=lambda p: [i["gt_id"] for i in
            R.sample_instances(db, "patient_diagnosis", "public", 4, 21)].index(p["gt_id"]))]
    assert {p["gt_id"]: p["reward"] for p in _preds(m)} == {p["gt_id"]: p["reward"] for p in _preds(full)}


def test_resume_on_another_workspace_and_retry_errors(db, tmp_path):
    first = _run(tmp_path / "ws_a", n=4, adapter=ScriptedAdapter(R._env, fail=True))   # server died: 4 error records
    assert all(p.get("error") for p in _preds(first))
    # the partial directory is pushed to the other workspace's volume; the rerun there retries the errored episodes
    dst = tmp_path / "ws_b" / first.name
    dst.mkdir(parents=True)
    (dst / "predictions.jsonl").write_text((first / "predictions.jsonl").read_text())
    again = _run(tmp_path / "ws_b", n=4, retry_errors=True)
    assert len(_preds(again)) == 8                                   # appended, not rewritten
    assert S.merge([tmp_path / "ws_b"], tmp_path / "merged") == 0
    merged = _preds(tmp_path / "merged" / first.name)
    assert len(merged) == 4 and not any(p.get("error") for p in merged)
    man = json.loads((tmp_path / "merged" / first.name / "manifest.json").read_text())
    assert man["duplicates_dropped"] == 4 and man["errors"] == 0


def test_merge_refuses_different_settings_and_flags_missing(db, tmp_path):
    a = _run(tmp_path / "ws_a", shard=(0, 2))
    b = _run(tmp_path / "ws_b", shard=(1, 2))
    man = json.loads((b / "manifest.json").read_text())
    man["limits"]["max_episode_output_tokens"] = 1
    (b / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="limits"):
        S.merge([tmp_path / "ws_a", tmp_path / "ws_b"], tmp_path / "merged")
    assert S.merge([tmp_path / "ws_a"], tmp_path / "merged_a") == 1        # half the sample: incomplete, exit 1
    assert len(json.loads((tmp_path / "merged_a" / a.name / "manifest.json").read_text())["missing"]) == 2


def test_group_variance_shards_merge(tmp_path):
    def rows(ids, samples, reward):
        return [{"task": "patient_diagnosis", "gt_id": g, "sample": s, "reward": reward(g, s), "error": None}
                for g in ids for s in samples]
    for name, ids in (("a", [1, 2]), ("b", [3])):
        d = tmp_path / name
        d.mkdir()
        data = rows(ids, range(4), lambda g, s: float(s % 2) if g != 2 else 0.0)
        (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
        GV.write_summary(d, data, 4, "qwen3.5-9b-local", "reward-v3")
    # a retried sample: an error row and its later success
    with (tmp_path / "b" / "samples.jsonl").open("a") as fh:
        fh.write(json.dumps({"task": "patient_diagnosis", "gt_id": 3, "sample": 0, "reward": 0.0, "error": "boom"}) + "\n")
    assert S.merge_gv([tmp_path / "a", tmp_path / "b"], tmp_path / "m") == 0
    keep = json.loads((tmp_path / "m" / "prompts.json").read_text())
    assert keep["keep"] == [1, 3] and keep["k"] == 4                   # prompt 2 has no spread
    assert len((tmp_path / "m" / "samples.jsonl").read_text().splitlines()) == 12
