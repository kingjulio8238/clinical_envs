"""Stage-8 pre-launch audit fixes (audit/STAGE8_TODO.md, "Pre-launch audit"): order matching by test name (R1),
name-equivalence diagnosis credit (R2), the combined specialty unit (R3), and the runner's cost, pooling, sharing,
order logging, single-arm and pairing guarantees (H1, H3-H7). No network."""

from __future__ import annotations

import json

import pytest

from eval import degenerate as D
from eval import protocol_run as R
from eval import stage7 as S7
from eval.adapters import ModelResponse
from eval.local_env import LocalEnv
from eval.scoring import dx_credit
from eval.tests.test_protocol import ScriptedAdapter

pytestmark = pytest.mark.reward_hacking


@pytest.fixture(scope="module")
def db():
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    return R.shared_db()


def _gt(db, task, gt_id):
    return next(i for i in db.instances(task, "public") if i["gt_id"] == gt_id)["gt"]


# ---------------------------------------------------------------------------
# R1: an order by test name reaches the finding named by its result
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gt_id,query,expect", [
    (77983, "audiogram", "Bilateral symmetric high-frequency hearing loss"),
    (77983, "hearing test", "Bilateral symmetric high-frequency hearing loss"),
    (76879, "fern test", "Ferning test"),
    (77371, "stress test", "SPECT myocardial perfusion imaging reversible defect"),
    (77371, "nuclear stress test", "SPECT myocardial perfusion imaging reversible defect"),
    (77371, "myocardial perfusion imaging", "SPECT myocardial perfusion imaging reversible defect"),
    (77371, "Exercise stress test", "SPECT myocardial perfusion imaging reversible defect"),     # re-smoke orders
    (77371, "Cardiac stress test", "SPECT myocardial perfusion imaging reversible defect"),
    (76924, "Obstetric ultrasound", "Amniotic fluid index"),
    (76846, "24-hour urine protein", "Proteinuria"),
    (77518, "serum sodium", "Hyponatremia"),                                                        # matcher check (Sol, n=20)
    (77518, "Basic metabolic panel", "Hyponatremia"),
    (77652, "ABG", "Hypercapnia"),
    (77755, "iron studies", "Serum iron"),                                                          # matcher check (Qwen, n=20)
    (77213, "Lyme serology", "Borrelia burgdorferi antibody"),
])
def test_order_by_test_name_reveals_the_result_named_finding(db, gt_id, query, expect):
    orderable = S7.orderable_from_gt(_gt(db, "test_selection", gt_id))
    assert all(f.get("context") is not None for f in orderable), "order context missing: run build_stage7_tasks.py --patch-order-context"
    r = S7.order_result(query, orderable)
    matched = r["matched"] if isinstance(r["matched"], list) else [r["matched"]]
    assert expect in matched, r


def test_order_of_an_unrelated_test_reveals_nothing(db):
    orderable = S7.orderable_from_gt(_gt(db, "test_selection", 77371))
    for q in ("chest CT", "lipase", "urine culture", "Stress echocardiogram", "Cardiac catheterization", "xray ultrasound"):
        assert S7.order_result(q, orderable)["matched"] is None, q


def test_generic_orders_do_not_reveal_the_chart(db):
    """A query of generic words ("labs", "imaging", "shows") matches by name only: no single word is a shotgun."""
    insts = db.instances("test_selection", "public")
    for q in ("labs", "imaging", "shows", "results", "test", "level", "blood test", "normal", "obstetric cardiac urine fetal"):
        hit = 0
        for i in insts:
            r = S7.order_result(q, S7.orderable_from_gt(i["gt"]))
            m = r["matched"] if isinstance(r["matched"], list) else [r["matched"]] if r["matched"] else []
            hit += bool(set(m) & set(i["gt"]["discriminating"]))
        assert hit / len(insts) < 0.05, (q, hit)


def test_order_context_uses_name_number_and_wording():
    sections = [(1, "labs", "Laboratory studies:\n- Platelet count: 95,000/uL\n- Hemoglobin: 11.2 g/dL\nBasic metabolic panel reveals a decreased serum potassium level."),
                (2, "imaging", "Audiometry reveals bilateral, symmetric high-frequency hearing loss."),
                (3, "hpi", "Audiometry was never done.")]
    orderable = [{"name": "Thrombocytopenia", "value": "95,000/uL"}, {"name": "Hypokalemia", "value": "decreased serum potassium"},
                 {"name": "Bilateral symmetric high-frequency hearing loss", "value": None}, {"name": "Rinne test", "value": "normal"}]
    ctx = S7.order_context(sections, orderable)
    assert "platelet" in ctx["Thrombocytopenia"].split() and "hemoglobin" not in ctx["Thrombocytopenia"].split()
    assert "potassium" in ctx["Hypokalemia"].split()
    assert "audiometry" in ctx["Bilateral symmetric high-frequency hearing loss"].split()
    assert ctx["Rinne test"] == ""                                 # not in a result section: name matching only
    rows = [{**f, "context": ctx[f["name"]]} for f in orderable]
    assert S7.order_matches("platelets", rows) == ["Thrombocytopenia"]
    assert S7.order_matches("audiogram", rows) == ["Bilateral symmetric high-frequency hearing loss"]


def test_single_turn_workup_is_matched_like_order_test(db):
    gt = _gt(db, "test_selection", 77983)
    from eval.scoring_tasks7 import score_test_selection_item
    s = score_test_selection_item({"icd10": "H91.10", "name": "Presbycusis", "tests_ordered": ["audiogram"]}, gt)
    assert s["discriminating_ordered"] == 1 and s["workup_coverage"] == 1.0


# ---------------------------------------------------------------------------
# R2: name-equivalence credit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred_name,gt_name,pc,gc,expect", [
    ("Presbycusis", "Presbycusis", "H90.5", "H91.13", 0.75),                                            # same name, other code
    ("Preterm premature rupture of membranes", "Preterm premature rupture of membranes (PPROM)", "O42.10", "P01.1", 0.75),
    ("Stable angina pectoris", "Stable Angina Pectoris due to Coronary Artery Disease", "I20.9", "I25.119", 0.5),
    ("Tourette's disorder", "Tourette Syndrome", "F95.0", "F95.2", 0.75),
    ("Pulmonary hypertension", "Essential hypertension", "I27.0", "I10", 0.0),                           # other block
    ("Type 1 diabetes mellitus", "Type 2 diabetes mellitus", "E10.9", "E11.9", 0.0),                     # contradicting digit
    ("Acute pancreatitis", "Chronic pancreatitis", "K85.9", "K86.1", 0.0),                               # contradicting qualifier
    ("Non-ST elevation myocardial infarction", "ST elevation myocardial infarction", "I25.2", "I20.0", 0.0),
    ("Anemia", "Sickle cell anemia", "D64.9", "D57.1", 0.0),
    ("Whatever", "Presbycusis", "H91.13", "H91.13", 1.0),                                              # the exact code still wins
    # reward-noise audit of the full run's zeros (GPT-6 Sol): the reference named in full inside a more specific name
    ("Sepsis due to right lower lobe pneumonia with septic shock", "Septic shock", "A41.9", "R65.21", 0.5),
    ("Acute graft-versus-host disease following allogeneic HSCT", "Acute graft-versus-host disease", "T86.09", "D89.810", 0.5),
    ("Gastroesophageal reflux in infant (physiologic reflux without esophagitis)",
     "Physiologic gastroesophageal reflux (Infant regurgitation)", "K21.9", "P92.1", 0.5),
    ("Pulmonary hypertension", "Hypertension", "I27.0", "I10", 0.0),                                    # one-word reference
    # RL-readiness probe: a more specific name may add qualifiers, not a catalogue of diseases
    ("Injury of radial nerve at upper arm level, left arm, initial encounter (radial nerve palsy with wrist and finger "
     "drop following humeral shaft fracture)", "Radial nerve injury", "S44.22XA", "S54.21XA", 0.5),
    ("septic shock pneumonia heart failure kidney injury stroke sepsis asthma cirrhosis lupus gout", "Septic shock",
     "Z99.9", "R65.21", 0.0),
    ("Shock (septic shock pneumonia heart failure kidney injury stroke sepsis asthma cirrhosis lupus gout)", "Septic shock",
     "Z99.9", "R65.21", 0.0),
    ("Mucolipidosis type II (I-cell disease)", "Mucolipidosis II (I-cell disease)", "Q77.1", "E77.0", 0.5),   # "I-cell" is no digit
    ("Chronic kidney disease stage 3", "Acute kidney injury", "N18.3", "N17.9", 0.0),
    ("Traumatic compartment syndrome of left lower extremity", "Compartment syndrome of hand", "T79.A22A", "T79.A12A", 0.0),
])
def test_name_credit(pred_name, gt_name, pc, gc, expect):
    assert dx_credit(pc, gc, pred_name, gt_name) == pytest.approx(expect)


def test_name_credit_reaches_every_diagnosis_scorer(db):
    from eval.scoring import score_patient_diagnosis_item
    from eval.scoring_tasks7 import score_differential_item, score_test_selection_item
    gt = {"active_diagnoses": [{"icd10": "H91.13", "display_name": "Presbycusis", "acuity": "chronic"}], "chronic_conditions": []}
    pd = score_patient_diagnosis_item({"active_diagnoses": [{"icd10": "H90.5", "name": "Presbycusis", "acuity": "chronic"}]}, gt)
    assert pd["problem_list_recall"] == pytest.approx(0.75)
    dd = score_differential_item({"differential": [{"icd10": "A01.41", "name": "Congenital toxoplasmosis"}]},
                                 {"correct": [{"icd10": "P37.1", "display_name": "Congenital toxoplasmosis"}], "distractors": []})
    assert dd["differential_top1"] == pytest.approx(0.75)
    ts = score_test_selection_item({"icd10": "H90.5", "name": "Presbycusis", "tests_ordered": []},
                                   {"diagnosis": {"icd10": "H91.13", "name": "Presbycusis"}, "discriminating": ["x"], "orderable": []})
    assert ts["workup_icd_credit"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# R3: the specialty task is scored as served (involved/absent mixture); abstaining is not a ceiling
# ---------------------------------------------------------------------------

def test_specialty_conditioned_abstain_is_not_saturated(db):
    insts = R.sample_instances(db, "specialty_conditioned", "public", 60, 0)
    assert {i["gt"].get("involvement") for i in insts} == {"involved", "absent"}
    preds = [D.POLICIES["specialty_absent"]["abstain_always"](db, i) for i in insts]
    m = D.score(db, "specialty_conditioned", preds, insts)
    assert m["specialty_reward"] < 0.75, m


# ---------------------------------------------------------------------------
# H1, H3-H7: the runner
# ---------------------------------------------------------------------------

class Billed(ScriptedAdapter):
    def call_multi_turn_with_retry(self, system, messages, tools=None, max_retries=None):
        r = super().call_multi_turn_with_retry(system, messages, tools, max_retries)
        r.cost_usd, r.provider = 0.5, "TestProvider"
        return r


def test_runner_pools_units_uses_billed_cost_and_caps_spend(tmp_path, db):
    outs = R.run_units("glm-5.3-flash", ["patient_diagnosis", "lab_triage", "test_selection"], n=2, seed=4, workers=3,
                       out_root=tmp_path, adapter=Billed(R._env), prices=(1.0, 2.0), quiet=True)
    assert set(outs) == {"patient_diagnosis", "lab_triage", "test_selection"}
    for task, out in outs.items():
        preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
        assert len(preds) == 2 and all(p["reward"] >= 1 - 1e-9 for p in preds), (task, preds)
        assert all(p["cost_usd"] == pytest.approx(p["billed_usd"]) and p["billed_usd"] > 0 for p in preds)
        assert all(p["list_cost_usd"] > 0 and p["providers"] == {"TestProvider": p["turns"]} for p in preds)
        if task == "test_selection":                                   # H5: the order trace is recorded
            assert all(p["orders"] >= 1 and p["order_log"] and p["unmatched_orders"] == 0 for p in preds)
        mf = json.loads((out / "manifest.json").read_text())
        assert mf["task"] == task and mf["n_recorded"] == 2
    capped = R.run_units("glm-5.3-flash", ["patient_diagnosis"], n=6, seed=5, workers=1, out_root=tmp_path / "cap",
                         adapter=Billed(R._env), prices=(1.0, 2.0), quiet=True, max_usd=0.9)
    preds = (capped["patient_diagnosis"] / "predictions.jsonl").read_text().splitlines()
    assert len(preds) < 6 and json.loads((capped["patient_diagnosis"] / "manifest.json").read_text())["stopped"]


def test_worker_envs_share_one_release_db(db):
    import threading
    envs = []
    t = threading.Thread(target=lambda: envs.append(R._env()))
    t.start(); t.join()
    envs.append(R._env())
    assert envs[0] is not envs[1] and envs[0].db is envs[1].db is R.shared_db()


def test_single_arm_spends_no_actions(tmp_path, db):
    out = R.run("glm-5.3-flash", "patient_diagnosis", n=2, arm="single", seed=6, workers=1, out_root=tmp_path,
                adapter=ScriptedAdapter(R._env), prices=(1.0, 2.0), quiet=True)
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    assert all(p["steps"] <= 1 and p["turns"] == 1 for p in preds), preds


def test_atypical_sample_is_paired_with_the_typical_sample(db):
    typical = {i["gt_id"] for i in R.sample_instances(db, "patient_diagnosis", "public", 120, 0)}
    atyp = R.sample_instances(db, "atypical_diagnosis", "public", 120, 0)
    assert len(atyp) >= 60 and all(i["gt"]["parent_gt_id"] in typical for i in atyp)


# ---------------------------------------------------------------------------
# the HTTP environment serves the same Stage-7 episodes (order matching and name credit reach /env)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["test_selection", "differential_diagnosis", "error_detection", "lab_triage",
                                  "atypical_diagnosis", "specialty_absent", "specialty_involved"])
def test_stage7_and_specialty_episodes_match_the_http_env(db, task):
    from eval.tests.test_local_env import _canon, _server
    client = _server()
    if client is None:
        pytest.skip("no environment server")
    env = LocalEnv(db=db)
    queries = ["stress test", "audiogram", "CBC", "ultrasound", "chest x-ray", "basic metabolic panel", "urinalysis"]
    for inst in db.instances(task, "public")[:3] + ([next(i for i in db.instances(task, "public") if i["gt_id"] == 77371)]
                                                   if task == "test_selection" else []):
        ro = env.reset(gt_id=inst["gt_id"])
        ep = client.post("/env/reset", json={"gt_id": inst["gt_id"]}).json()
        for key in ("task", "submit_tool", "instructions", "intro", "task_inputs", "tools"):
            assert getattr(ro, key) == ep[key], (inst["gt_id"], key)
        calls = [("view_results", {"patient_id": ro.patient_id, "result_type": "labs"})]
        if task == "test_selection":
            calls += [("order_test", {"name": q}) for q in queries]
        for name, args in calls:
            local = env.step(name, args)[0]
            remote = client.post("/env/step", json={"episode_id": ep["episode_id"], "name": name, "arguments": args}).json()["observation"]
            assert _canon(local) == _canon(remote), (inst["gt_id"], name, args)
        # a clinically named but miscoded answer: name credit must agree on both sides
        pred = env.oracle()
        if task in ("test_selection", "differential_diagnosis"):
            pred = json.loads(json.dumps(pred).replace('"icd10": "', '"icd10": "Z'))
        r_remote = client.post("/env/step", json={"episode_id": ep["episode_id"], "name": ro.submit_tool, "arguments": pred}).json()
        _, r_local, done, _ = env.step(ro.submit_tool, pred)
        assert done and r_local == pytest.approx(r_remote["reward"], abs=1e-9), (inst["gt_id"], r_local, r_remote)


# ---------------------------------------------------------------------------
# H2/H14: streamed OpenRouter calls are assembled correctly and a stalled provider fails fast
# ---------------------------------------------------------------------------

def _mock_openrouter(monkeypatch, body_iter):
    import httpx as _httpx
    import eval.adapters as A
    real = _httpx.Client

    def handler(request):
        return _httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body_iter())

    monkeypatch.setattr(A.httpx, "Client", lambda *a, **k: real(transport=_httpx.MockTransport(handler)))
    from eval.config import MODEL_REGISTRY
    return A.create_adapter(MODEL_REGISTRY["qwen3.5-9b"])        # OPENROUTER_API_KEY is set by the caller


def test_streamed_call_assembles_tool_calls_usage_and_provider(monkeypatch):
    def body():
        ev = [{"provider": "P1", "choices": [{"delta": {"reasoning": "think"}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "order_", "arguments": '{"na'}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "test", "arguments": 'me": "CBC"}'}}]}}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012}}]
        yield b": OPENROUTER PROCESSING\n\n"
        for e in ev:
            yield f"data: {json.dumps(e)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    a = _mock_openrouter(monkeypatch, body)
    r = a.call_multi_turn("s", [{"role": "user", "content": "u"}], tools=[{"type": "function", "function": {"name": "order_test"}}])
    assert r.tool_calls == [{"name": "order_test", "arguments": {"name": "CBC"}}] and r.raw_tool_calls[0]["id"] == "c1"
    assert r.cost_usd == pytest.approx(0.0012) and r.provider == "P1" and (r.input_tokens, r.output_tokens) == (10, 5)


def test_stalled_stream_fails_fast(monkeypatch):
    import time as _t
    import httpx as _httpx

    def body():
        for _ in range(100):                      # keep-alive comments only: connection alive, nothing generated
            yield b": OPENROUTER PROCESSING\n\n"
            _t.sleep(0.2)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("SH_FIRST_TOKEN_S", "1")
    a = _mock_openrouter(monkeypatch, body)
    t0 = _t.time()
    with pytest.raises(_httpx.ReadTimeout, match="first token"):
        a.call("s", "u")
    assert _t.time() - t0 < 6


def test_mid_stream_provider_error_is_retried_and_never_kills_the_run(monkeypatch, tmp_path, db):
    """Full run: a provider failure inside the 200 stream raised a non-retried HTTPStatusError, and reading its
    unread streaming body in the error handler crashed the whole run."""
    calls = {"n": 0}

    def body():
        calls["n"] += 1
        if calls["n"] == 1:
            yield b'data: {"error": {"code": 502, "message": "Upstream error from Venice: Stream interrupted"}}\n\n'
            return
        ev = {"choices": [{"delta": {"content": json.dumps({"active_diagnoses": []})}}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}}
        yield ("data: " + json.dumps(ev) + "\n\n").encode()
        yield b"data: [DONE]\n\n"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("EVAL_RETRY_BASE_DELAY", "0")
    a = _mock_openrouter(monkeypatch, body)
    r = a.call_multi_turn_with_retry("s", [{"role": "user", "content": "u"}])
    assert calls["n"] == 2 and "active_diagnoses" in r.text
    # an exception whose response is an unread stream is recorded as an episode error, not raised
    import httpx as _httpx

    class Boom:
        config = type("C", (), {"name": "b", "temperature": 0.0})()

        def call_multi_turn_with_retry(self, *a, **k):
            req = _httpx.Request("POST", "http://x")
            resp = _httpx.Response(200, stream=_httpx.ByteStream(b""), request=req)
            raise RuntimeError("exhausted") from _httpx.HTTPStatusError("s", request=req, response=resp)
    out = R.run("glm-5.3-flash", "patient_diagnosis", n=1, seed=8, workers=1, out_root=tmp_path, adapter=Boom(), prices=(1.0, 1.0), quiet=True)
    p = json.loads((out / "predictions.jsonl").read_text().splitlines()[0])
    assert p["reward"] == 0.0 and "exhausted" in p["error"]


def test_single_arm_tool_call_gets_one_format_retry_and_is_not_executed(tmp_path, db):
    """Full run: in the single arm Qwen called order_test (not offered) in 43% of test_selection episodes, which
    executed the order and then forced an empty submission (0)."""

    class OrdersFirst(ScriptedAdapter):
        def call_multi_turn_with_retry(self, system, messages, tools=None, max_retries=None):
            if len(messages) == 1:
                call = {"name": "order_test", "arguments": {"name": "CBC"}}
                raw = [{"id": "c0", "type": "function", "function": {"name": "order_test", "arguments": json.dumps(call["arguments"])}}]
                return ModelResponse(text="", input_tokens=10, output_tokens=5, latency_ms=1, raw_json=None, tool_calls=[call], raw_tool_calls=raw)
            args = R._env().oracle(); args.pop("tests_ordered", None)
            call = {"name": R._env().submit_tool, "arguments": args}
            raw = [{"id": "c1", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(args)}}]
            return ModelResponse(text="", input_tokens=10, output_tokens=5, latency_ms=1, raw_json=None, tool_calls=[call], raw_tool_calls=raw)

    out = R.run("glm-5.3-flash", "differential_diagnosis", n=2, arm="single", seed=9, workers=1, out_root=tmp_path,
                adapter=OrdersFirst(R._env), prices=(1.0, 1.0), quiet=True)
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    assert all(p["reward"] >= 1 - 1e-9 and p["format_retry"] == ["order_test"] and p["turns"] == 2 and p["steps"] <= 1
               and p["orders"] == 0 for p in preds), preds


def test_kitchen_sink_and_hedged_names_score_the_floor(db):
    """RL-readiness probe (2026-09-27): before the specificity cap, one 'name' made of the 3,000 most frequent label
    words scored differential_diagnosis 0.27 (floor 0.03) and patient_diagnosis 0.06 — an exploit a policy optimizing
    name credit would find. Both probes are floor policies now (eval/degenerate.py) and must stay at the floor."""
    for task in ("patient_diagnosis", "atypical_diagnosis", "differential_diagnosis", "test_selection"):
        insts = db.instances(task, "public")
        for pol in ("name_sink", "hedge_one_name"):
            preds = [D.POLICIES[task][pol](db, i) for i in insts]
            assert D.score(db, task, preds, insts)[D.PRIMARY_METRIC[task]] <= 0.01, (task, pol)
