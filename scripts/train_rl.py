"""GRPO training of the RL candidate on the environment (RL readiness C1/C5), with ART (LocalBackend: vLLM + LoRA).
Training prompts come from train-split patients outside the dev set; checkpoints are validated on the dev patients
(eval/rl_rollout.py `is_dev_patient`), never on public / heldout / private (audit/RL_SUCCESS_CRITERIA.md).

    python scripts/train_rl.py --run-name smoke --max-steps 3 [--units patient_diagnosis,...] [--prompts prompts.json]
        [--rollouts-per-group 8] [--groups-per-step 8] [--lr 1e-5] [--val-every 5] [--val-per-unit 20]
    python scripts/train_rl.py --dry-run --max-steps 2      # CPU: a scripted policy client, no ART / GPU

At start-up the frozen reward is checked (eval.reward_version); a drifted scorer stops the run. Each step: groups of
`rollouts-per-group` episodes on `groups-per-step` train prompts (the C4 filter when given), GRPO update, per-step
JSONL log (rewards, monitor aggregates, alerts, tokens, timings) under results/rl/<run>/; every `val-every` steps a
dev evaluation per unit (one episode each, evaluation sampling). Episodes that failed for infrastructure reasons
are exceptions, never trained on.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402
from eval import rl_monitor as M  # noqa: E402
from eval.rl_rollout import TRAIN_UNITS, TRANSFER_UNITS, RolloutConfig, dev_instances, rollout, start_up, train_prompts  # noqa: E402


def load_prompts(db, units: list[str], prompts_file: str | None, seed: int) -> list[dict]:
    """Train prompts: train-split instances of non-dev patients (P4), restricted to the C4 keep-list when given."""
    keep = set(json.loads(Path(prompts_file).read_text())["keep"]) if prompts_file else None
    return train_prompts(db, units, seed, keep)


def load_val(db, units: list[str], per_unit: int, seed: int) -> list[dict]:
    """Checkpoint selection on the train-dev patients (never trained on), never on public / heldout / private."""
    return dev_instances(db, units, per_unit, seed)


def numeric(d: dict) -> dict:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def art_configs(args) -> tuple[dict, dict]:
    """ART internal model config and OpenAI-server config for Qwen3.5 (P6). Plain dicts (ART's TypedDicts), so they are
    testable without ART installed.

    - `allow_unvalidated_arch`: ART 0.5.20's model registry validates Qwen3.5-4B / 27B, not 9B (same Qwen3.5 dense
      architecture); the 9B runs through the same path with validation relaxed. Fallbacks: --base-model
      Qwen/Qwen3.5-4B (validated; needs its own local baseline), or TRL's GRPO (see audit/RL_TRAINING_PLAN.md).
    - bf16 LoRA training (`load_in_4bit` off, `load_in_16bit` on): ART's default trains on 4-bit weights while vLLM
      samples from bf16 weights; the policy that is trained must be the policy that is served.
    - `max_seq_length` / `max_model_len`: 65,536 (Qwen3.5 defaults to 262,144, which would size training memory for
      contexts no episode reaches: the per-episode output cap is 32,768).
    - vision off (`limit_mm_per_prompt` 0): the checkpoint is a vision-language model; the task is text only.
    - vLLM server: ART defaults to the `hermes` tool parser; Qwen3.5 emits `<tool_call><function=...>` XML, parsed by
      `qwen3_coder`, with thinking split out by the `qwen3` reasoning parser (Qwen3.5 model card).
    """
    internal = {
        "allow_unvalidated_arch": True,
        "init_args": {"load_in_4bit": False, "load_in_16bit": True, "max_seq_length": args.max_seq_length,
                      "gpu_memory_utilization": args.gpu_memory_utilization},
        "engine_args": {"max_model_len": args.max_seq_length, "enable_sleep_mode": True,
                        "limit_mm_per_prompt": {"image": 0, "video": 0}},
    }
    server = {"server_args": {"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}}
    return internal, server


QWEN3_5_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj"]
"""ART's validated LoRA targets for the Qwen3.5 dense architecture (registry entry for 4B / 27B), set explicitly: for
the unvalidated 9B, ART would fall back to the generic list and leave the linear-attention (Gated DeltaNet)
projections `in_proj_qkv`, `in_proj_z`, `out_proj` — 3 of every 4 layers — untrained. Qwen3.5-9B has the same
architecture (`qwen3_5_text`, 32 layers, full attention every 4th)."""


def lora_config(args) -> dict:
    return {"rank": args.lora_rank, "alpha": args.lora_alpha, "target_modules": list(QWEN3_5_LORA_TARGETS)}


class ArtTrainer:
    """The ART side: model registration, the policy client, trajectory groups, the GRPO step."""

    def __init__(self, args):
        import art
        from art.local import LocalBackend
        self.art = art
        self.args = args
        self.internal, self.server = art_configs(args)
        self.backend = LocalBackend(path=args.art_path)
        self.model = art.TrainableModel(name=args.model_name, project=args.project, run_name=args.run_name,
                                        base_model=args.base_model, lora_config=lora_config(args),
                                        _internal_config=self.internal)

    async def setup(self):
        await self.model.register(self.backend, _openai_client_config=self.server)
        self.client = self.model.openai_client()

    def policy(self):
        return self.client, self.model.get_inference_name()

    async def group(self, inst: dict, k: int, cfg: RolloutConfig):
        art = self.art

        async def one():
            client, name = self.policy()
            res = await rollout(client, name, inst, cfg)
            if res.error:
                raise RuntimeError(res.error)
            return art.Trajectory(messages_and_choices=res.messages_and_choices, tools=res.tools, reward=res.reward,
                                  metrics={**numeric(res.record.get("signals") or {}), **numeric(res.record.get("metrics") or {})},
                                  metadata={"gt_id": inst["gt_id"], "task": inst["task"], "record": json.dumps(res.record, default=str)[:20000]})
        return art.TrajectoryGroup((one() for _ in range(k)), metadata={"gt_id": inst["gt_id"], "task": inst["task"]})

    async def gather(self, groups, max_exceptions: int):
        return await self.art.gather_trajectory_groups(groups, max_exceptions=max_exceptions)

    async def train(self, groups):
        return await self.backend.train(self.model, groups, learning_rate=self.args.lr)

    async def step(self) -> int:
        return await self.model.get_step()

    async def close(self):
        await self.backend.close()


class DryRunTrainer:
    """CPU stand-in with the same surface: a scripted policy (the label oracle's answer, or an empty one), no update."""

    def __init__(self, args):
        self.args, self._step = args, 0

    async def setup(self):
        from eval.tests.rl_fakes import ScriptedAsyncClient
        self.client = ScriptedAsyncClient()

    def policy(self):
        return self.client, "dry-run-policy"

    async def group(self, inst, k, cfg):
        async def one():
            res = await rollout(self.client, "dry-run-policy", inst, cfg)
            if res.error:
                raise RuntimeError(res.error)
            return res
        return [one() for _ in range(k)]

    async def gather(self, groups, max_exceptions: int):
        out = []
        for g in groups:
            g = await g
            out.append(await asyncio.gather(*g, return_exceptions=True))
        return out

    async def train(self, groups):
        self._step += 1
        return type("R", (), {"step": self._step, "metrics": {}})()

    async def step(self) -> int:
        return self._step

    async def close(self):
        pass


def _records(groups) -> list[dict]:
    out = []
    for g in groups:
        for t in getattr(g, "trajectories", g):
            if isinstance(t, BaseException):
                continue
            rec = getattr(t, "record", None)
            if rec is None:
                meta = getattr(t, "metadata", {}) or {}
                rec = json.loads(meta.get("record") or "{}")
            out.append(rec)
    return out


async def main_async(args) -> int:
    reward = start_up()                                      # frozen reward (A4/C1)
    db = D.ReleaseDB(shared=True)
    units = args.units.split(",")
    prompts = load_prompts(db, units, args.prompts, args.seed)
    val = load_val(db, units + list(TRANSFER_UNITS), args.val_per_unit, args.seed)
    out = Path(os.environ["SH_RL_RESULTS"]) if os.environ.get("SH_RL_RESULTS") else ROOT / "results" / "rl" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), **reward, "n_prompts": len(prompts), "n_val": len(val)}, indent=1, default=str))
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline and Path(args.baseline).exists() else {}
    trainer = (DryRunTrainer if args.dry_run else ArtTrainer)(args)
    await trainer.setup()
    cfg_train = RolloutConfig(per_turn_tokens=args.per_turn_tokens, per_episode_tokens=args.per_episode_tokens, logprobs=True)
    cfg_val = RolloutConfig(per_turn_tokens=args.per_turn_tokens, per_episode_tokens=args.per_episode_tokens, seed=args.seed)
    history: list[dict] = []
    start_step = await trainer.step()
    k, g = args.rollouts_per_group, args.groups_per_step
    log = (out / "steps.jsonl").open("a")
    try:
        for step in range(start_step, args.max_steps):
            batch = [prompts[(step * g + j) % len(prompts)] for j in range(g)]
            t0 = time.time()
            groups = await trainer.gather([trainer.group(inst, k, cfg_train) for inst in batch], max_exceptions=k * g)
            t_roll = time.time() - t0
            recs = _records(groups)
            t1 = time.time()
            result = await trainer.train(groups)
            t_train = time.time() - t1
            agg = M.aggregate(recs)
            entry = {"step": step + 1, "rollout_s": round(t_roll, 1), "train_s": round(t_train, 1), "episodes": len(recs),
                     "exceptions": k * g - len(recs), "output_tokens": sum(r.get("output_tokens", 0) for r in recs),
                     "train_reward": agg.get("reward_mean"), "train": agg,
                     "train_metrics": numeric(getattr(result, "metrics", {}) or {})}
            if args.val_every and ((step + 1) % args.val_every == 0 or step + 1 == args.max_steps):
                t2 = time.time()
                vgroups = await trainer.gather([trainer.group(inst, 1, cfg_val) for inst in val], max_exceptions=len(val))
                vrecs = _records(vgroups)
                vagg = M.aggregate(vrecs)
                by_unit = {}
                for inst, grp in zip(val, vgroups):
                    rs = _records([grp])
                    if rs:
                        by_unit.setdefault(inst["task"], []).append(float(rs[0].get("reward") or 0))
                entry.update(dev_reward=vagg.get("reward_mean"), dev=vagg, val_s=round(time.time() - t2, 1),
                             dev_by_unit={u: sum(v) / len(v) for u, v in by_unit.items()})
                history.append({"train_reward": entry["train_reward"], "dev_reward": entry["dev_reward"]})
                entry["alerts"] = M.alerts(vagg, baseline, history)
            log.write(json.dumps(entry, default=str) + "\n"); log.flush()
            print(f"step {step + 1}/{args.max_steps}: train reward {entry['train_reward']:.3f}, {entry['episodes']} episodes, "
                  f"rollout {t_roll:.0f}s, train {t_train:.0f}s" + (f", dev {entry['dev_reward']:.3f}" if "dev_reward" in entry else "")
                  + (f", ALERTS {entry['alerts']}" if entry.get("alerts") else ""), flush=True)
    finally:
        log.close()
        await trainer.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--project", default="clinical-envs")
    ap.add_argument("--model-name", default="qwen35-9b-clinical")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--units", default=",".join(TRAIN_UNITS))
    ap.add_argument("--prompts", default=None, help="C4 prompt filter (JSON with a 'keep' list of gt_ids)")
    ap.add_argument("--rollouts-per-group", type=int, default=8)
    ap.add_argument("--groups-per-step", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-per-unit", type=int, default=20)
    ap.add_argument("--per-turn-tokens", type=int, default=4096)
    ap.add_argument("--per-episode-tokens", type=int, default=32768)
    ap.add_argument("--baseline", default=None, help="dev-set monitor aggregate of the base model for alerts")
    ap.add_argument("--art-path", default=str(ROOT / ".art"))
    ap.add_argument("--max-seq-length", type=int, default=65536)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.dry_run and not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "disabled")
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
