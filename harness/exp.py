"""Experiment sweep on the T4 (scratch repo only): solution-only runs with different settings."""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_task as rt  # noqa: E402
from common import read_json  # noqa: E402

out = (REPO / "runs_exp").resolve()
out.mkdir(parents=True, exist_ok=True)
task = rt.load_task()
args = argparse.Namespace(out=out, train_file=(REPO / "data" / "train.jsonl").resolve(),
                          eval_file=(REPO / "data" / "eval_public.jsonl").resolve(),
                          seed=task["training"]["seed"], timeout=900)
rt.warm_up(out)
BASE_ENV = dict(rt.OFFLINE_ENV)
plan = json.loads((HERE / "exp.json").read_text())
lines = []


def mean(xs):
    return round(statistics.mean(xs), 4) if xs else None


for cfg in plan:
    name = cfg["name"]
    env = dict(BASE_ENV, **cfg.get("env", {}))
    trace = out / f"{name}.trace.jsonl"
    trace.unlink(missing_ok=True)
    if cfg.get("trace"):
        env["SOUP_KERNELS_TRACE"] = str(trace)
    rt.OFFLINE_ENV = env
    res = rt.train(cfg.get("side", "solution"), name, args)
    m = read_json(out / name / "metrics.json", {})
    steps = m.get("step_s") or []
    own = sum((m.get("own_kernel_ms") or {}).values())
    loss = (res.get("log") or {}).get("loss") or []
    row = {"run": name, "ok": res["exit_code"] == 0, "wall": res["wall_s"],
           "peak": res["gpu_peak_mib"], "alloc": m.get("torch_max_allocated_mib"),
           "res": m.get("torch_max_reserved_mib"), "loop": m.get("train_loop_s"),
           "med": m.get("median_step_s"), "apply": m.get("apply_s"),
           "s0": steps[:3], "share": round(100 * own / 1000 / m["timed_loop_s"], 1)
           if m.get("timed_loop_s") else None,
           "loss": [round(x, 4) for x in loss[:3]] + [round(loss[-1], 4) if loss else None]}
    if trace.exists():
        ev = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
        fwd = [e for e in ev if e["ev"] == "fwd"][2:]
        bwd = [e for e in ev if e["ev"] == "bwd"][2:]
        t = {e["ev"]: e["t"] for e in ev if e["ev"] in ("apply_start", "apply_end",
                                                         "warmup_start", "warmup_end", "exit")}
        first = [e for e in ev if e["ev"] == "fwd"][:1]
        row["tr"] = {"tok": mean([e["tokens"] for e in fwd]),
                     "fcpu": mean([e["layers_cpu"] for e in fwd]),
                     "fhead": mean([e["head"] for e in fwd]),
                     "bcpu": mean([e["cpu"] for e in bwd]), "blag": mean([e["gpu_lag"] for e in bwd]),
                     "maxres_f": max([e["reserved"] for e in fwd] or [0]),
                     "maxres_b": max([e["reserved"] for e in bwd] or [0]),
                     "warm": round(t.get("warmup_end", 0) - t.get("warmup_start", 0), 2),
                     "fw_wait": [e["s"] for e in ev if e["ev"] == "first_wait"],
                     "to_first": round(first[0]["t"] - t["apply_end"], 2) if first and
                     "apply_end" in t else None,
                     "groups": [e["s"] for e in ev if e["ev"] == "compiled"]}
    lines.append(row)
    print("EXP " + json.dumps(row), flush=True)

print("\n===== EXPERIMENTS =====")
for row in lines:
    print(json.dumps(row))
