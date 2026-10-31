"""Optional timing trace for our own analysis: off unless SOUP_KERNELS_TRACE names a file.

When on, it appends one JSON line per event (model-side timings, allocator state). It adds one
synchronization per step (after backward), so traced runs are not the measured ones.
"""

import json
import os
import time

PATH = os.environ.get("SOUP_KERNELS_TRACE", "")


def on():
    return bool(PATH)


def log(**fields):
    if not PATH:
        return
    fields["t"] = round(time.time(), 4)
    with open(PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields) + "\n")


def memory():
    import torch

    mib = 2 ** 20
    return {"alloc": round(torch.cuda.memory_allocated() / mib),
            "reserved": round(torch.cuda.memory_reserved() / mib),
            "max_alloc": round(torch.cuda.max_memory_allocated() / mib),
            "max_reserved": round(torch.cuda.max_memory_reserved() / mib)}


def install_backward_timer():
    """Times the CPU side of loss.backward() and how far the GPU is behind when it returns."""
    import torch

    original = torch.autograd.backward

    def backward(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        log(ev="bwd", cpu=round(t1 - t0, 4), gpu_lag=round(t2 - t1, 4), **memory())
        return result

    torch.autograd.backward = backward
