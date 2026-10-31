"""Asynchronous offload of saved activations to pinned host memory, with prefetch in backward.

Without gradient checkpointing (the task forbids switching Soup's) every layer keeps its saved
activations on the GPU until backward reaches it. The biggest ones are ours to manage: the
fp16 gate|up output of the MLP (17,920 values per token), the fp32 residual inputs of the two
RMSNorms and the fp16 LoRA inputs. They are written to pinned host buffers on a side CUDA stream
while the GPU keeps computing (the copy engine runs in parallel with the SMs), and read back one
or two tensors ahead of the backward pass, so neither direction stalls the compute stream when
PCIe keeps up.

Correctness rules (the usual stream-ordering ones):
- the D2H copy waits for the producer (side.wait_stream(main)) and the source block is kept
  alive for the side stream (record_stream) until the copy is done;
- the H2D copy waits for the main stream (the destination block comes from the main stream's
  pool) and the main stream waits on the copy's event before the first use;
- host buffers are reused across steps by save order; a slot is rewritten in the next forward
  only after the previous backward consumed it (both are ordered after the main stream).
Backward runs the layers in reverse, so the tensors are prefetched in reverse save order. A
tensor that was not prefetched is fetched on demand (correct, just not overlapped).
"""

import os

import torch


def _flag(name, default):
    return os.environ.get(name, default) == "1"


class _Handle:
    __slots__ = ("owner", "index", "host", "nbytes", "shape", "dtype", "gpu", "event", "step")

    def __init__(self, owner, index, host, nbytes, shape, dtype, step):
        self.owner, self.index, self.host, self.nbytes = owner, index, host, nbytes
        self.shape, self.dtype, self.step = shape, dtype, step
        self.gpu = self.event = None


class Offloader:
    PREFETCH = 5  # one layer's worth of saved tensors: the next MLP is fetched a layer early

    def __init__(self):
        self.stream = None
        self.pool = []
        self.handles = []
        self.step = 0
        self.active = True  # off for the last layers: their backward comes first anyway

    def begin_step(self):
        self.handles = []
        self.step += 1
        per_layer = (4 if offload_resid_flag() else 0) + (1 if offload_mlp_flag() else 0)
        self.PREFETCH = max(1, per_layer)

    def save(self, t):
        """Start copying `t` (contiguous CUDA tensor) to host; returns a handle for load()."""
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=t.device)
        index = len(self.handles)
        nbytes = t.numel() * t.element_size()
        if index < len(self.pool) and self.pool[index].numel() >= nbytes:
            host = self.pool[index]
        else:
            # batches vary in length: round up so a slot is (re)pinned only a few times
            size = 1 << max(20, (nbytes - 1).bit_length())
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            if index < len(self.pool):
                self.pool[index] = host
            else:
                self.pool.append(host)
        main = torch.cuda.current_stream(t.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            host[:nbytes].copy_(t.reshape(-1).view(torch.uint8), non_blocking=True)
        t.record_stream(self.stream)
        handle = _Handle(self, index, host, nbytes, tuple(t.shape), t.dtype, self.step)
        self.handles.append(handle)
        return handle

    def _fetch(self, h, device):
        if h.gpu is not None:
            return
        main = torch.cuda.current_stream(device)
        buf = torch.empty(h.nbytes, dtype=torch.uint8, device=device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            buf.copy_(h.host[:h.nbytes], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        h.gpu, h.event = buf, event

    def load(self, h, device):
        self._fetch(h, device)
        torch.cuda.current_stream(device).wait_event(h.event)
        out = h.gpu.view(h.dtype).view(h.shape)
        h.gpu = h.event = None
        if h.step == self.step:
            for back in range(1, self.PREFETCH + 1):
                j = h.index - back
                if j >= 0 and j < len(self.handles):
                    self._fetch(self.handles[j], device)
        return out


OFFLOADER = Offloader()


class Saved:
    """A tensor kept for backward either as is or offloaded to host."""

    __slots__ = ("tensor", "handle", "device")

    def __init__(self, t, offload):
        self.device = t.device
        if offload and t.is_cuda and t.numel():
            self.tensor, self.handle = None, OFFLOADER.save(t.contiguous())
        else:
            self.tensor, self.handle = t, None

    def get(self):
        if self.handle is None:
            return self.tensor
        t = OFFLOADER.load(self.handle, self.device)
        self.handle = None
        return t


def offload_mlp_flag():
    return _flag("SOUP_KERNELS_OFFLOAD_MLP", "1")


def offload_resid_flag():
    return _flag("SOUP_KERNELS_OFFLOAD_RESID", "1")


def offload_mlp():
    return OFFLOADER.active and offload_mlp_flag()


def offload_resid():
    return OFFLOADER.active and offload_resid_flag()


def keep_last_layers():
    return int(os.environ.get("SOUP_KERNELS_OFFLOAD_KEEP_LAST", "1"))
