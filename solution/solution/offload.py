"""Asynchronous offload of saved activations to pinned host memory, with prefetch in backward.

Without gradient checkpointing (the task forbids switching Soup's) every layer keeps its saved
activations on the GPU until backward reaches it. The biggest ones are ours to manage: the
fp16 gate|up output of the MLP (17,920 values per token), the fp32 residual inputs of the two
RMSNorms, the fp16 LoRA inputs and what SDPA keeps for its backward (q, k, v, output). They are
written to pinned host buffers on a side CUDA stream while the GPU keeps computing (the copy
engine runs in parallel with the SMs), and read back about one layer ahead of the backward pass,
so neither direction stalls the compute stream when PCIe keeps up.

Correctness rules (the usual stream-ordering ones):
- the D2H copy waits for the producer (side.wait_stream(main)) and the source block is kept
  alive for the side stream (record_stream) until the copy is done;
- the H2D copy waits for the main stream (the destination block comes from the main stream's
  pool) and the main stream waits on the copy's event before the first use;
- host buffers are reused across steps by save order; a slot is rewritten in the next forward
  only after the previous backward consumed it (both are ordered after the main stream).
Backward runs the layers in reverse, so the tensors are prefetched in reverse save order. A
tensor that was not prefetched is fetched on demand (correct, just not overlapped).

Memory rule: a block whose copy is still queued cannot be reused by the allocator, and the CPU
runs many layers ahead of the GPU, so without a bound every copied-but-not-yet-done tensor of
the forward pass stays reserved. The forward therefore waits (on the CPU) whenever more than
`SOUP_KERNELS_OFFLOAD_INFLIGHT_MB` of copies are queued; that keeps the GPU footprint near what
is really alive while the copies still overlap with about two layers of compute.
"""

import os
from collections import deque

import torch


def _flag(name, default):
    return os.environ.get(name, default) == "1"


def dense(t):
    """True if `t` covers one gap-free block of memory (any order of dimensions)."""
    expected = 1
    for stride, size in sorted((s, n) for s, n in zip(t.stride(), t.shape) if n != 1):
        if stride != expected:
            return False
        expected *= size
    return True


class _Handle:
    __slots__ = ("owner", "index", "host", "nbytes", "shape", "stride", "dtype", "gpu",
                 "event", "step", "loaded")

    def __init__(self, owner, index, host, nbytes, shape, stride, dtype, step):
        self.owner, self.index, self.host, self.nbytes = owner, index, host, nbytes
        self.shape, self.stride, self.dtype, self.step = shape, stride, dtype, step
        self.gpu = self.event = None
        self.loaded = False


class Offloader:
    def __init__(self):
        self.stream = None
        self.pool = []
        self.handles = []
        self.step = 0
        self.active = True  # off for the last layers: their backward comes first anyway
        self.inflight = deque()  # (event, bytes) of device->host copies maybe not done yet
        self.inflight_bytes = 0
        self.layer_bytes = 0  # bytes saved by the current layer (forward)
        self.window = 0  # prefetch window in backward: the most bytes one layer saved

    def begin_step(self):
        self.handles = []
        self.step += 1
        self.layer_bytes = 0
        self.window = 0

    def begin_layer(self):
        self.window = max(self.window, self.layer_bytes)
        self.layer_bytes = 0

    def save(self, t):
        """Start copying `t` (a dense CUDA tensor, any strides) to host; returns a handle."""
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
        flat = t.as_strided((t.numel(),), (1,), t.storage_offset())
        main = torch.cuda.current_stream(t.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            host[:nbytes].copy_(flat.view(torch.uint8), non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        t.record_stream(self.stream)
        handle = _Handle(self, index, host, nbytes, tuple(t.shape), tuple(t.stride()), t.dtype,
                         self.step)
        self.handles.append(handle)
        self.layer_bytes += nbytes
        limit = inflight_limit()
        if limit > 0:
            self.inflight.append((done, nbytes))
            self.inflight_bytes += nbytes
            while self.inflight and self.inflight_bytes > limit:
                event, size = self.inflight.popleft()
                event.synchronize()
                self.inflight_bytes -= size
        return handle

    def _fetch(self, h, device):
        if h.gpu is not None or h.loaded:
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
        out = h.gpu.view(h.dtype).as_strided(h.shape, h.stride)
        h.gpu = h.event = None
        h.loaded = True
        if h.step == self.step:
            # prefetch the tensors saved before this one, about one layer's worth of bytes
            window = max(self.window, self.layer_bytes) or (1 << 62)
            ahead = 0
            j = h.index - 1
            while j >= 0 and ahead < window:
                prev = self.handles[j]
                ahead += prev.nbytes
                self._fetch(prev, device)
                j -= 1
        return out


OFFLOADER = Offloader()


class Saved:
    """A tensor kept for backward either as is or offloaded to host."""

    __slots__ = ("tensor", "handle", "device")

    def __init__(self, t, offload):
        self.device = t.device
        if offload and t.is_cuda and t.numel():
            if not dense(t):
                t = t.contiguous()
            self.tensor, self.handle = None, OFFLOADER.save(t)
        else:
            self.tensor, self.handle = t, None

    def get(self):
        if self.handle is not None:
            self.tensor = OFFLOADER.load(self.handle, self.device)
            self.handle = None
        return self.tensor


def pack_saved(t):
    """saved_tensors_hooks pack hook: offload big CUDA tensors an op saves for backward."""
    if isinstance(t, torch.Tensor) and t.is_cuda and OFFLOADER.active and \
            t.numel() * t.element_size() >= (1 << 16) and dense(t):
        return Saved(t, True)
    return t


def unpack_saved(obj):
    return obj.get() if isinstance(obj, Saved) else obj


def offload_mlp_flag():
    return _flag("SOUP_KERNELS_OFFLOAD_MLP", "1")


def offload_resid_flag():
    return _flag("SOUP_KERNELS_OFFLOAD_RESID", "1")


def offload_attn_flag():
    return _flag("SOUP_KERNELS_OFFLOAD_ATTN", "1")


def offload_mlp():
    return OFFLOADER.active and offload_mlp_flag()


def offload_resid():
    return OFFLOADER.active and offload_resid_flag()


def offload_attn():
    return OFFLOADER.active and offload_attn_flag()


def inflight_limit():
    return int(float(os.environ.get("SOUP_KERNELS_OFFLOAD_INFLIGHT_MB", "256")) * 2**20)


def keep_last_layers():
    return int(os.environ.get("SOUP_KERNELS_OFFLOAD_KEEP_LAST", "1"))
