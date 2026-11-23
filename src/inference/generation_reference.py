"""Eager correctness reference and common full-depth prefill for generation.

CPU timings are not benchmark timings. CUDA benchmarks use CudaDecoder.
The fixed random-stream domain is retained to reproduce the published cohort.
"""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import hashlib
import math
import random
import time

import torch
import torch.nn.functional as F

from src.model.rope import apply_rope


def sample_depth(lambdas, uniform):
    """Inverse CDF with final-depth residual mass; inputs exclude last hazard."""
    if not 0 <= uniform < 1:
        raise ValueError("Depth uniform must lie in [0, 1)")
    survival = 1.0
    for d, lam in enumerate(lambdas, 1):
        if not math.isfinite(float(lam)) or not 0 <= float(lam) <= 1:
            raise ValueError("Non-finite or invalid exit hazard")
        survival *= 1.0 - float(lam)
        if uniform < 1.0 - survival:
            return d
    return len(lambdas) + 1


def uniform_stream(seed, sample_id, kind):
    digest = hashlib.sha256(
        f"looplm-generation-v1:{seed}:{sample_id}:{kind}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "little"))


@dataclass
class Work:
    prefill_layer_tokens: int = 0
    decode_loops: int = 0
    foreground_decode_loops: int = 0
    background_decode_loops: int = 0
    gate_calls: int = 0
    vocabulary_calls: int = 0
    kv_copy_bytes: int = 0
    matmul_flops: int = 0


class ReferenceDecoder:
    def __init__(self, model, *, max_new_tokens=512, num_steps=None):
        if model.training:
            raise ValueError("Generation requires model.eval()")
        for flag in ("use_moe_recurrence", "use_q_act"):
            if getattr(model, flag, False):
                raise ValueError(f"Unsupported recurrence option: {flag}")
        self.model = model
        self.use_grt = bool(model.use_grt)
        self.device = next(model.parameters()).device
        self.cuda = self.device.type == "cuda"
        self.T = num_steps or model.config.max_recurrent_steps
        self.L = len(model.layers)
        self.H = model.config.num_heads
        self.D = model.config.hidden_size // self.H
        self.max_new_tokens = max_new_tokens
        self.asynchronous = False
        self.fg = (
            torch.cuda.Stream(device=self.device, priority=-1) if self.cuda else None
        )
        self.bg = (
            torch.cuda.Stream(device=self.device, priority=0) if self.cuda else None
        )
        if self.cuda:
            self.fg.wait_stream(torch.cuda.current_stream(self.device))
            self.bg.wait_stream(torch.cuda.current_stream(self.device))

    def context(self, stream):
        return torch.cuda.stream(stream) if self.cuda else nullcontext()

    def autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16) if self.cuda else nullcontext()
        )

    def synchronize(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _event(self, stream):
        if not self.cuda or not self.asynchronous:
            return None
        e = torch.cuda.Event()
        e.record(stream)
        self.events.append(e)
        return e

    def _wait(self, stream, event):
        if self.cuda and event is not None:
            stream.wait_event(event)

    def reset(self, prompt_length):
        self.synchronize()
        self.work = Work()
        self.events = []
        capacity = prompt_length + self.max_new_tokens
        if capacity > self.model.config.max_seq_len:
            raise ValueError("Prompt plus generation exceeds model context")
        dtype = torch.bfloat16 if self.cuda else next(self.model.parameters()).dtype
        # Separate allocations retain ordinary contiguous [B,H,S,D] views.
        self.keys = [
            [
                torch.empty(
                    1, self.H, capacity, self.D, device=self.device, dtype=dtype
                )
                for _ in range(self.L)
            ]
            for _ in range(self.T)
        ]
        self.values = [[torch.empty_like(k) for k in row] for row in self.keys]
        self.cos, self.sin = self.model.rope.get_cos_sin(capacity, self.device)
        self.ready = [None] * self.T
        if self.use_grt:
            self.grt_anchors = torch.zeros(
                1,
                capacity,
                self.model.config.hidden_size,
                device=self.device,
                dtype=torch.float32,
            )
        if self.cuda:
            self.fg.wait_stream(torch.cuda.current_stream(self.device))
            self.bg.wait_stream(torch.cuda.current_stream(self.device))

    def _block(self, h, depth, layer_index, pos, prefill):
        block = self.model.layers[layer_index]
        attn = block.attn
        x = block.pre_attn_norm(h)
        S = x.shape[1]
        q = attn.q_proj(x).view(1, S, self.H, self.D).transpose(1, 2)
        k = attn.k_proj(x).view(1, S, self.H, self.D).transpose(1, 2)
        v = attn.v_proj(x).view(1, S, self.H, self.D).transpose(1, 2)
        cos, sin = self.cos[pos : pos + S], self.sin[pos : pos + S]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        self.keys[depth][layer_index][:, :, pos : pos + S].copy_(k)
        self.values[depth][layer_index][:, :, pos : pos + S].copy_(v)
        # Prefill uses the original tensors (same as model.forward). Decode uses
        # persistent per-loop, per-layer caches, with no repeated concatenation.
        if prefill:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(
                q,
                self.keys[depth][layer_index][:, :, : pos + S],
                self.values[depth][layer_index][:, :, : pos + S],
                is_causal=False,
            )
        out = out.transpose(1, 2).contiguous().view(1, S, -1)
        h = h + block.post_attn_norm(attn.o_proj(out))
        h = h + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(h)))
        width = self.model.config.hidden_size
        inter = self.model.config.intermediate_size
        key_len = S if prefill else pos + S
        # Dense GEMM-equivalent FLOPs: QKVO + three SwiGLU projections + QK/AV.
        # Excludes elementwise/norm/softmax/RoPE, memory movement and kernel overhead.
        self.work.matmul_flops += (
            8 * S * width * width + 6 * S * width * inter + 4 * S * key_len * width
        )
        return h

    def _transformer_loop(
        self, h, depth, pos, *, prefill=False, stream=None, background=False
    ):
        self._wait(stream, self.ready[depth])
        for layer in range(self.L):
            h = self._block(h, depth, layer, pos, prefill)
        self.ready[depth] = self._event(stream)
        if prefill:
            self.work.prefill_layer_tokens += h.shape[1] * self.L
        else:
            self.work.decode_loops += 1
            if background:
                self.work.background_decode_loops += 1
            else:
                self.work.foreground_decode_loops += 1
        return h

    def _loop(self, h, depth, pos, *, prefill=False, stream=None, background=False):
        if not self.use_grt:
            return self._transformer_loop(
                h, depth, pos, prefill=prefill, stream=stream, background=background
            )
        length = h.shape[1]
        if depth == 0:
            self.grt_anchors[:, pos : pos + length].copy_(h)
        previous = h.float()
        anchor = self.grt_anchors[:, pos : pos + length]
        projected = self.model.grt_input_proj(
            torch.cat([previous, anchor], dim=-1).to(h.dtype)
        )
        proposal = self._transformer_loop(
            projected, depth, pos, prefill=prefill, stream=stream, background=background
        )
        gate_input = torch.cat(
            [self.model.grt_state_norm(previous), self.model.grt_anchor_norm(anchor)],
            dim=-1,
        ).to(h.dtype)
        gate = torch.sigmoid(
            self.model.grt_gate(gate_input) / self.model.grt_gate_temperature
        ).float()
        self.work.matmul_flops += length * 10 * self.model.config.hidden_size**2
        return (gate * previous + (1.0 - gate) * proposal.float()).to(h.dtype)

    def _gate_tensor(self, h):
        x = h  # The exit head always reads the raw hidden state.
        self.work.gate_calls += 1
        self.work.matmul_flops += 2 * self.model.config.hidden_size
        # Match the checkpoint's sigmoid precision, then accumulate survival in f64.
        return torch.sigmoid(self.model.exit_gate(x)).float().reshape(())

    def _gate(self, h):
        value = self._gate_tensor(h).item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite exit hazard")
        return value

    def vocabulary_logits(self, h):
        self.work.vocabulary_calls += 1
        self.work.matmul_flops += (
            2 * self.model.config.hidden_size * self.model.config.vocab_size
        )
        return self.model.lm_head(self.model.final_norm(h)).float().reshape(-1)

    def _sample_token(self, h, uniform):
        logits = self.vocabulary_logits(h)
        probs = logits.softmax(-1)
        cdf = probs.double().cumsum(-1)
        target = cdf[-1] * uniform
        index = torch.searchsorted(cdf, target, right=True).clamp_max(probs.numel() - 1)
        index = torch.where(
            torch.isfinite(cdf[-1]) & (cdf[-1] > 0), index, torch.full_like(index, -1)
        )
        value = int(index.item())
        if value < 0:
            raise RuntimeError("Non-finite or empty vocabulary distribution")
        return value

    def prefill(self, prompt):
        with self.context(self.fg), self.autocast():
            ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            h = self.model.embed(ids)
            last = []
            for d in range(self.T):
                h = self._loop(h, d, 0, prefill=True, stream=self.fg)
                last.append(h[:, -1:, :].clone())
        return last

    def _fill_tail(self, h, exit_depth, pos):
        if exit_depth == self.T:
            return
        self._wait(self.bg, self.ready[exit_depth - 1])
        if self.cuda:
            h.record_stream(self.bg)
        with self.context(self.bg), self.autocast():
            for d in range(exit_depth, self.T):
                h = self._loop(h, d, pos, stream=self.bg, background=True)

    def _copy_tail(self, exit_depth, pos):
        for d in range(exit_depth, self.T):
            for l in range(self.L):
                self.keys[d][l][:, :, pos : pos + 1].copy_(
                    self.keys[exit_depth - 1][l][:, :, pos : pos + 1]
                )
                self.values[d][l][:, :, pos : pos + 1].copy_(
                    self.values[exit_depth - 1][l][:, :, pos : pos + 1]
                )
                self.work.kv_copy_bytes += (
                    2 * self.H * self.D * self.keys[d][l].element_size()
                )
            self.ready[d] = self._event(self.fg)

    def decode_token(self, token, pos, mode, depth_u, word_u, on_emit=None):
        """Return sampled token/depth, then queue/copy missing cache entries."""
        with self.context(self.fg), self.autocast():
            h = self.model.embed(torch.tensor([[token]], device=self.device))
            lambdas = []
            states = []
            survival = 1.0
            depth = self.T
            for d in range(self.T):
                h = self._loop(h, d, pos, stream=self.fg)
                if mode == 0:
                    states.append(h)
                if d < self.T - 1:
                    lam = self._gate_tensor(h) if mode == 0 else self._gate(h)
                    lambdas.append(lam)
                    if mode != 0:
                        survival *= 1.0 - lam
                if mode != 0 and (d == self.T - 1 or depth_u < 1.0 - survival):
                    depth = d + 1
                    break
            if mode == 0:
                lambdas = torch.stack(lambdas).tolist() if lambdas else []
                depth = sample_depth(lambdas, depth_u)
                h = states[depth - 1]
            sampled = self._sample_token(h, word_u)
            emitted = time.perf_counter()
            if on_emit is not None:
                on_emit(sampled, depth, emitted)
            if mode == 2:
                self._copy_tail(depth, pos)
        if mode == 1:
            self._fill_tail(h, depth, pos)
        return sampled, depth, emitted

    @torch.inference_mode()
    def generate(self, prompt, *, mode, seed, sample_id, eos_token_id, on_emit=None):
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1 or 2")
        if not prompt or self.max_new_tokens < 1:
            raise ValueError("Empty prompt or nonpositive output length")
        depth_rng = uniform_stream(seed, sample_id, "depth")
        word_rng = uniform_stream(seed, sample_id, "word")
        self.asynchronous = mode == 1
        self.reset(len(prompt))
        self.synchronize()
        started = time.perf_counter()
        states = self.prefill(prompt)
        self.synchronize()
        prefill_done = time.perf_counter()
        prefill_flops = self.work.matmul_flops
        with self.context(self.fg), self.autocast():
            gates = [self._gate_tensor(h) for h in states[:-1]]
            lambdas = torch.stack(gates).tolist() if gates else []
            depth = sample_depth(lambdas, depth_rng.random())
            token = self._sample_token(states[depth - 1], word_rng.random())
        ids = [token]
        depths = [depth]
        emits = [time.perf_counter()]
        if on_emit is not None:
            on_emit(token, depth, emits[0])

        def emit_callback(token, depth, emitted):
            ids.append(token)
            depths.append(depth)
            emits.append(emitted)
            if on_emit is not None:
                on_emit(token, depth, emitted)

        del states
        while len(ids) < self.max_new_tokens and ids[-1] != eos_token_id:
            self.decode_token(
                ids[-1],
                len(prompt) + len(ids) - 1,
                mode,
                depth_rng.random(),
                word_rng.random(),
                emit_callback,
            )
        self.synchronize()
        finished = time.perf_counter()
        result = {
            "token_ids": ids,
            "exit_depths": depths,
            "generated_tokens": len(ids),
            "eos": ids[-1] == eos_token_id,
            "prefill_seconds": prefill_done - started,
            "ttft_seconds": emits[0] - started,
            "emission_seconds": emits[-1] - started,
            "decode_visible_seconds": emits[-1] - emits[0],
            "drain_seconds": finished - emits[-1],
            "service_seconds": finished - started,
            "prefill_matmul_flops": prefill_flops,
            "decode_matmul_flops": self.work.matmul_flops - prefill_flops,
            "inter_token_seconds": [b - a for a, b in zip(emits, emits[1:])],
            "work": asdict(self.work),
            "backend": "cuda_streams" if self.cuda else "cpu_reference",
        }
        return result
