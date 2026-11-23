"""Batch-one stochastic generation with device-side exit control.

All modes share BF16 linear operands, fused RMSNorm/RoPE, and CUDA graphs.
Mode 1 uses one independent background stream per remaining depth; each
worker follows token order and can pick up newly published work.
"""

from __future__ import annotations
from dataclasses import asdict
import time
import torch
from src.inference.generation_reference import (
    ReferenceDecoder,
    Work,
    sample_depth,
    uniform_stream,
)
from src.inference.generation_kernels import (
    load_control_library,
    load_depth_library,
    pointers,
    check_result,
    check_call,
    make_decode_view,
    exact_rope,
    load_rope_library,
    require_cuda_environment,
)


class CudaDecoder(ReferenceDecoder):
    backend = "cuda_conditional_graph_dedicated_depth"

    def __init__(self, model, *, max_new_tokens=512, num_steps=None, capacity=None):
        require_cuda_environment()
        super().__init__(model, max_new_tokens=max_new_tokens, num_steps=num_steps)
        if not self.cuda:
            raise ValueError("CudaDecoder requires a CUDA model")
        if any(not hasattr(block, "pre_attn_norm") for block in model.layers):
            raise ValueError(
                "Generation requires sandwich-normalized transformer blocks"
            )
        self.lib = load_control_library()
        self.scheduler = load_depth_library()
        self.decode_model = make_decode_view(model)
        self.rope_library = load_rope_library()
        self.fixed_capacity = capacity
        self.capacity = 0
        self.fg_graphs = {}
        self.captures = []
        self.saved = []
        self.workers = []
        self.owned_streams = []
        self.spec_foreground = 0
        self.submitted_workers = 0
        self.asynchronous = False

    def close(self):
        self.synchronize()
        for _, _, _, graph in self.workers:
            self.scheduler.ce_destroy(graph)
        self.workers.clear()
        for mode, graph in self.fg_graphs.items():
            (self.scheduler if mode == 1 else self.lib).ce_destroy(graph)
        self.fg_graphs.clear()
        self.spec_foreground = 0
        self.captures.clear()
        self.saved.clear()
        for handle in self.owned_streams:
            check_call(self.scheduler, self.scheduler.sd_destroy_stream(handle))
        self.owned_streams.clear()
        self.capacity = 0

    def _capture(self, fn):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.inference_mode(), self.autocast():
            result = fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            torch.cuda.graph(graph, stream=stream),
            torch.inference_mode(),
            self.autocast(),
        ):
            result = fn()
        self.captures.append(graph)
        if result is not None:
            self.saved.append(result)
        return graph, result

    def _loop_body(self, h, input_ids, depth, with_gate):
        model = self.decode_model
        pos = input_ids[1:2]
        if self.use_grt:
            if depth == 0:
                self.grt_anchors.index_copy_(1, pos, h)
            previous = h.clone()
            anchor = self.grt_anchors.index_select(1, pos)
            state = model.grt_input_proj(
                torch.cat([previous.float(), anchor.float()], dim=-1).to(h.dtype)
            )
        else:
            state = h
        cos = self.cos.index_select(0, pos)
        sin = self.sin.index_select(0, pos)
        lengths = torch.cat((self.zero_length, (pos + 1).to(torch.int32)))
        for layer, block in enumerate(model.layers):
            attn = block.attn
            x = block.pre_attn_norm(state)
            q = attn.q_proj(x).view(1, 1, self.H, self.D).transpose(1, 2)
            k = attn.k_proj(x).view(1, 1, self.H, self.D).transpose(1, 2)
            v = attn.v_proj(x).view(1, 1, self.H, self.D).transpose(1, 2)
            q = exact_rope(q, cos, sin, self.rope_library)
            k = exact_rope(k, cos, sin, self.rope_library)
            self.keys[depth][layer].index_copy_(2, pos, k.to(self.key_storage.dtype))
            self.values[depth][layer].index_copy_(
                2, pos, v.to(self.value_storage.dtype)
            )
            out = (
                torch.ops.aten._flash_attention_forward(
                    q.to(torch.bfloat16)[0].transpose(0, 1),
                    self.keys[depth][layer][0].transpose(0, 1),
                    self.values[depth][layer][0].transpose(0, 1),
                    self.query_lengths,
                    lengths,
                    1,
                    self.capacity,
                    0.0,
                    False,
                    False,
                )[0]
                .contiguous()
                .view(1, 1, -1)
            )
            state = state + block.post_attn_norm(attn.o_proj(out))
            state = state + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(state)))
        if self.use_grt:
            gate_input = torch.cat(
                [
                    model.grt_state_norm(previous.float()),
                    model.grt_anchor_norm(anchor.float()),
                ],
                dim=-1,
            ).to(h.dtype)
            copy_gate = torch.sigmoid(
                model.grt_gate(gate_input) / model.grt_gate_temperature
            ).float()
            h.copy_(
                (copy_gate * previous.float() + (1.0 - copy_gate) * state.float()).to(
                    h.dtype
                )
            )
        else:
            h.copy_(state)
        if with_gate:
            return (
                torch.sigmoid(model.exit_gate(h if self.use_grt else state))
                .float()
                .reshape(())
            )
        return None

    def _readout(self):
        logits = (
            self.model.lm_head(self.model.final_norm(self.selected)).float().reshape(-1)
        )
        probs = logits.softmax(-1)
        cdf = probs.double().cumsum(-1)
        target = cdf[-1] * self.uniforms[1]
        index = torch.searchsorted(cdf, target, right=True).clamp_max(probs.numel() - 1)
        return torch.where(
            torch.isfinite(cdf[-1]) & (cdf[-1] > 0), index, torch.full_like(index, -1)
        )

    def _ptrs(self):
        return pointers(
            [
                self.input,
                self.uniforms,
                self.survival,
                self.control,
                self.ready_gpu,
                self.stats_gpu,
                self.hidden,
                self.selected,
                self.history,
                self.depth_history,
                self.output,
                self.word,
                self.trace_gpu,
            ]
        )

    def _state_ptrs(self):
        return pointers(
            [self.published, self.scheduler_error, self.completed, self.continuation]
        )

    @torch.inference_mode()
    def prepare(self, prompt_length):
        required = prompt_length + self.max_new_tokens
        capacity = self.fixed_capacity or required
        if capacity < required or capacity > self.model.config.max_seq_len:
            raise ValueError("Invalid cache capacity")
        if self.capacity == capacity:
            return
        if self.capacity:
            self.close()
        if not 1 <= self.T <= 32:
            raise ValueError("Expected 1..32 recurrent depths")
        self.capacity = capacity
        width = self.model.config.hidden_size
        self.key_storage = torch.zeros(
            (self.T, self.L, 1, self.H, capacity, self.D),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.value_storage = torch.zeros_like(self.key_storage)
        self.keys = [
            [self.key_storage[d, l] for l in range(self.L)] for d in range(self.T)
        ]
        self.values = [
            [self.value_storage[d, l] for l in range(self.L)] for d in range(self.T)
        ]
        self.cos, self.sin = self.model.rope.get_cos_sin(capacity, self.device)
        self.zero_length = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.query_lengths = torch.tensor([0, 1], device=self.device, dtype=torch.int32)
        self.input = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.host_input = torch.empty(2, dtype=torch.int64, pin_memory=True)
        self.host_uniforms = torch.empty(2, dtype=torch.float64, pin_memory=True)
        self.uniforms = torch.zeros(2, device=self.device, dtype=torch.float64)
        self.survival = torch.ones(1, device=self.device, dtype=torch.float64)
        self.control = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.output = torch.zeros(3, device=self.device, dtype=torch.int64)
        self.ready_gpu = torch.full(
            (self.T,), -1, device=self.device, dtype=torch.int32
        )
        self.stats_gpu = torch.zeros(3, device=self.device, dtype=torch.int64)
        self.hidden = torch.zeros(1, 1, width, device=self.device, dtype=torch.float32)
        self.selected = torch.zeros_like(self.hidden)
        self.history = torch.zeros(
            capacity, width, device=self.device, dtype=torch.float32
        )
        self.depth_history = torch.zeros(
            capacity, device=self.device, dtype=torch.int64
        )
        self.trace_gpu = torch.zeros(
            capacity, self.T, 2, 2, device=self.device, dtype=torch.int64
        )
        self.ready = [None] * self.T
        self.events = []
        self.work = Work()
        self.published = torch.full((1,), -1, device=self.device, dtype=torch.int32)
        self.scheduler_error = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.completed = torch.zeros(capacity, device=self.device, dtype=torch.int32)
        self.continuation = torch.zeros_like(self.history)

        if self.use_grt:
            self.grt_anchors = torch.zeros(
                1, capacity, width, device=self.device, dtype=torch.float32
            )

        embed, _ = self._capture(
            lambda: self.hidden.copy_(self.model.embed(self.input[:1].view(1, 1)))
        )
        loops = []
        gates = []
        for depth in range(self.T):
            graph, gate = self._capture(
                lambda d=depth: self._loop_body(
                    self.hidden, self.input, d, d < self.T - 1
                )
            )
            loops.append(graph.raw_cuda_graph())
            gates.append(gate.data_ptr() if gate is not None else 0)
        readout, self.word = self._capture(self._readout)
        for mode in (0, 2):
            self.fg_graphs[mode] = check_result(
                self.lib,
                self.lib.ce_make_fg(
                    mode,
                    self.T,
                    width,
                    capacity,
                    self._ptrs(),
                    pointers(loops),
                    pointers(gates),
                    embed.raw_cuda_graph(),
                    readout.raw_cuda_graph(),
                ),
            )
        self.spec_foreground = check_result(
            self.scheduler,
            self.scheduler.sd_make_foreground(
                self.T,
                width,
                capacity,
                self._ptrs(),
                self._state_ptrs(),
                pointers(loops),
                pointers(gates),
                embed.raw_cuda_graph(),
                readout.raw_cuda_graph(),
            ),
        )
        self.fg_graphs[1] = self.spec_foreground
        for depth in range(1, self.T):
            hidden = torch.zeros_like(self.hidden)
            ids = torch.zeros_like(self.input)
            body, _ = self._capture(
                lambda d=depth, h=hidden, i=ids: self._loop_body(h, i, d, False)
            )
            ptrs = list(self._ptrs())
            ptrs[0] = ids.data_ptr()
            ptrs[6] = hidden.data_ptr()
            graph = check_result(
                self.scheduler,
                self.scheduler.sd_make_worker(
                    self.T,
                    width,
                    capacity,
                    depth,
                    pointers(ptrs),
                    self._state_ptrs(),
                    body.raw_cuda_graph(),
                ),
            )
            handle = check_result(self.scheduler, self.scheduler.sd_create_stream())
            self.owned_streams.append(handle)
            self.workers.append(
                (
                    torch.cuda.ExternalStream(handle, device=self.device),
                    hidden,
                    ids,
                    graph,
                )
            )
        if (
            len(set(self.owned_streams)) != self.T - 1
            or self.fg.cuda_stream in self.owned_streams
        ):
            raise RuntimeError("Invalid dedicated stream ownership")
        self.synchronize()

    @torch.inference_mode()
    def reset(self, prompt_length):
        self.prepare(prompt_length)
        self.synchronize()
        self.work = Work()
        self.events = []
        self.ready = [None] * self.T
        self.key_storage.zero_()
        self.value_storage.zero_()
        self.stats_gpu.zero_()
        self.trace_gpu.zero_()
        self.control.zero_()
        self.ready_gpu.fill_(prompt_length - 1)
        self.published.fill_(prompt_length - 1)
        self.scheduler_error.zero_()
        self.completed.zero_()
        self.continuation.zero_()
        self.submitted_workers = 0
        current = torch.cuda.current_stream(self.device)
        self.fg.wait_stream(current)
        for stream, _, _, _ in self.workers:
            stream.wait_stream(current)

    def submit_remaining_depths(self, exit_depth):
        if not 1 <= exit_depth <= self.T:
            raise ValueError("Invalid sampled exit depth")
        for stream, _, _, graph in self.workers[exit_depth - 1 :]:
            check_call(
                self.scheduler, self.scheduler.ce_launch(graph, stream.cuda_stream)
            )
        self.submitted_workers += self.T - exit_depth

    @torch.inference_mode()
    def decode_token(self, token, pos, mode, depth_u, word_u, on_emit=None):
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1 or 2")
        if not 0 <= depth_u < 1 or not 0 <= word_u < 1:
            raise ValueError("Invalid sampling uniform")
        self.host_input[0] = token
        self.host_input[1] = pos
        self.host_uniforms[0] = depth_u
        self.host_uniforms[1] = word_u
        library = self.scheduler if mode == 1 else self.lib
        with self.context(self.fg):
            self.input.copy_(self.host_input, non_blocking=True)
            self.uniforms.copy_(self.host_uniforms, non_blocking=True)
            check_call(
                library, library.ce_launch(self.fg_graphs[mode], self.fg.cuda_stream)
            )
            # One host result read per emitted token; no per-loop CPU decision.
            sampled, depth, error = self.output.tolist()
            emitted = time.perf_counter()
            if error:
                raise RuntimeError(f"GPU exit control error={error}, position={pos}")
            if on_emit is not None:
                on_emit(sampled, depth, emitted)
            if mode == 2:
                check_call(
                    self.lib,
                    self.lib.ce_copy_kv(
                        self.fg_graphs[mode],
                        self.fg.cuda_stream,
                        self.key_storage.data_ptr(),
                        self.value_storage.data_ptr(),
                        self.L,
                        self.H,
                        self.D,
                    ),
                )
        if mode == 1:
            self.submit_remaining_depths(depth)
        return sampled, depth, emitted

    @torch.inference_mode()
    def generate(self, prompt, *, mode, seed, sample_id, eos_token_id, on_emit=None):
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1 or 2")
        if not prompt or self.max_new_tokens < 1:
            raise ValueError("Empty generation")
        depth_rng = uniform_stream(seed, sample_id, "depth")
        word_rng = uniform_stream(seed, sample_id, "word")
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

        def emit(token, depth, when):
            ids.append(token)
            depths.append(depth)
            emits.append(when)
            if on_emit is not None:
                on_emit(token, depth, when)

        del states
        while len(ids) < self.max_new_tokens and ids[-1] != eos_token_id:
            self.decode_token(
                ids[-1],
                len(prompt) + len(ids) - 1,
                mode,
                depth_rng.random(),
                word_rng.random(),
                emit,
            )
        self.synchronize()
        finished = time.perf_counter()
        fg, bg, copies = self.stats_gpu.tolist()
        expected_fg = (len(ids) - 1) * self.T if mode == 0 else sum(depths[1:])
        expected_bg = (len(ids) - 1) * self.T - expected_fg if mode == 1 else 0
        if (fg, bg) != (expected_fg, expected_bg):
            raise RuntimeError("GPU executed-loop count mismatch")
        expected_copies = sum(self.T - d for d in depths[1:]) if mode == 2 else 0
        if copies != expected_copies:
            raise RuntimeError("GPU copied-depth count mismatch")
        self.work.decode_loops = fg + bg
        self.work.foreground_decode_loops = fg
        self.work.background_decode_loops = bg
        self.work.vocabulary_calls = len(ids)
        self.work.gate_calls = (
            self.T
            - 1
            + sum(self.T - 1 if mode == 0 else min(d, self.T - 1) for d in depths[1:])
        )
        self.work.kv_copy_bytes = (
            copies * self.L * 2 * self.H * self.D * self.key_storage.element_size()
        )
        W = self.model.config.hidden_size
        I = self.model.config.intermediate_size
        attention_pairs = sum(
            (len(prompt) + i) * (self.T if mode in (0, 1) else depths[i])
            for i in range(1, len(ids))
        )
        self.work.matmul_flops = (
            prefill_flops
            + (fg + bg) * self.L * (8 * W * W + 6 * W * I)
            + 4 * attention_pairs * self.L * W
            + self.work.vocabulary_calls * 2 * W * self.model.config.vocab_size
            + self.work.gate_calls * 2 * W
        )
        result = dict(
            token_ids=ids,
            exit_depths=depths,
            generated_tokens=len(ids),
            eos=ids[-1] == eos_token_id,
            prefill_seconds=prefill_done - started,
            ttft_seconds=emits[0] - started,
            emission_seconds=emits[-1] - started,
            decode_visible_seconds=emits[-1] - emits[0],
            drain_seconds=finished - emits[-1],
            service_seconds=finished - started,
            prefill_matmul_flops=prefill_flops,
            decode_matmul_flops=self.work.matmul_flops - prefill_flops,
            inter_token_seconds=[b - a for a, b in zip(emits, emits[1:])],
            work=asdict(self.work),
            backend=self.backend,
            attention_capacity=self.capacity,
            attention_backend="aten_flash_attention_varlen_device_lengths",
            decode_host_result_transfers=len(ids) - 1,
            per_loop_host_decision_transfers=0,
        )
        error = int(self.scheduler_error.cpu().tolist()[0])
        if error:
            raise RuntimeError(f"Background depth scheduler error={error}")
        if self.use_grt:
            extra = (
                result["work"]["decode_loops"] * 10 * self.model.config.hidden_size**2
            )
            self.work.matmul_flops += extra
            result["work"]["matmul_flops"] += extra
            result["decode_matmul_flops"] += extra
            result["grt_controller_matmul_flops"] = extra
        result.update(
            recurrence="grt" if self.use_grt else "vanilla",
            submitted_background_graphs=self.submitted_workers,
            scheduler="owned_depth_streams",
        )
        return result
