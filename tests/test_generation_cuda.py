"""Explicitly opt-in GPU tests; never consume a GPU during ordinary CPU CI."""

import os

import pytest
import torch

from src.inference.cuda_generation import CudaDecoder
from src.model.rope import apply_rope
from tests.test_generation_reference import tiny

pytestmark = pytest.mark.skipif(
    os.environ.get("DMO_TEST_CUDA") != "1",
    reason="Set DMO_TEST_CUDA=1 inside an idle GPU allocation",
)


@pytest.fixture(autouse=True)
def cuda_environment(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    monkeypatch.setenv("CUDA_DEVICE_MAX_COPY_CONNECTIONS", "32")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with torch.inference_mode():
        yield


def assert_dependencies(trace, prompt_length, output_length, steps):
    for position in range(prompt_length, prompt_length + output_length - 1):
        for depth in range(steps):
            spans = [
                (
                    int(trace[position, depth, kind, 0]),
                    int(trace[position, depth, kind, 1]),
                )
                for kind in (0, 1)
            ]
            executed = [(begin, end) for begin, end in spans if begin or end]
            assert len(executed) == 1
            begin, end = executed[0]
            assert 0 < begin <= end
            if position > prompt_length:
                assert begin >= int(trace[position - 1, depth, :, 1].max())
            if depth:
                assert begin >= int(trace[position, depth - 1, :, 1].max())


@pytest.mark.parametrize("grt", [False, True])
@pytest.mark.parametrize("bias", [None, 1000.0, -1000.0])
def test_fused_generation_preserves_unfused_outputs_caches_and_work(
    monkeypatch, grt, bias
):
    import src.inference.cuda_generation as implementation

    model = tiny(grt).cuda()
    if bias is not None:
        model.exit_gate.weight.zero_()
        model.exit_gate.bias.fill_(bias)
    reference = CudaDecoder(model, max_new_tokens=24, capacity=64)
    actual = CudaDecoder(model, max_new_tokens=24, capacity=64)
    try:
        # Capture an unfused arithmetic reference, retaining the same attention
        # interface and scheduler so kernel differences can be checked exactly.
        reference.decode_model = model
        with monkeypatch.context() as patch:
            patch.setattr(
                implementation, "exact_rope", lambda x, c, s, lib: apply_rope(x, c, s)
            )
            reference.prepare(4)
        actual.prepare(4)
        assert len(actual.owned_streams) == len(set(actual.owned_streams)) == 7
        assert actual.fg.cuda_stream not in actual.owned_streams
        assert len(actual.captures) == 17
        paired = {}
        for seed in (1234, 2345, 3456):
            for mode in (0, 1, 2):
                kwargs = dict(mode=mode, seed=seed, sample_id=11, eos_token_id=-1)
                expected = reference.generate([1, 2, 3, 4], **kwargs)
                result = actual.generate([1, 2, 3, 4], **kwargs)
                for key in ("token_ids", "exit_depths", "work"):
                    assert result[key] == expected[key]
                torch.testing.assert_close(
                    reference.key_storage, actual.key_storage, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    reference.value_storage, actual.value_storage, atol=0, rtol=0
                )
                if mode == 1:
                    assert_dependencies(actual.trace_gpu.cpu(), 4, 24, 8)
                    assert result["submitted_background_graphs"] == sum(
                        8 - d for d in result["exit_depths"][1:]
                    )
                if bias is not None:
                    assert result["exit_depths"] == [1 if bias > 0 else 8] * 24
                paired[mode] = result
            assert paired[0]["token_ids"] == paired[1]["token_ids"]
            assert paired[0]["exit_depths"] == paired[1]["exit_depths"]
        eos = actual.generate(
            [1, 2, 3, 4],
            mode=1,
            seed=3456,
            sample_id=11,
            eos_token_id=paired[1]["token_ids"][0],
        )
        assert eos["generated_tokens"] == 1 and eos["work"]["decode_loops"] == 0
        assert eos["submitted_background_graphs"] == 0
    finally:
        reference.close()
        actual.close()
    assert not actual.workers and not actual.owned_streams and not actual.fg_graphs


def test_no_per_loop_host_decision_and_only_required_depth_submissions(monkeypatch):
    decoder = CudaDecoder(tiny().cuda(), max_new_tokens=8, capacity=64)
    try:
        decoder.reset(4)
        decoder.prefill([1, 2, 3, 4])
        decoder.synchronize()
        reads, submissions = [], []
        original = torch.Tensor.tolist

        def track_read(tensor):
            if tensor.is_cuda:
                reads.append(tuple(tensor.shape))
            return original(tensor)

        native = decoder.scheduler.ce_launch
        graphs = [graph for _, _, _, graph in decoder.workers]

        def track_launch(graph, stream):
            if graph in graphs:
                submissions.append(graph)
            return native(graph, stream)

        def forbidden(*a, **k):
            raise AssertionError("Unexpected scalar CPU readback")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "tolist", track_read)
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(decoder.scheduler, "ce_launch", track_launch)
            _, depth, _ = decoder.decode_token(7, 4, 1, 0.6, 0.35)
            decoder.synchronize()
        assert reads == [(3,)]
        assert submissions == graphs[depth - 1 :]
    finally:
        decoder.close()


@pytest.mark.parametrize("steps", [1, 2, 8])
def test_reset_rebuild_and_depth_boundaries(steps):
    decoder = CudaDecoder(tiny(steps=steps).cuda(), max_new_tokens=8)
    try:
        for length in (3, 4, 3):
            prompt = list(range(1, length + 1))
            a = decoder.generate(
                prompt, mode=0, seed=1234, sample_id=2, eos_token_id=-1
            )
            b = decoder.generate(
                prompt, mode=1, seed=1234, sample_id=2, eos_token_id=-1
            )
            assert (
                a["token_ids"] == b["token_ids"]
                and a["exit_depths"] == b["exit_depths"]
            )
            assert len(decoder.workers) == steps - 1
    finally:
        decoder.close()
