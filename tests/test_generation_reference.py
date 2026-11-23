"""Check the three inference semantics against the training model on CPU."""

import pytest
import torch

from src.inference.generation_reference import (
    ReferenceDecoder,
    sample_depth,
    uniform_stream,
)
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM, compute_exit_distribution


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(grt=False, steps=8):
    torch.manual_seed(72)
    cfg = LoopLMConfig(
        vocab_size=41,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=96,
        max_recurrent_steps=steps,
    )
    model = LoopLM(cfg, use_grt=grt).eval()
    if grt:
        with torch.no_grad():
            model.grt_gate[-1].weight.normal_(0, 0.04)
            model.grt_gate[-1].bias.copy_(torch.linspace(-1, 1, 32))
    return model


@pytest.mark.parametrize("grt", [False, True])
def test_cached_states_match_full_prefix_forward(grt):
    model = tiny(grt)
    decoder = ReferenceDecoder(model, max_new_tokens=10)
    prompt = [3, 5, 8, 11]
    recurrence = "grt" if grt else "naive"
    with torch.inference_mode():
        decoder.reset(len(prompt))
        states = decoder.prefill(prompt)
        expected = model(torch.tensor([prompt]), recurrence=recurrence)
        for depth, state in enumerate(states):
            torch.testing.assert_close(
                decoder.vocabulary_logits(state),
                expected.logits[depth][0, -1],
                atol=3e-5,
                rtol=3e-5,
            )
            assert decoder._gate(state) == pytest.approx(
                expected.exit_lambdas[depth][0, -1].item(), abs=2e-6
            )
        for token in (7, 4, 9):
            position = len(prompt)
            prompt.append(token)
            expected = model(torch.tensor([prompt]), recurrence=recurrence)
            state = model.embed(torch.tensor([[token]]))
            for depth in range(8):
                state = decoder._loop(state, depth, position)
                torch.testing.assert_close(
                    decoder.vocabulary_logits(state),
                    expected.logits[depth][0, -1],
                    atol=3e-5,
                    rtol=3e-5,
                )
                assert decoder._gate(state) == pytest.approx(
                    expected.exit_lambdas[depth][0, -1].item(), abs=2e-6
                )
        if grt:
            torch.testing.assert_close(
                decoder.grt_anchors[:, : len(prompt)],
                model.embed(torch.tensor([prompt])),
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("grt", [False, True])
@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_coupled_modes_and_executed_flops(grt, seed):
    decoder = ReferenceDecoder(tiny(grt), max_new_tokens=12)
    results = [
        decoder.generate([1, 2, 3], mode=mode, seed=seed, sample_id=8, eos_token_id=-1)
        for mode in (0, 1, 2)
    ]
    assert results[0]["token_ids"] == results[1]["token_ids"]
    assert results[0]["exit_depths"] == results[1]["exit_depths"]
    for mode, result in enumerate(results):
        work = result["work"]
        assert work["decode_loops"] == (
            88 if mode != 2 else sum(result["exit_depths"][1:])
        )
        if mode == 1:
            assert work["foreground_decode_loops"] == sum(result["exit_depths"][1:])
            assert work["background_decode_loops"] == 88 - sum(
                result["exit_depths"][1:]
            )
        width, inter, layers, length = 32, 64, 2, 3
        controller = 10 * width**2 if grt else 0
        prefill = 8 * (
            layers
            * (
                8 * length * width**2
                + 6 * length * width * inter
                + 4 * length**2 * width
            )
            + length * controller
        )
        pairs = sum(
            (length + i) * (8 if mode != 2 else result["exit_depths"][i])
            for i in range(1, 12)
        )
        decode = (
            work["decode_loops"]
            * (layers * (8 * width**2 + 6 * width * inter) + controller)
            + 4 * pairs * layers * width
            + work["gate_calls"] * 2 * width
            + work["vocabulary_calls"] * 2 * width * 41
        )
        assert result["prefill_matmul_flops"] == prefill
        assert result["decode_matmul_flops"] == decode


def test_exit_head_reads_raw_hidden(monkeypatch):
    model = tiny()
    decoder = ReferenceDecoder(model, max_new_tokens=8)
    decoder.reset(2)
    hidden = torch.randn(1, 1, 32)
    expected = torch.sigmoid(model.exit_gate(hidden)).item()

    def forbidden(*args):
        raise AssertionError("Exit head must not invoke final norm")

    monkeypatch.setattr(model.final_norm, "forward", forbidden)
    assert decoder._gate(hidden) == pytest.approx(expected)


@pytest.mark.parametrize("grt", [False, True])
def test_kv_reuse_changes_only_current_position_and_same_layer(grt):
    decoder = ReferenceDecoder(tiny(grt), max_new_tokens=4)
    with torch.inference_mode():
        decoder.reset(3)
        decoder.prefill([2, 3, 4])
        before = [
            [(k[:, :, :3].clone(), v[:, :, :3].clone()) for k, v in zip(ks, vs)]
            for ks, vs in zip(decoder.keys, decoder.values)
        ]
        _, depth, _ = decoder.decode_token(9, 3, 2, 0.0, 0.3)
        assert depth == 1 and decoder.work.decode_loops == 1
        for depth in range(1, 8):
            for layer in range(decoder.L):
                torch.testing.assert_close(
                    decoder.keys[depth][layer][:, :, 3],
                    decoder.keys[0][layer][:, :, 3],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    decoder.values[depth][layer][:, :, 3],
                    decoder.values[0][layer][:, :, 3],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    decoder.keys[depth][layer][:, :, :3],
                    before[depth][layer][0],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    decoder.values[depth][layer][:, :, :3],
                    before[depth][layer][1],
                    atol=0,
                    rtol=0,
                )


@pytest.mark.parametrize("grt", [False, True])
def test_forced_last_depth_and_immediate_eos(grt):
    model = tiny(grt)
    with torch.no_grad():
        model.exit_gate.weight.zero_()
        model.exit_gate.bias.fill_(-1000)
    decoder = ReferenceDecoder(model, max_new_tokens=5)
    results = [
        decoder.generate([1, 2, 3], mode=m, seed=4, sample_id=9, eos_token_id=-1)
        for m in (0, 1, 2)
    ]
    assert results[0]["token_ids"] == results[1]["token_ids"] == results[2]["token_ids"]
    assert all(r["exit_depths"] == [8] * 5 for r in results)
    for mode in (0, 1, 2):
        result = decoder.generate(
            [1, 2, 3],
            mode=mode,
            seed=4,
            sample_id=9,
            eos_token_id=results[0]["token_ids"][0],
        )
        assert result["generated_tokens"] == 1 and result["eos"]
        assert result["work"]["decode_loops"] == 0


def test_depth_distribution_and_random_stream_keys():
    hazards = [0.2, 0.5, 0.25]
    expected = compute_exit_distribution([torch.tensor(v) for v in hazards + [0.0]])
    uniforms = [(i + 0.5) / 10000 for i in range(10000)]
    counts = [sum(sample_depth(hazards, u) == d for u in uniforms) for d in range(1, 5)]
    torch.testing.assert_close(torch.tensor(counts) / 10000, expected)
    assert (
        uniform_stream(1234, 7, "depth").random()
        == uniform_stream(1234, 7, "depth").random()
    )
    assert (
        uniform_stream(1234, 7, "depth").random()
        != uniform_stream(1234, 8, "depth").random()
    )
    assert (
        uniform_stream(1234, 7, "depth").random()
        != uniform_stream(1234, 7, "word").random()
    )
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            sample_depth([bad], 0.5)
