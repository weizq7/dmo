import torch
import pytest
from src.model.config import LoopLMConfig
from src.model.looplm import (
    LoopLM,
    LoopLMOutput,
    compute_exit_distribution,
    model_kwargs_from_checkpoint,
)


@pytest.fixture
def cfg():
    # Tiny config to keep tests fast
    return LoopLMConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=32,
        max_recurrent_steps=4,
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return LoopLM(cfg)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (2, 8))  # (B=2, S=8)


# ── Output structure ─────────────────────────────────────────────────────────

def test_returns_correct_number_of_steps(model, cfg, input_ids):
    out = model(input_ids)
    assert len(out.logits) == cfg.max_recurrent_steps
    assert len(out.exit_lambdas) == cfg.max_recurrent_steps


def test_logits_shape(model, cfg, input_ids):
    out = model(input_ids)
    B, S = input_ids.shape
    for t, logits in enumerate(out.logits):
        assert logits.shape == (B, S, cfg.vocab_size), f"Wrong shape at step {t}"


def test_exit_lambdas_shape(model, cfg, input_ids):
    out = model(input_ids)
    B, S = input_ids.shape
    for t, lam in enumerate(out.exit_lambdas):
        assert lam.shape == (B, S), f"Wrong lambda shape at step {t}"


def test_exit_lambdas_in_unit_interval(model, input_ids):
    out = model(input_ids)
    for lam in out.exit_lambdas:
        assert lam.min() >= 0.0 and lam.max() <= 1.0


def test_gradient_checkpointing_runs_only_in_training(model, input_ids, monkeypatch):
    import src.model.looplm as looplm_module

    original_checkpoint = looplm_module.checkpoint
    checkpoint_calls = []

    def recording_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(1)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(looplm_module, "checkpoint", recording_checkpoint)
    model.gradient_checkpointing = True
    model.train()
    output = model(input_ids, num_steps=2)

    assert len(checkpoint_calls) == 2 * model.config.num_layers
    sum(logits.sum() for logits in output.logits).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    checkpoint_calls.clear()
    model.eval()
    with torch.no_grad():
        model(input_ids, num_steps=2)
    assert checkpoint_calls == []


# ── Exit distribution ────────────────────────────────────────────────────────

def test_exit_distribution_sums_to_one(model, input_ids):
    out = model(input_ids)
    probs = compute_exit_distribution(out.exit_lambdas)  # (T, B, S)
    total = probs.sum(dim=0)  # (B, S)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-5, rtol=1e-5)


def test_exit_distribution_non_negative(model, input_ids):
    out = model(input_ids)
    probs = compute_exit_distribution(out.exit_lambdas)
    assert probs.min() >= 0.0


def test_exit_distribution_single_step():
    """With T=1, all probability mass must go to the last step."""
    lam = [torch.full((1, 4), 0.8)]  # T=1, λ=0.8 (doesn't matter for T=1)
    probs = compute_exit_distribution(lam)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-6, rtol=1e-6)


# ── Custom num_steps ─────────────────────────────────────────────────────────

def test_custom_num_steps(model, cfg, input_ids):
    for T in [1, 2, cfg.max_recurrent_steps]:
        out = model(input_ids, num_steps=T)
        assert len(out.logits) == T
        assert len(out.exit_lambdas) == T


def test_t1_produces_valid_logits(model, cfg, input_ids):
    """With a single recurrent step the model should behave like a standard LM."""
    out = model(input_ids, num_steps=1)
    assert len(out.logits) == 1
    logits = out.logits[0]
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
    assert torch.isfinite(logits).all()


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradients_flow_to_all_params(model, input_ids):
    out = model(input_ids)
    # Simple loss: sum of all logits across all steps
    loss = sum(l.sum() for l in out.logits) + sum(g.sum() for g in out.exit_lambdas)
    loss.backward()
    for name, p in model.named_parameters():
        # lm_head.weight is tied to embed.weight — skip the alias
        if name == "lm_head.weight":
            continue
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.all(p.grad == 0), f"Zero grad for {name}"


# ── Weight sharing ────────────────────────────────────────────────────────────

def test_lm_head_weight_tied_to_embedding(model):
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_layer_weights_are_independent(model):
    """Each layer in the stack should be a distinct module with its own weights."""
    if len(model.layers) < 2:
        pytest.skip("Need at least 2 layers")
    w0 = model.layers[0].attn.q_proj.weight
    w1 = model.layers[1].attn.q_proj.weight
    assert w0.data_ptr() != w1.data_ptr()


# ── Recurrent steps improve loss ──────────────────────────────────────────────

def test_later_steps_have_different_logits(model, input_ids):
    """Hidden state must change between recurrent steps."""
    out = model(input_ids, num_steps=2)
    assert not torch.allclose(out.logits[0], out.logits[1])


def test_grt_recurrence_shapes_and_gradients(cfg, input_ids):
    torch.manual_seed(7)
    model = LoopLM(
        cfg,
        use_grt=True,
        grt_state_noise=0.0,
        grt_gate_noise=0.0,
    )
    out = model(input_ids, recurrence="grt")
    assert len(out.logits) == cfg.max_recurrent_steps
    assert len(out.exit_lambdas) == cfg.max_recurrent_steps
    assert all(x.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size) for x in out.logits)

    loss = sum(x.square().mean() for x in out.logits)
    loss.backward()
    assert model.grt_input_proj.weight.grad is not None
    assert model.grt_gate[0].weight.grad is not None
    assert model.grt_gate[2].weight.grad is not None


def test_grt_requires_explicit_model_support(model, input_ids):
    with pytest.raises(ValueError, match="use_grt=True"):
        model(input_ids, recurrence="grt")


def test_model_kwargs_from_checkpoint_detects_standard_variants(cfg):
    vanilla = LoopLM(cfg)
    grt = LoopLM(cfg, use_grt=True, grt_state_noise=0.0, grt_gate_noise=0.0)
    q_act = LoopLM(cfg, use_q_act=True)

    assert model_kwargs_from_checkpoint({"model_state_dict": vanilla.state_dict()}) == {}
    assert model_kwargs_from_checkpoint({"model_state_dict": grt.state_dict()})["use_grt"]
    assert model_kwargs_from_checkpoint({"model_state_dict": q_act.state_dict()})["use_q_act"]


def test_grt_checkpoint_can_be_restored_with_inferred_kwargs(cfg, input_ids):
    source = LoopLM(cfg, use_grt=True, grt_state_noise=0.0, grt_gate_noise=0.0)
    checkpoint = {"model_state_dict": source.state_dict()}
    restored = LoopLM(cfg, **model_kwargs_from_checkpoint(checkpoint))
    restored.load_state_dict(checkpoint["model_state_dict"])
    source.eval()
    restored.eval()
    source_out = source(input_ids, recurrence="grt")
    restored_out = restored(input_ids, recurrence="grt")
    for source_logits, restored_logits in zip(source_out.logits, restored_out.logits):
        torch.testing.assert_close(source_logits, restored_logits)
