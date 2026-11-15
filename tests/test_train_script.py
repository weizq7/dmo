"""Tests for the training script logic (data pipeline + end-to-end smoke test).

These tests use synthetic data to avoid network downloads.
"""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

from src.model.config import LoopLMConfig
from src.training.trainer import Trainer, TrainerConfig, _infinite
from src.training.data import tokenize_and_chunk
from scripts.train import build_model_config, parse_args


# ── Data pipeline ─────────────────────────────────────────────────────────────

def test_tokenize_and_chunk_shape():
    """Chunks must be (N, seq_len+1) so trainer can split input/target."""

    class FakeTokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            # Return deterministic token ids based on text length
            return list(range(len(text) % 50 + 1))

    tokenizer = FakeTokenizer()
    texts = ["hello world " * 20] * 50
    seq_len = 16
    chunks = tokenize_and_chunk(texts, tokenizer, seq_len)

    assert chunks.ndim == 2
    assert chunks.shape[1] == seq_len + 1
    assert chunks.dtype == torch.long


def test_tokenize_and_chunk_no_overflow():
    """All token ids must be non-negative integers."""
    class FakeTokenizer:
        eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [2, 3, 4, 5] * 10

    chunks = tokenize_and_chunk(["dummy"] * 20, FakeTokenizer(), seq_len=8)
    assert (chunks >= 0).all()


# ── End-to-end smoke test ─────────────────────────────────────────────────────

@pytest.fixture
def tiny_cfg():
    return LoopLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        max_recurrent_steps=4,
    )


def make_synthetic_dataloader(vocab_size, seq_len=9, n=200, batch_size=4, seed=0):
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n, seq_len))
    return DataLoader(TensorDataset(tokens), batch_size=batch_size, shuffle=True, drop_last=True)


def test_smoke_train_100_steps(tiny_cfg, tmp_path):
    """Train 100 steps on synthetic data with the default mixture objective."""
    trainer_cfg = TrainerConfig(
        lr=3e-3,
        max_steps=100,
        num_recurrent_steps=4,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_wandb=False,
        device="cpu",
    )
    torch.manual_seed(42)
    trainer = Trainer(tiny_cfg, trainer_cfg)
    dl = make_synthetic_dataloader(tiny_cfg.vocab_size, seq_len=9)

    history = defaultdict(list)
    data_iter = _infinite(dl)

    while trainer.step < trainer_cfg.max_steps:
        batch = next(data_iter)
        diag = trainer.train_step(batch)
        history["loss"].append(diag["loss"].item())
        history["task_loss"].append(diag["task_loss"].item())
        history["entropy"].append(diag["entropy"].item())
        history["per_step_losses"].append(diag["per_step_losses"])

    # Default training has no entropy term, so total loss equals mixture NLL.
    torch.testing.assert_close(
        torch.tensor(history["loss"]), torch.tensor(history["task_loss"])
    )
    assert torch.isfinite(torch.tensor(history["entropy"])).all()

    # All per-step losses remain finite during optimization.
    for row in history["per_step_losses"]:
        for v in row:
            assert torch.isfinite(v), "Non-finite per-step loss"


def test_per_step_losses_are_distinct(tiny_cfg, tmp_path):
    """Per-step losses must differ from each other — the model must actually be
    doing different computations at each recurrent step.

    Note: verifying that *later* steps have *lower* loss than earlier ones requires
    real text data and sufficient training. That property is checked by
    scripts/train.py on wikitext/FineWeb-Edu, not here.
    """
    trainer_cfg = TrainerConfig(
        lr=3e-3,
        max_steps=50,
        beta_kl=0.1,
        num_recurrent_steps=4,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_wandb=False,
        device="cpu",
    )
    torch.manual_seed(0)
    trainer = Trainer(tiny_cfg, trainer_cfg)
    fixed_batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))

    diag = trainer.train_step(fixed_batch)
    losses = [v.item() for v in diag["per_step_losses"]]

    # Not all per-step losses should be identical
    assert len(set(round(l, 6) for l in losses)) > 1, (
        f"All per-step losses are identical: {losses}"
    )
    # All must be finite and positive
    for t, l in enumerate(losses):
        assert l > 0 and torch.isfinite(torch.tensor(l)), f"Invalid loss at step t={t+1}: {l}"


def test_grt_training_path(tiny_cfg, tmp_path):
    trainer_cfg = TrainerConfig(
        lr=3e-3,
        max_steps=1,
        num_recurrent_steps=3,
        recurrence="grt",
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_wandb=False,
        device="cpu",
    )
    torch.manual_seed(123)
    trainer = Trainer(
        tiny_cfg,
        trainer_cfg,
        model_kwargs={"use_grt": True, "grt_state_noise": 0.0, "grt_gate_noise": 0.0},
    )
    batch = torch.randint(0, tiny_cfg.vocab_size, (2, 9))
    diag = trainer.train_step(batch)
    assert diag["_stepped"]
    assert torch.isfinite(diag["loss"])
    assert trainer.raw_model.grt_input_proj.weight.grad is None


def test_public_scale_configs_are_available(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--model-config", "scale_135m", "--seq-len", "128"]
    )
    args = parse_args()
    cfg = build_model_config(args)
    assert cfg.hidden_size == 1024
    assert cfg.max_recurrent_steps == 8
    assert cfg.max_seq_len == 128

    monkeypatch.setattr(
        "sys.argv", ["train.py", "--model-config", "scale_370m"]
    )
    args = parse_args()
    cfg = build_model_config(args)
    assert cfg.hidden_size == 1280
    assert cfg.num_layers == 12
    assert cfg.max_recurrent_steps == 8
