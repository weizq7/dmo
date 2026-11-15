import os
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import LoopLMConfig
from src.training.trainer import Trainer, TrainerConfig, _warmup_cosine_lr
from src.training.objectives import compute_looplm_loss


@pytest.fixture
def tiny_cfg():
    return LoopLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        max_recurrent_steps=2,
    )


@pytest.fixture
def tiny_trainer_cfg(tmp_path):
    return TrainerConfig(
        lr=1e-3,
        max_steps=10,
        log_every=5,
        save_every=999,  # don't auto-save during short tests
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=False,
        device="cpu",
    )


def make_dataloader(vocab_size: int, seq_len: int = 9, n_batches: int = 20, B: int = 4, seed: int = 0):
    """Returns a DataLoader of random token batches of shape (B, seq_len)."""
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_batches * B, seq_len))
    return DataLoader(TensorDataset(tokens), batch_size=B, shuffle=False)


# ── Initialization ────────────────────────────────────────────────────────────

def test_trainer_creates_model(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    from src.model.looplm import LoopLM
    assert isinstance(trainer.model, LoopLM)


def test_trainer_step_starts_at_zero(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    assert trainer.step == 0


def test_warmup_cosine_schedule_reaches_floor():
    assert _warmup_cosine_lr(6e-4, 6e-5, 1, 22000, 2000) == pytest.approx(3e-7)
    assert _warmup_cosine_lr(6e-4, 6e-5, 2000, 22000, 2000) == pytest.approx(6e-4)
    assert _warmup_cosine_lr(6e-4, 6e-5, 22000, 22000, 2000) == pytest.approx(6e-5)
    assert _warmup_cosine_lr(6e-4, 6e-5, 25000, 22000, 2000) == pytest.approx(6e-5)


def test_constant_schedule_keeps_optimizer_learning_rate(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.train_step(torch.randint(0, tiny_cfg.vocab_size, (4, 9)))
    assert trainer.optimizer.param_groups[0]["lr"] == initial_lr


def test_warmup_cosine_schedule_updates_optimizer_learning_rate(tiny_cfg, tmp_path):
    config = TrainerConfig(
        lr=6e-4,
        lr_schedule="warmup_cosine",
        warmup_steps=2,
        min_lr=6e-5,
        lr_total_steps=4,
        max_steps=4,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "schedule"),
        use_wandb=False,
        device="cpu",
    )
    trainer = Trainer(tiny_cfg, config)
    batch = torch.randint(0, tiny_cfg.vocab_size, (2, 9))
    observed = []
    for _ in range(4):
        diag = trainer.train_step(batch)
        observed.append(diag["lr"].item())
    assert observed[0] == pytest.approx(3e-4)
    assert observed[1] == pytest.approx(6e-4)
    assert observed[-1] == pytest.approx(6e-5)


# ── Single train step ─────────────────────────────────────────────────────────

def test_train_step_returns_diagnostics(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))  # (B, S+1)
    diag = trainer.train_step(batch)
    for key in ("loss", "task_loss", "entropy", "avg_exit_step", "grad_norm"):
        assert key in diag


def test_train_step_increments_step(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    trainer.train_step(batch)
    assert trainer.step == 1


def test_train_step_produces_finite_loss(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    diag = trainer.train_step(batch)
    assert torch.isfinite(diag["loss"])


def test_validation_caps_batches_weights_tokens_and_restores_model_mode(
    tiny_cfg, tmp_path
):
    trainer_cfg = TrainerConfig(
        max_steps=1,
        num_recurrent_steps=2,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "validation"),
        use_wandb=False,
        device="cpu",
        use_amp=False,
    )
    trainer = Trainer(tiny_cfg, trainer_cfg)
    tokens = torch.randint(0, tiny_cfg.vocab_size, (5, 9))
    dataloader = DataLoader(TensorDataset(tokens), batch_size=2, shuffle=False)

    trainer.raw_model.train()
    metrics = trainer.evaluate_validation(dataloader, max_batches=2)
    assert trainer.raw_model.training
    assert metrics["val/num_batches"] == 2
    assert metrics["val/num_tokens"] == 4 * (tokens.shape[1] - 1)

    # The two capped batches must produce the same token-mean loss as evaluating
    # their concatenation in one batch, despite the loader's smaller final batch
    # being outside the cap.
    trainer.raw_model.eval()
    with torch.no_grad():
        output = trainer.raw_model(
            tokens[:4, :-1], num_steps=trainer.num_recurrent_steps
        )
        _, reference = compute_looplm_loss(
            output.logits,
            output.exit_lambdas,
            tokens[:4, 1:],
            beta=trainer.config.beta_kl,
            objective="mixture_nll",
        )
    torch.testing.assert_close(
        torch.tensor(metrics["val/mixture_nll"]), reference["mixture_nll"]
    )

    trainer.raw_model.eval()
    trainer.evaluate_validation(dataloader, max_batches=1)
    assert not trainer.raw_model.training


def test_empty_local_validation_shard_keeps_ddp_reduction_shape(
    tiny_cfg, tiny_trainer_cfg, monkeypatch
):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    monkeypatch.setattr("src.training.trainer._is_distributed", lambda: True)
    reduced_shapes = []

    def fake_all_reduce(tensor, op):
        reduced_shapes.append(tuple(tensor.shape))
        # Simulate another rank contributing one batch of valid tokens.
        tensor[-2] = 8
        tensor[-1] = 1

    monkeypatch.setattr("src.training.trainer.dist.all_reduce", fake_all_reduce)
    metrics = trainer.evaluate_validation([], max_batches=1)

    assert reduced_shapes == [(6 + trainer.num_recurrent_steps + 2,)]
    assert metrics["val/num_tokens"] == 8
    assert metrics["val/num_batches"] == 1


# ── Smoke test: loss decreases when overfitting on a fixed batch ──────────────

def test_loss_decreases_overfitting(tiny_cfg, tmp_path):
    """Train 30 steps on the same batch — loss must decrease (overfitting check)."""
    trainer_cfg = TrainerConfig(
        lr=1e-2,
        max_steps=30,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=False,
        device="cpu",
        num_recurrent_steps=2,
    )
    torch.manual_seed(0)
    trainer = Trainer(tiny_cfg, trainer_cfg)
    fixed_batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))

    first_loss = trainer.train_step(fixed_batch)["loss"].item()
    for _ in range(28):
        trainer.train_step(fixed_batch)
    last_loss = trainer.train_step(fixed_batch)["loss"].item()

    assert last_loss < first_loss, (
        f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
    )


# ── Checkpointing ─────────────────────────────────────────────────────────────

def test_save_and_load_checkpoint(tiny_cfg, tiny_trainer_cfg, tmp_path):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    for _ in range(3):
        trainer.train_step(batch)

    path = trainer.save_checkpoint()
    assert path.exists()

    # Create a fresh trainer and load the checkpoint
    trainer2 = Trainer(tiny_cfg, tiny_trainer_cfg)
    loaded_step = trainer2.load_checkpoint(path)

    assert loaded_step == 3
    assert trainer2.step == 3

    # Weights should match
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), trainer2.model.named_parameters()
    ):
        torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")


def test_checkpoint_filename_contains_step(tiny_cfg, tiny_trainer_cfg, tmp_path):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    trainer.train_step(batch)
    path = trainer.save_checkpoint()
    assert "step_0000001" in path.name


# ── Full training loop ────────────────────────────────────────────────────────

def test_train_loop_runs_max_steps(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    dl = make_dataloader(tiny_cfg.vocab_size, seq_len=9, n_batches=5)
    trainer.train(dl)
    assert trainer.step == tiny_trainer_cfg.max_steps


def test_train_loop_cycles_dataloader(tiny_cfg, tiny_trainer_cfg):
    """Loop must continue past the end of the dataloader by cycling."""
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    # Dataloader with only 2 batches, but max_steps=10
    dl = make_dataloader(tiny_cfg.vocab_size, seq_len=9, n_batches=1, B=4)
    trainer.train(dl)  # should not raise StopIteration
    assert trainer.step == tiny_trainer_cfg.max_steps


@pytest.mark.parametrize("method", ["dmo", "dmo_entropy", "soft", "soft_entropy", "uniform"])
def test_all_public_training_methods_smoke(tiny_cfg, tmp_path, method):
    trainer_cfg = TrainerConfig(
        lr=1e-3,
        max_steps=1,
        training_method=method,
        beta_kl=0.1,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / method),
        use_wandb=False,
        device="cpu",
    )
    trainer = Trainer(tiny_cfg, trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (2, 9))
    diag = trainer.train_step(batch)
    assert diag["_stepped"]
    assert torch.isfinite(diag["loss"])


def test_uniform_defaults_to_configured_depth_and_reuses_depth_for_accumulation(
    tiny_cfg, tmp_path
):
    trainer_cfg = TrainerConfig(
        lr=1e-3,
        max_steps=1,
        num_recurrent_steps=2,
        training_method="uniform",
        grad_accum_steps=2,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "uniform"),
        use_wandb=False,
        device="cpu",
    )
    trainer = Trainer(tiny_cfg, trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (2, 9))
    first = trainer.train_step(batch)
    assert not first["_stepped"]
    sampled = first["sampled_depth"]
    assert 1 <= sampled <= 3  # default 2*x-1 for x=2
    assert trainer._uniform_depth_for_update == sampled

    second = trainer.train_step(batch)
    assert second["_stepped"]
    assert second["sampled_depth"] == sampled
    assert trainer._uniform_depth_for_update is None


def test_uniform_explicit_max_depth_is_respected(tiny_cfg, tmp_path):
    trainer_cfg = TrainerConfig(
        max_steps=1,
        num_recurrent_steps=2,
        training_method="uniform",
        uniform_max_depth=1,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "uniform_one"),
        use_wandb=False,
        device="cpu",
    )
    trainer = Trainer(tiny_cfg, trainer_cfg)
    diag = trainer.train_step(torch.randint(0, tiny_cfg.vocab_size, (2, 9)))
    assert diag["sampled_depth"] == 1


def test_uniform_default_depth_is_two_x_minus_one(tiny_cfg, tmp_path):
    trainer_cfg = TrainerConfig(
        training_method="uniform",
        num_recurrent_steps=2,
        checkpoint_dir=str(tmp_path / "uniform_bad"),
        use_wandb=False,
        device="cpu",
    )
    trainer = Trainer(tiny_cfg, trainer_cfg)
    assert trainer._uniform_depth_limit() == 3
    diag = trainer.train_step(torch.randint(0, tiny_cfg.vocab_size, (2, 9)))
    assert 1 <= diag["sampled_depth"] <= 3
