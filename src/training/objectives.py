import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.looplm import compute_exit_distribution


def compute_looplm_loss(
    logits_per_step: list[Tensor],
    exit_lambdas: list[Tensor],
    targets: Tensor,
    beta: float = 0.0,
    ignore_index: int = -100,
    objective: str = "mixture_nll",
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Train with either the finite-mixture NLL or the expected-CE objective.

    ``objective="mixture_nll"`` uses
    ``L_mix = -log Σ_t p(t|x) exp(-L^(t))``.  ``objective="expected_ce"``
    uses the exit-weighted cross-entropy ``Σ_t p(t|x) L^(t)``.  Both can
    optionally include the same exit-distribution entropy bonus.

    With the default beta=0, this is the negative log-likelihood under the
    exit-weighted mixture of per-depth next-token distributions. A nonzero beta
    adds the repository's optional entropy bonus.

    Args:
        logits_per_step: T tensors of shape (B, S, vocab_size)
        exit_lambdas:    T tensors of shape (B, S) — raw sigmoid gate outputs λ_t
        targets:         (B, S) ground-truth token ids; use ignore_index for padding
        beta:            optional entropy regularization weight (default: 0)
        ignore_index:    token id to ignore in CE loss (default -100)
        objective:       ``"mixture_nll"`` or ``"expected_ce"``

    Returns:
        total_loss:  scalar tensor, differentiable w.r.t. both LM and gate params
        diagnostics: dict with detached scalars for logging
    """
    if objective not in {"mixture_nll", "expected_ce"}:
        raise ValueError(
            "objective must be 'mixture_nll' or 'expected_ce', "
            f"got {objective!r}"
        )

    T = len(logits_per_step)
    B, S, V = logits_per_step[0].shape

    # ── Per-step cross-entropy losses ────────────────────────────────────────
    # L^(t): (B, S) — per-token CE at each recurrent step
    ce_per_step: list[Tensor] = []
    for logits in logits_per_step:
        ce = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, S)
        ce_per_step.append(ce)

    # ── Exit distribution ────────────────────────────────────────────────────
    # p(t|x): (T, B, S) — sums to 1 over dim 0
    exit_probs = compute_exit_distribution(exit_lambdas)

    # ── Valid-token mask ─────────────────────────────────────────────────────
    mask = (targets != ignore_index).float()  # (B, S)
    n_valid = mask.sum().clamp(min=1.0)

    # ── Mixture task loss ─────────────────────────────────────────────────────
    # -log Σ_t p(t|x) exp(-L^(t)), averaged over valid tokens.
    ce_stack = torch.stack(ce_per_step, dim=0)         # (T, B, S)
    log_exit_probs = exit_probs.float().clamp_min(1e-30).log()
    mixture_nll_per_token = -torch.logsumexp(
        log_exit_probs - ce_stack.float(), dim=0
    )
    expected_ce_per_token = (exit_probs.float() * ce_stack.float()).sum(dim=0)
    mixture_nll = (mixture_nll_per_token * mask).sum() / n_valid
    expected_ce = (expected_ce_per_token * mask).sum() / n_valid
    task_loss = mixture_nll if objective == "mixture_nll" else expected_ce

    # ── Entropy regularization ───────────────────────────────────────────────
    # H(p) = -Σ_t p(t) · log p(t), averaged over valid tokens
    entropy = -(exit_probs.float() * log_exit_probs).sum(dim=0)  # (B, S)
    mean_entropy = (entropy * mask).sum() / n_valid

    # ── Total loss ───────────────────────────────────────────────────────────
    total_loss = task_loss - beta * mean_entropy

    # ── Diagnostics (detached) ───────────────────────────────────────────────
    with torch.no_grad():
        step_indices = torch.arange(1, T + 1, dtype=torch.float32, device=exit_probs.device)
        avg_exit_step = (exit_probs * step_indices.view(T, 1, 1)).sum(dim=0)  # (B, S)
        mean_avg_exit_step = (avg_exit_step * mask).sum() / n_valid

    diagnostics: dict[str, Tensor | float] = {
        "loss": total_loss.detach(),
        "task_loss": task_loss.detach(),
        "mixture_nll": mixture_nll.detach(),
        "expected_ce": expected_ce.detach(),
        "objective": objective,
        "entropy": mean_entropy.detach(),
        "avg_exit_step": mean_avg_exit_step.detach(),
        "per_step_losses": [ce.mean().detach() for ce in ce_per_step],
    }

    return total_loss, diagnostics


def compute_sampled_depth_final_ce(
    logits_per_step: list[Tensor],
    targets: Tensor,
    sampled_depth: int,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, Tensor | float | int]]:
    """Compute ordinary next-token CE at one uniformly sampled depth.

    This is the fixed-depth sampling baseline: the exit distribution is not
    used as a training target, and only the logits at ``sampled_depth``
    contribute to the loss.
    """
    if not 1 <= sampled_depth <= len(logits_per_step):
        raise ValueError(
            f"sampled_depth must be in [1, {len(logits_per_step)}], "
            f"got {sampled_depth}"
        )

    logits = logits_per_step[sampled_depth - 1]
    B, S, V = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, S)
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / n_valid
    return loss, {
        "loss": loss.detach(),
        "task_loss": loss.detach(),
        "mixture_nll": loss.detach(),
        "expected_ce": loss.detach(),
        "entropy": torch.zeros((), device=loss.device, dtype=loss.dtype),
        "avg_exit_step": torch.tensor(
            float(sampled_depth), device=loss.device, dtype=loss.dtype
        ),
        "sampled_depth": sampled_depth,
        "per_step_losses": [loss.detach()],
    }


def compute_adaptive_gate_loss(
    logits_per_step: list[Tensor],
    exit_lambdas: list[Tensor],
    targets: Tensor,
    k: float = 50.0,
    gamma: float = 0.005,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, Tensor | list]]:
    """Stage II adaptive gate training objective (paper Eq. 6).

    LM parameters are frozen by the caller; this loss only flows gradients
    through exit_lambdas (the gate). All CE computations use detached logits.

    For each step t = 2..T_max:
      I^(t)  = max(0, L^(t-1)_detached − L^(t)_detached)   per-token improvement
      w^(t)  = sigmoid(k · (I^(t) − γ))                    ideal continuation label
      loss_t = BCE(1 − λ^(t), w^(t))                        gate should continue iff w≈1

    Args:
        logits_per_step: T tensors of shape (B, S, vocab_size)
        exit_lambdas:    T tensors of shape (B, S) — raw sigmoid gate outputs λ_t
        targets:         (B, S) ground-truth token ids
        k:               sharpness of the sigmoid label (paper: 50.0)
        gamma:           improvement threshold below which gate should exit (paper: 0.005)
        ignore_index:    token id to ignore in CE loss

    Returns:
        total_loss:  scalar, differentiable w.r.t. gate params only (logits are detached)
        diagnostics: dict with detached scalars for logging
    """
    T = len(logits_per_step)
    assert T >= 2, "Adaptive gate loss requires at least 2 recurrent steps"
    B, S, V = logits_per_step[0].shape

    # ── Detached per-step CE losses ───────────────────────────────────────────
    # All CE is computed with detached logits — LM gets no gradients here
    ce_per_step: list[Tensor] = []
    for logits in logits_per_step:
        ce = F.cross_entropy(
            logits.detach().reshape(B * S, V),
            targets.reshape(B * S),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, S)
        ce_per_step.append(ce)

    # ── Valid-token mask ──────────────────────────────────────────────────────
    mask = (targets != ignore_index).float()  # (B, S)
    n_valid = mask.sum().clamp(min=1.0)

    # ── Per-step BCE losses (t = 2..T, i.e. index 1..T-1) ────────────────────
    bce_per_step: list[Tensor] = []
    mean_w_per_step: list[float] = []

    for t in range(1, T):
        # Improvement: how much did step t reduce the loss vs step t-1?
        improvement = (ce_per_step[t - 1] - ce_per_step[t]).clamp(min=0.0)  # (B, S)

        # Ideal continuation label: ≈1 when improvement is large, ≈0 otherwise
        w = torch.sigmoid(k * (improvement - gamma))  # (B, S)

        # Gate should predict "continue" (high 1−λ) when w≈1
        continuation = (1.0 - exit_lambdas[t]).clamp(min=1e-7, max=1 - 1e-7)
        bce = F.binary_cross_entropy(continuation, w, reduction="none")  # (B, S)
        bce_per_step.append(bce)

        with torch.no_grad():
            mean_w_per_step.append(((w * mask).sum() / n_valid).item())

    # ── Average over steps and valid tokens ───────────────────────────────────
    bce_stack = torch.stack(bce_per_step, dim=0)  # (T-1, B, S)
    total_loss = (bce_stack * mask.unsqueeze(0)).sum() / (n_valid * (T - 1))

    diagnostics: dict[str, Tensor | list] = {
        "loss": total_loss.detach(),
        "mean_w_per_step": mean_w_per_step,   # avg ideal-continuation label per step
        "per_step_bce": [b.mean().detach() for b in bce_per_step],
    }

    return total_loss, diagnostics


def compute_q_act_loss(
    logits_per_step: list[Tensor],
    q_values_per_step: list[Tensor],
    exit_lambdas: list[Tensor],
    targets: Tensor,
    beta: float = 0.0,
    q_weight: float = 0.1,
    q_gamma: float = 0.99,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Q-learning ACT loss: mixture NLL + optional entropy and Q TD losses.

    The Q-head learns to predict the value of halting vs continuing.
    Reward at step t = improvement in CE: r_t = CE_{t-1} - CE_t
    Q(halt, t)     target = r_t  (halt: collect reward, stop)
    Q(continue, t) target = r_t + gamma * max(Q(halt,t+1), Q(cont,t+1))

    The task loss uses the exit-weighted mixture likelihood.

    Args:
        logits_per_step: T tensors of (B, S, V)
        q_values_per_step: T tensors of (B, S, 2) — [Q(halt), Q(continue)]
        exit_lambdas: T tensors of (B, S) — still used for task loss weighting
        targets: (B, S)
        beta: optional entropy regularization weight for task loss
        q_weight: weight for Q-learning loss relative to task loss
        q_gamma: discount factor for TD targets
        ignore_index: padding token id
    """
    T = len(logits_per_step)
    B, S, V = logits_per_step[0].shape

    # ── Per-step CE losses ────────────────────────────────────────────────────
    ce_per_step: list[Tensor] = []
    for logits in logits_per_step:
        ce = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, S)
        ce_per_step.append(ce)

    # ── Mixture task loss (same as compute_looplm_loss) ────────────────────────
    exit_probs = compute_exit_distribution(exit_lambdas)
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1.0)

    ce_stack = torch.stack(ce_per_step, dim=0)
    log_exit_probs = exit_probs.float().clamp_min(1e-30).log()
    mixture_nll = -torch.logsumexp(log_exit_probs - ce_stack.float(), dim=0)
    task_loss = (mixture_nll * mask).sum() / n_valid

    entropy = -(exit_probs.float() * log_exit_probs).sum(dim=0)
    mean_entropy = (entropy * mask).sum() / n_valid

    # ── Q-learning TD loss ────────────────────────────────────────────────────
    ce_detached = [ce.detach() for ce in ce_per_step]
    q_losses: list[Tensor] = []

    for t in range(T):
        q_vals = q_values_per_step[t]  # (B, S, 2): [Q_halt, Q_continue]
        q_halt = q_vals[:, :, 0]
        q_cont = q_vals[:, :, 1]

        # Reward: improvement in CE vs previous step
        if t == 0:
            reward = torch.zeros_like(ce_detached[0])
        else:
            reward = (ce_detached[t - 1] - ce_detached[t]).clamp(min=0.0)

        # TD target for Q(halt): immediate reward (halting ends computation)
        q_halt_target = reward

        # TD target for Q(continue): reward + gamma * max Q at next step
        if t < T - 1:
            with torch.no_grad():
                next_q = q_values_per_step[t + 1].detach()
                next_max_q = next_q.max(dim=-1).values
            q_cont_target = reward + q_gamma * next_max_q
        else:
            # Last step: must halt
            q_cont_target = reward

        td_halt = (q_halt - q_halt_target.detach()).pow(2)
        td_cont = (q_cont - q_cont_target.detach()).pow(2)
        q_loss_t = (td_halt + td_cont) * mask
        q_losses.append(q_loss_t)

    q_loss = torch.stack(q_losses, dim=0).sum(dim=0).sum() / (n_valid * T)

    # ── Combined loss ─────────────────────────────────────────────────────────
    total_loss = (task_loss - beta * mean_entropy) + q_weight * q_loss

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with torch.no_grad():
        step_indices = torch.arange(1, T + 1, dtype=torch.float32, device=exit_probs.device)
        avg_exit_step = (exit_probs * step_indices.view(T, 1, 1)).sum(dim=0)
        mean_avg_exit_step = (avg_exit_step * mask).sum() / n_valid

        halt_fracs = []
        for t in range(T):
            q_vals = q_values_per_step[t].detach()
            halt_frac = ((q_vals[:, :, 0] > q_vals[:, :, 1]).float() * mask).sum() / n_valid
            halt_fracs.append(halt_frac.item())

    diagnostics: dict[str, Tensor | float] = {
        "loss": total_loss.detach(),
        "task_loss": task_loss.detach(),
        "entropy": mean_entropy.detach(),
        "q_loss": q_loss.detach(),
        "avg_exit_step": mean_avg_exit_step.detach(),
        "per_step_losses": [ce.mean().detach() for ce in ce_per_step],
        "q_halt_fracs": halt_fracs,
    }

    return total_loss, diagnostics
