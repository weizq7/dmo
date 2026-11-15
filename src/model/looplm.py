from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from src.model.config import LoopLMConfig
from src.model.rope import RotaryEmbedding
from src.model.transformer import RMSNorm, TransformerBlock, PostNormTransformerBlock


class LoopLMOutput(NamedTuple):
    # One tensor per recurrent step
    logits: list[Tensor]       # each (B, S, vocab_size)
    exit_lambdas: list[Tensor] # each (B, S) — raw sigmoid gate outputs λ_t
    q_values: list[Tensor] | None = None  # each (B, S, 2) — Q(halt), Q(continue)
    aux_losses: list[Tensor] | None = None  # auxiliary losses (e.g. MoE load balance)


def compute_exit_distribution(exit_lambdas: list[Tensor]) -> Tensor:
    """Compute p(t|x) from raw gate outputs λ_t.

    For t < T_max:  p(t|x) = λ_t · S_{t-1}   where S_0 = 1
    For t = T_max:  p(T_max|x) = S_{T_max-1}

    Args:
        exit_lambdas: list of T tensors, each (B, S)

    Returns:
        exit_probs: (T, B, S) — sums to 1 over the T dimension
    """
    T = len(exit_lambdas)
    probs = []
    survival = torch.ones_like(exit_lambdas[0])  # S_0 = 1

    for t, lam in enumerate(exit_lambdas):
        if t < T - 1:
            p_t = lam * survival
            survival = survival * (1.0 - lam)
        else:
            # Last step absorbs remaining probability mass
            p_t = survival
        probs.append(p_t)

    return torch.stack(probs, dim=0)  # (T, B, S)


def model_kwargs_from_checkpoint(checkpoint: dict) -> dict:
    """Recover the standard recurrence modules used by a saved checkpoint."""
    state = checkpoint["model_state_dict"]
    kwargs = dict(checkpoint.get("model_kwargs") or {})
    if "q_head.weight" in state:
        kwargs.setdefault("use_q_act", True)
    if "grt_input_proj.weight" in state:
        kwargs.setdefault("use_grt", True)
    return kwargs


class LoopLM(nn.Module):
    """Looped Language Model with shared-weight recurrence.

    A stack of `num_layers` TransformerBlocks is applied `num_steps` times,
    reusing the same weights each pass. An LM head and an exit gate are applied
    after every recurrent step.

    Supports ablation modes:
        - H/L split: partition layers into slow (H) and fast (L) modules
        - Q-learning ACT: replace exit gate with Q-head for halt/continue
        - MoE recurrence: route to expert TransformerBlocks per step
    """

    def __init__(
        self,
        config: LoopLMConfig,
        use_postnorm: bool = False,
        use_lecun_init: bool = False,
        use_q_act: bool = False,
        use_grt: bool = False,
        grt_state_noise: float = 0.1,
        grt_gate_noise: float = 0.1,
        grt_gate_temperature: float = 1.0,
        use_moe_recurrence: bool = False,
        num_expert_layers: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.config = config
        self.use_q_act = use_q_act
        self.use_grt = bool(use_grt)
        self.grt_state_noise = float(grt_state_noise)
        self.grt_gate_noise = float(grt_gate_noise)
        self.grt_gate_temperature = float(grt_gate_temperature)
        if self.grt_gate_temperature <= 0.0:
            raise ValueError("grt_gate_temperature must be positive")
        if self.grt_state_noise < 0.0 or self.grt_gate_noise < 0.0:
            raise ValueError("GRT noise scales must be non-negative")
        self.use_moe_recurrence = use_moe_recurrence

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        if use_moe_recurrence:
            from src.model.moe_layer import MoELayer
            self.moe_layer = MoELayer(
                config,
                num_experts=num_expert_layers,
                top_k=moe_top_k,
                use_postnorm=use_postnorm,
            )
            # Still keep regular layers for non-MoE parts (e.g. H/L split H-module)
            self.layers = nn.ModuleList()
        else:
            block_cls = PostNormTransformerBlock if use_postnorm else TransformerBlock
            self.layers = nn.ModuleList(
                [block_cls(config) for _ in range(config.num_layers)]
            )

        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Exit gate (standard Bernoulli)
        self.exit_gate = nn.Linear(config.hidden_size, 1, bias=True)

        # Q-learning ACT head (optional, replaces exit gate for halting decisions)
        if use_q_act:
            self.q_head = nn.Linear(config.hidden_size, 2, bias=True)

        # Tie LM head weights to the embedding matrix (standard practice)
        self.lm_head.weight = self.embed.weight

        self.rope = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_heads,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        self.gradient_checkpointing = False

        if use_lecun_init:
            self._init_weights_lecun()
        else:
            self._init_weights()

        if self.use_grt:
            # GRT conditions each recurrent proposal on the current state and
            # the fixed input embedding, then mixes it with the current state
            # using an elementwise copy/write gate.
            d = config.hidden_size
            self.grt_state_norm = nn.LayerNorm(d)
            self.grt_anchor_norm = nn.LayerNorm(d)
            self.grt_input_proj = nn.Linear(2 * d, d)
            self.grt_gate = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.SiLU(),
                nn.Linear(d, d),
            )
            self._init_grt_weights(use_lecun_init=use_lecun_init)

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        nn.init.zeros_(self.exit_gate.bias)
        nn.init.normal_(self.exit_gate.weight, mean=0.0, std=std)
        if self.use_q_act:
            nn.init.zeros_(self.q_head.bias)
            nn.init.normal_(self.q_head.weight, mean=0.0, std=std)
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)
        if self.use_moe_recurrence:
            for p in self.moe_layer.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights_lecun(self) -> None:
        """Truncated LeCun Normal init (HRM-style): std = 1/sqrt(fan_in), truncated at 2σ."""
        import math
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def _init_grt_weights(self, use_lecun_init: bool = False) -> None:
        """Initialize GRT modules in a near-copy regime."""
        if use_lecun_init:
            import math
            for module in (self.grt_input_proj, *self.grt_gate.modules()):
                if isinstance(module, nn.Linear):
                    std = 1.0 / math.sqrt(module.weight.shape[1])
                    nn.init.trunc_normal_(
                        module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # sigmoid(4) is approximately 0.982: start close to copying h.
        nn.init.constant_(self.grt_gate[-1].bias, 4.0)

    def _apply_layers(
        self,
        h: Tensor,
        layers: list[nn.Module],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """Apply a list of transformer layers to hidden state."""
        use_ckpt = self.training and self.gradient_checkpointing
        for layer in layers:
            if use_ckpt:
                h = checkpoint(layer, h, cos, sin, attn_mask, use_reentrant=False)
            else:
                h = layer(h, cos, sin, attn_mask)
        return h

    def _read_out(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute logits, exit lambda, and optionally Q-values from hidden state."""
        logits = self.lm_head(self.final_norm(h))
        lam = torch.sigmoid(self.exit_gate(h)).squeeze(-1)
        q_vals = self.q_head(h) if self.use_q_act else None
        return logits, lam, q_vals

    def forward(
        self,
        input_ids: Tensor,
        num_steps: int | None = None,
        attention_mask: Tensor | None = None,
        detach_between_steps: bool = False,
        recurrence: str = "naive",
        # H/L split params
        use_hl_split: bool = False,
        n_h_layers: int = 2,
        t_inner: int = 2,
        t_outer: int = 2,
    ) -> LoopLMOutput:
        """
        Args:
            input_ids:      (B, S) token indices
            num_steps:      number of recurrent steps; defaults to config.max_recurrent_steps
            attention_mask: optional (B, 1, S, S) additive float mask
            detach_between_steps: detach hidden state between steps
            use_hl_split:   enable H/L module split
            n_h_layers:     number of H (slow) layers (first N layers)
            t_inner:        L inner loop steps
            t_outer:        H outer loop steps
        """
        if num_steps is None:
            num_steps = self.config.max_recurrent_steps

        if recurrence not in ("naive", "grt"):
            raise ValueError(f"Unknown recurrence strategy: {recurrence}")
        if recurrence == "grt" and not self.use_grt:
            raise ValueError("recurrence='grt' requires model construction with use_grt=True")

        B, S = input_ids.shape
        cos, sin = self.rope.get_cos_sin(S, input_ids.device)
        h = self.embed(input_ids)

        if self.use_moe_recurrence:
            return self._forward_moe(h, num_steps, cos, sin, attention_mask, detach_between_steps)
        elif use_hl_split:
            return self._forward_hl_split(h, cos, sin, attention_mask, n_h_layers, t_inner, t_outer)
        elif recurrence == "grt":
            return self._forward_grt(h, num_steps, cos, sin, attention_mask, detach_between_steps)
        else:
            return self._forward_standard(h, num_steps, cos, sin, attention_mask, detach_between_steps)

    def _forward_grt(
        self,
        h: Tensor,
        num_steps: int,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        detach_between_steps: bool,
    ) -> LoopLMOutput:
        """Run the minimal GRT recurrent update.

        The initial embedding is held fixed as an anchor throughout the
        recurrence. Each step proposes a shared-stack update from the current
        state and anchor, then uses an elementwise gate to copy the old state
        or write the proposal into the next state.
        """
        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        anchor = h

        for _ in range(num_steps):
            if detach_between_steps:
                h = h.detach()

            h_fp32 = h.float()
            anchor_fp32 = anchor.float()
            state_input = h_fp32
            if self.training and self.grt_state_noise > 0.0:
                state_input = state_input + torch.randn_like(state_input) * self.grt_state_noise

            proposal_input = torch.cat([state_input, anchor_fp32], dim=-1).to(h.dtype)
            proposal = self._apply_layers(
                self.grt_input_proj(proposal_input),
                list(self.layers),
                cos,
                sin,
                attn_mask,
            )

            gate_input = torch.cat(
                [self.grt_state_norm(h_fp32), self.grt_anchor_norm(anchor_fp32)],
                dim=-1,
            ).to(h.dtype)
            gate_logits = self.grt_gate(gate_input) / self.grt_gate_temperature
            if self.training and self.grt_gate_noise > 0.0:
                gate_logits = gate_logits + torch.randn_like(gate_logits) * self.grt_gate_noise
            copy_gate = torch.sigmoid(gate_logits).float()
            h = (copy_gate * h_fp32 + (1.0 - copy_gate) * proposal.float()).to(h.dtype)

            logits, lam, _ = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)

        return LoopLMOutput(logits=logits_per_step, exit_lambdas=exit_lambdas)

    def _forward_standard(
        self,
        h: Tensor,
        num_steps: int,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        detach_between_steps: bool,
    ) -> LoopLMOutput:
        """Standard recurrent forward pass."""
        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []

        for _ in range(num_steps):
            if detach_between_steps:
                h = h.detach()
            h = self._apply_layers(h, list(self.layers), cos, sin, attn_mask)
            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
        )

    def _forward_hl_split(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        n_h_layers: int,
        t_inner: int,
        t_outer: int,
    ) -> LoopLMOutput:
        """H/L split forward: L runs t_inner steps, H updates once, repeat t_outer times.

        H layers = self.layers[:n_h_layers]  (slow, global context)
        L layers = self.layers[n_h_layers:]  (fast, local refinement)

        Execution pattern per outer step:
            1. L layers run t_inner times (with detach between inner steps)
            2. H layers run once
            3. Read out logits/exit
        """
        h_layers = list(self.layers[:n_h_layers])
        l_layers = list(self.layers[n_h_layers:])

        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []

        for outer in range(t_outer):
            # L inner loop: fast refinement
            for inner in range(t_inner):
                if inner > 0 or outer > 0:
                    h = h.detach()
                h = self._apply_layers(h, l_layers, cos, sin, attn_mask)

            # H update: slow global context
            h = self._apply_layers(h, h_layers, cos, sin, attn_mask)

            # Read out after each H update
            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
        )

    def _forward_moe(
        self,
        h: Tensor,
        num_steps: int,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        detach_between_steps: bool,
    ) -> LoopLMOutput:
        """MoE recurrence: route to expert layers at each step."""
        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []
        aux_losses: list[Tensor] = []

        for _ in range(num_steps):
            if detach_between_steps:
                h = h.detach()
            h, lb_loss = self.moe_layer(h, cos, sin, attn_mask)
            aux_losses.append(lb_loss)

            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
            aux_losses=aux_losses,
        )
