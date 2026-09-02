"""Motion dynamics modules for the GR00T N1.6 DynVLA reproduction.

The classes in this file extend the pretrained AlternateVLDiT without
renaming or replacing any of its parameters.  This lets the official GR00T
action expert remain the initialization for both source pretraining and
few-shot adaptation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from gr00t.model.modules.dit import AlternateVLDiT


class TrajectoryDynamicsEncoder(nn.Module):
    """Amortized initializer for the per-trajectory FM inversion code."""

    def __init__(
        self,
        action_dim: int,
        output_dim: int,
        action_horizon: int,
        hidden_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(action_dim * 2, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        num_heads = max(1, hidden_dim // 64)
        while hidden_dim % num_heads != 0:
            num_heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        deltas = torch.cat((actions[:, :1], actions[:, 1:] - actions[:, :-1]), dim=1)
        tokens = self.input_projection(torch.cat((actions, deltas), dim=-1))
        tokens = tokens + self.position_embedding[:, : actions.shape[1]].to(tokens.dtype)
        encoded = self.encoder(tokens).mean(dim=1)
        return self.output_norm(self.output_projection(encoded))


class TransitionEncoder(nn.Module):
    """Encode persistent transition statistics from an action chunk."""

    def __init__(self, action_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        deltas = torch.cat((actions[:, :1], actions[:, 1:] - actions[:, :-1]), dim=1)
        transitions = torch.cat((actions, deltas), dim=-1)
        return self.output_norm(self.network(transitions).mean(dim=1))


class DynamicsBank(nn.Module):
    """Learnable motion prototypes with differentiable soft retrieval."""

    def __init__(self, codebook_size: int, hidden_dim: int, temperature: float) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(codebook_size, hidden_dim))
        self.temperature = float(temperature)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize only the motion prototypes for the random-bank ablation."""
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def forward(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_norm = F.normalize(queries.float(), dim=-1)
        prototype_norm = F.normalize(self.prototypes.float(), dim=-1)
        logits = query_norm @ prototype_norm.transpose(-1, -2)
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        weights = torch.softmax(logits, dim=-1)
        retrieved = weights.to(self.prototypes.dtype) @ self.prototypes
        return retrieved.to(queries.dtype), weights


class DynVLAAlternateVLDiT(AlternateVLDiT):
    """AlternateVLDiT with midpoint dynamics identification and injection."""

    def __init__(
        self,
        *args,
        constraint_depth: int,
        codebook_size: int,
        codebook_temperature: float,
        dynamics_teacher_dropout: float = 0.5,
        disable_dynamics_bank: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 1 <= constraint_depth < len(self.transformer_blocks):
            raise ValueError(
                f"constraint_depth must be in [1, {len(self.transformer_blocks) - 1}], "
                f"got {constraint_depth}"
            )
        self.constraint_depth = int(constraint_depth)
        self.dynamics_teacher_dropout = float(dynamics_teacher_dropout)
        self.disable_dynamics_bank = bool(disable_dynamics_bank)
        self.identification_token = nn.Parameter(torch.empty(1, 1, self.inner_dim))
        nn.init.normal_(self.identification_token, mean=0.0, std=0.02)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(self.inner_dim),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        self.bank = DynamicsBank(codebook_size, self.inner_dim, codebook_temperature)
        self.register_buffer("adaptation_code", torch.empty(0), persistent=False)

    def set_adaptation_code(self, code: torch.Tensor | None) -> None:
        if code is None:
            self.adaptation_code = torch.empty(0, device=self.identification_token.device)
            return
        code = torch.as_tensor(
            code,
            device=self.identification_token.device,
            dtype=self.identification_token.dtype,
        )
        if code.ndim == 1:
            code = code.unsqueeze(0)
        if code.ndim != 2 or code.shape[-1] != self.inner_dim:
            raise ValueError(
                f"adaptation code must have shape [N, {self.inner_dim}], got {tuple(code.shape)}"
            )
        self.adaptation_code = code.detach()

    def set_dynvla_trainable(self, trainable: bool, train_bank: bool = True) -> None:
        self.identification_token.requires_grad_(trainable)
        self.query_projection.requires_grad_(trainable)
        self.bank.requires_grad_(trainable and train_bank and not self.disable_dynamics_bank)

    def set_pretrained_modules_to_eval_mode(self) -> None:
        """Keep frozen upstream DiT dropout disabled while DynVLA stays in train mode."""
        self.timestep_encoder.eval()
        self.transformer_blocks.eval()
        self.norm_out.eval()
        self.proj_out_1.eval()
        self.proj_out_2.eval()

    def _initial_prefix(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        dynamics_hint: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        learned = self.identification_token.expand(batch_size, -1, -1).to(
            device=device, dtype=dtype
        )
        using_adaptation_code = False
        if dynamics_hint is None and self.adaptation_code.numel() > 0:
            dynamics_hint = self.adaptation_code.mean(dim=0, keepdim=True).expand(batch_size, -1)
            using_adaptation_code = True
        if dynamics_hint is None:
            used_teacher = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return learned, used_teacher

        hint = dynamics_hint.to(device=device, dtype=dtype).unsqueeze(1)
        if not self.training:
            used_teacher = torch.full(
                (batch_size,), using_adaptation_code, dtype=torch.bool, device=device
            )
            return (hint if using_adaptation_code else learned), used_teacher
        used_teacher = torch.rand(batch_size, device=device) >= self.dynamics_teacher_dropout
        prefix = torch.where(used_teacher[:, None, None], hint, learned)
        return prefix, used_teacher

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
        dynamics_hint: torch.Tensor | None = None,
        return_dynvla_aux: bool = False,
    ):
        if image_mask is None:
            raise ValueError("image_mask is required")
        if backbone_attention_mask is None:
            raise ValueError("backbone_attention_mask is required")

        temb = self.timestep_encoder(timestep)
        batch_size = hidden_states.shape[0]
        prefix, used_teacher = self._initial_prefix(
            batch_size, hidden_states.dtype, hidden_states.device, dynamics_hint
        )
        hidden_states = torch.cat((prefix, hidden_states), dim=1).contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        all_hidden_states = [hidden_states]
        layer_queries = []
        bank_weights = None
        retrieved_prototypes = None

        if not self.config.interleave_self_attention:
            raise ValueError("DynVLA requires interleaved self attention")

        for index, block in enumerate(self.transformer_blocks):
            if index % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                if index % (2 * self.attend_text_every_n_blocks) == 0:
                    current_attention_mask = non_image_attention_mask
                else:
                    current_attention_mask = image_attention_mask
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=current_attention_mask,
                    temb=temb,
                )

            if index < self.constraint_depth:
                layer_queries.append(self.query_projection(hidden_states[:, 0]))
            if index == self.constraint_depth - 1:
                stacked_queries = torch.stack(layer_queries, dim=1)
                if self.disable_dynamics_bank:
                    retrieved_prototypes = stacked_queries
                else:
                    retrieved_prototypes, bank_weights = self.bank(stacked_queries)
                hidden_states = torch.cat(
                    (retrieved_prototypes, hidden_states[:, 1:]), dim=1
                )
            all_hidden_states.append(hidden_states)

        if retrieved_prototypes is None:
            raise RuntimeError("dynamics tokens were not injected")
        if not self.disable_dynamics_bank and bank_weights is None:
            raise RuntimeError("dynamics bank weights are missing")
        hidden_states = hidden_states[:, retrieved_prototypes.shape[1] :]
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)

        auxiliary = {
            "layer_queries": torch.stack(layer_queries, dim=1),
            "bank_weights": bank_weights,
            "retrieved_prototypes": retrieved_prototypes,
            "used_teacher": used_teacher,
        }
        if return_dynvla_aux:
            return output, all_hidden_states, auxiliary
        if return_all_hidden_states:
            return output, all_hidden_states
        return output


class TemporalGrounding(nn.Module):
    """Bidirectional temporal grounding with an EMA transition target."""

    def __init__(
        self,
        dynamics_dim: int,
        action_dim: int,
        hidden_dim: int,
        projection_dim: int,
        temperature: float,
        ema_momentum: float,
    ) -> None:
        super().__init__()
        self.dynamics_projector = nn.Sequential(
            nn.Linear(dynamics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.transition_encoder = TransitionEncoder(action_dim, hidden_dim, projection_dim)
        self.target_transition_encoder = copy.deepcopy(self.transition_encoder)
        self.target_transition_encoder.requires_grad_(False)
        self.temperature = float(temperature)
        self.ema_momentum = float(ema_momentum)

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(
            self.target_transition_encoder.parameters(),
            self.transition_encoder.parameters(),
            strict=True,
        ):
            target.mul_(self.ema_momentum).add_(online, alpha=1.0 - self.ema_momentum)

    def forward(self, actions: torch.Tensor, dynamics: torch.Tensor) -> torch.Tensor:
        if actions.shape[0] < 2:
            return dynamics.new_zeros(())
        if self.training:
            self.update_target()
        context = F.normalize(self.dynamics_projector(dynamics).float(), dim=-1)
        transition_online = F.normalize(self.transition_encoder(actions).float(), dim=-1)
        with torch.no_grad():
            transition_target = F.normalize(self.target_transition_encoder(actions).float(), dim=-1)
        labels = torch.arange(actions.shape[0], device=actions.device)
        temperature = max(self.temperature, 1e-6)
        forward_logits = context @ transition_target.transpose(0, 1) / temperature
        backward_logits = transition_online @ context.detach().transpose(0, 1) / temperature
        return 0.5 * (
            F.cross_entropy(forward_logits, labels)
            + F.cross_entropy(backward_logits, labels)
        )
