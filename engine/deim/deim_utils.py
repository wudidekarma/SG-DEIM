import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import get_activation, bias_init_with_prob


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'

# default 3-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# Taken from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py#L14-L34
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Gate(nn.Module):
    def __init__(self, d_model, use_rmsnorm=False):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class QueryConditionedAttentionGating(nn.Module):
    """QCAG applied to genuine per-head attention responses.

    ``head_output`` must be shaped ``[B, N, H, d]`` and must be captured
    before head concatenation/output projection.  This requirement prevents
    the previous implementation from relabeling post-projection channel
    groups as attention heads.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        mode='headwise',
        use_residual=True,
        residual_ratio=0.1,
        condition_source='query',
        intervention='normal',
        intervention_shift=1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads.')
        if not 0.0 <= residual_ratio < 1.0:
            raise ValueError('residual_ratio must lie in [0, 1).')
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.use_residual = use_residual
        self.residual_ratio = float(residual_ratio)
        valid_condition_sources = {'query', 'global'}
        if condition_source not in valid_condition_sources:
            raise ValueError(
                'condition_source must be one of '
                f'{sorted(valid_condition_sources)}, got {condition_source!r}.'
            )
        valid_interventions = {
            'normal', 'identity', 'query_mean', 'query_shuffle',
            'head_mean', 'head_shuffle',
        }
        if intervention not in valid_interventions:
            raise ValueError(
                'intervention must be one of '
                f'{sorted(valid_interventions)}, got {intervention!r}.'
            )
        if int(intervention_shift) < 1:
            raise ValueError('intervention_shift must be a positive integer.')
        self.condition_source = condition_source
        self.intervention = intervention
        self.intervention_shift = int(intervention_shift)
        
        # Learnable temperature T in QCAG.
        self.temperature = nn.Parameter(torch.ones(1))
        
        if mode == 'headwise':
            # Query-conditioned head gate.
            self.gate_proj = nn.Linear(d_model, num_heads)
        elif mode == 'headwise_v2':
            # Two-layer MLP head gate used in the paper experiments.
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.SiLU(),
                nn.Linear(d_model // 4, num_heads)
            )
        elif mode == 'elementwise':
            # Query-conditioned element gate.
            self.gate_proj = nn.Linear(d_model, d_model)
        else:
            raise ValueError(f"Unknown gating mode: {mode}")
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Gate + residual compensation is exactly one at initialization.  The
        # decoder therefore starts from the ungated baseline while the last
        # projection can immediately learn query-dependent deviations.
        initial_gate = 1.0 - self.residual_ratio if self.use_residual else 1.0 - 1e-3
        initial_gate = min(max(initial_gate, 1e-3), 1.0 - 1e-3)
        initial_bias = torch.logit(torch.tensor(initial_gate)).item()
        if self.mode == 'headwise_v2':
            init.xavier_uniform_(self.gate_proj[0].weight)
            init.zeros_(self.gate_proj[0].bias)
            init.zeros_(self.gate_proj[-1].weight)
            init.constant_(self.gate_proj[-1].bias, initial_bias)
        else:
            init.zeros_(self.gate_proj.weight)
            init.constant_(self.gate_proj.bias, initial_bias)

    def compute_gate(self, query):
        if self.condition_source == 'global':
            # Parameter-matched control: every query in an image receives the
            # same per-head gate predicted from the mean decoder state.
            gate_query = query.mean(dim=-2, keepdim=True).expand_as(query)
        else:
            gate_query = query
        gate_score = self.gate_proj(gate_query)
        temperature = self.temperature.clamp(min=0.05, max=20.0)
        return torch.sigmoid(gate_score * temperature)

    def apply_intervention(self, gate):
        """Evaluation-only counterfactuals for query/head correspondence."""
        if self.intervention == 'normal':
            return gate
        if self.intervention == 'identity':
            # Forward adds residual_ratio * head_output below.  Using 1-r as
            # the gate therefore yields an exact unit multiplier, unlike a
            # naive all-one gate which would amplify the response to 1+r.
            identity_gate = 1.0 - self.residual_ratio if self.use_residual else 1.0
            return torch.full_like(gate, identity_gate)
        if self.intervention == 'query_mean':
            return gate.mean(dim=-2, keepdim=True).expand_as(gate)
        if self.intervention == 'query_shuffle':
            if gate.shape[-2] < 2:
                raise ValueError('query_shuffle requires at least two queries.')
            return torch.roll(
                gate, shifts=self.intervention_shift, dims=-2
            )
        if self.mode not in {'headwise', 'headwise_v2'}:
            raise ValueError(
                f'{self.intervention} is defined only for head-wise QCAG.'
            )
        if self.intervention == 'head_mean':
            return gate.mean(dim=-1, keepdim=True).expand_as(gate)
        if gate.shape[-1] < 2:
            raise ValueError('head_shuffle requires at least two heads.')
        return torch.roll(gate, shifts=self.intervention_shift, dims=-1)
    
    def forward(self, query, head_output, return_gate=False):
        """
        Args:
            query: [B, N, d_model] - current query state.
            head_output: [B, N, num_heads, head_dim], captured before head
                concatenation and output projection.
        Returns:
            gated heads with the same shape as ``head_output``.
        """
        if head_output.ndim != 4:
            raise ValueError(
                'QCAG requires pre-aggregation head output shaped [B, N, H, d].'
            )
        if head_output.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f'Expected head shape {(self.num_heads, self.head_dim)}, '
                f'got {tuple(head_output.shape[-2:])}.'
            )

        if self.training and self.intervention != 'normal':
            raise RuntimeError(
                'QCAG intervention is evaluation-only. '
                'Use intervention=normal for training.'
            )
        gate = self.apply_intervention(self.compute_gate(query))
        if self.mode in ['headwise', 'headwise_v2']:
            gate_for_output = gate.unsqueeze(-1)
        else:
            gate_for_output = gate.view(
                *gate.shape[:-1], self.num_heads, self.head_dim
            )

        gated_output = head_output * gate_for_output
        if self.use_residual:
            gated_output = gated_output + self.residual_ratio * head_output

        if return_gate:
            return gated_output, gate
        return gated_output


# Backward-compatible import alias. New code uses the paper acronym QCAG.
GatedAttentionOutput = QueryConditionedAttentionGating
