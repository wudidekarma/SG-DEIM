"""
SG-DEIM encoder: CGSR + CSAI
============================
This module implements the paper's encoder-side components:
1. CGSR: Context-Guided Spectral Redistribution at the encoder input.
2. C2/P2 detail bypass: optional shallow-detail injection into P3.
3. CSAI: Cross-Scale Adaptive Interaction before FPN/PAN feature addition.

The public encoder name is ``SGDEIMEncoder``. Historical names remain import
aliases only, so new configs and logs consistently follow the paper.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .hybrid_encoder import (
    ConvNormLayer_fuse, ConvNormLayer, SCDown, VGGBlock, 
    CSPLayer, RepNCSPELAN4, CSPLayer2, RepNCSPELAN5,
    TransformerEncoderLayer, TransformerEncoder
)
from .csai import (
    CrossScaleAdaptiveInteraction,
    LightweightCrossScaleAdaptiveInteraction,
)
from ..core import register

__all__ = [
    'ContextGuidedSpectralRedistribution',
    'SGDEIMEncoder',
    'CrossScaleSpectralRouter',
    'FreqAttention',
    'HybridEncoder_ECSMF',
]


class ContextGuidedSpectralRedistribution(nn.Module):
    """Context-guided, energy-balanced spectral redistribution.

    The earlier spectral module multiplied every frequency coordinate in a channel by the
    same scalar, which is exactly equivalent to spatial-domain channel
    scaling. CGSR instead predicts sample-, channel-, and band-dependent
    gains from spectral statistics and adjacent coarse-scale semantics.

    The band gains are mass-centered to remove every frequency-independent
    component.  A Parseval-consistent normalization then keeps the one-sided
    RFFT representation energy-balanced.  Only the induced spectral
    correction is written back through a zero-initialized residual scale.
    """

    def __init__(
        self,
        channel,
        context_channel=None,
        num_bands=4,
        reduction=4,
        max_strength=0.5,
        initial_bandwidth=None,
        eps=1e-6,
    ):
        super().__init__()
        if num_bands < 2:
            raise ValueError('CGSR requires at least two frequency bands.')
        if reduction < 1:
            raise ValueError('reduction must be a positive integer.')

        self.channel = channel
        self.context_channel = context_channel or channel
        self.num_bands = num_bands
        self.max_strength = float(max_strength)
        self.eps = float(eps)

        self.register_buffer('band_centers', torch.linspace(0.0, 1.0, num_bands))
        if initial_bandwidth is None:
            initial_bandwidth = 1.0 / max(num_bands - 1, 1)
        self.band_log_width = nn.Parameter(
            torch.full((num_bands,), self._inverse_softplus(initial_bandwidth))
        )

        # A point-wise projection aligns adjacent-scale semantics to the
        # channel space of the feature being routed.  Spatial sizes need not
        # match because the projected context is globally pooled.
        self.context_proj = nn.Conv2d(
            self.context_channel, channel, kernel_size=1, bias=False
        )
        self.context_norm = nn.LayerNorm(channel)

        # The same compact router is applied independently to each channel.
        # Its input is that channel's K relative band statistics plus one
        # aligned coarse-semantic descriptor.
        router_hidden = max(channel // reduction, num_bands * 2)
        self.router = nn.Sequential(
            nn.Linear(num_bands + 1, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, num_bands),
        )

        self.correction = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self._reset_parameters()

    @staticmethod
    def _inverse_softplus(value):
        value = torch.as_tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value)).item()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.context_proj.weight, a=1.0)
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        # A non-zero spectral correction is required for the zero-initialized
        # residual scale to receive a gradient on the first optimization step.
        nn.init.xavier_uniform_(self.router[-1].weight, gain=0.1)
        nn.init.zeros_(self.router[-1].bias)

    def _soft_bands(self, height, width, device):
        """Return soft radial bands and one-sided RFFT Hermitian weights."""
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius = (radius / (2.0 ** -0.5)).clamp_(0.0, 1.0)

        widths = F.softplus(self.band_log_width.float()).clamp_min(self.eps)
        distance = radius.unsqueeze(0) - self.band_centers.float().view(-1, 1, 1)
        logits = -0.5 * (distance / widths.view(-1, 1, 1)).square()
        bands = torch.softmax(logits, dim=0)

        hermitian = torch.full(
            (width // 2 + 1,), 2.0, device=device, dtype=torch.float32
        )
        hermitian[0] = 1.0
        if width % 2 == 0:
            hermitian[-1] = 1.0
        hermitian = hermitian.view(1, 1, 1, -1)
        return bands, hermitian

    def compute_balanced_mask(self, x, context=None):
        """Compute the CGSR mask and diagnostics without spatial write-back."""
        if context is None:
            context = x
        if x.ndim != 4 or context.ndim != 4:
            raise ValueError('CGSR expects BCHW feature tensors.')
        if x.shape[1] != self.channel:
            raise ValueError(f'Expected {self.channel} input channels, got {x.shape[1]}.')
        if context.shape[1] != self.context_channel:
            raise ValueError(
                f'Expected {self.context_channel} context channels, got {context.shape[1]}.'
            )

        _, _, height, width = x.shape
        x_fp32 = x.float()
        context_fp32 = context.float()
        spectrum = torch.fft.rfft2(x_fp32, norm='ortho')
        magnitude = spectrum.abs()
        bands, hermitian = self._soft_bands(height, width, x.device)

        weighted_bands = bands.unsqueeze(0).unsqueeze(0) * hermitian.unsqueeze(2)
        band_mass = weighted_bands.sum(dim=(-2, -1)).clamp_min(self.eps)
        band_stats = (
            torch.log1p(magnitude).unsqueeze(2) * weighted_bands
        ).sum(dim=(-2, -1)) / band_mass
        band_stats = F.layer_norm(band_stats, (self.num_bands,))

        semantic = self.context_proj(context_fp32).mean(dim=(-2, -1))
        semantic = torch.tanh(self.context_norm(semantic)).unsqueeze(-1)
        gains = torch.tanh(self.router(torch.cat([band_stats, semantic], dim=-1)))

        # pi is the mass of each band in the complete real-signal spectrum.
        band_mass_fraction = band_mass.flatten()
        band_mass_fraction = band_mass_fraction / band_mass_fraction.sum()
        centered_gains = gains - (
            gains * band_mass_fraction.view(1, 1, -1)
        ).sum(dim=-1, keepdim=True)

        log_mask = self.max_strength * torch.einsum(
            'bck,khw->bchw', centered_gains, bands
        )
        mask = torch.exp(log_mask)

        input_energy = (spectrum.abs().square() * hermitian).sum(
            dim=(-2, -1), keepdim=True
        )
        routed_energy = (
            (spectrum * mask).abs().square() * hermitian
        ).sum(dim=(-2, -1), keepdim=True)
        energy_scale = torch.sqrt(
            (input_energy + self.eps) / (routed_energy + self.eps)
        )
        balanced_mask = mask * energy_scale

        diagnostics = {
            'bands': bands,
            'band_statistics': band_stats,
            'raw_gains': gains,
            'centered_gains': centered_gains,
            'mask': balanced_mask,
            'input_energy': input_energy,
            'routed_energy': (
                (spectrum * balanced_mask).abs().square() * hermitian
            ).sum(dim=(-2, -1), keepdim=True),
        }
        return spectrum, balanced_mask, diagnostics

    def forward(self, x, context=None, return_diagnostics=False):
        input_dtype = x.dtype
        spectrum, balanced_mask, diagnostics = self.compute_balanced_mask(x, context)
        correction_spectrum = spectrum * (balanced_mask - 1.0)
        correction = torch.fft.irfft2(
            correction_spectrum, s=x.shape[-2:], norm='ortho'
        ).to(input_dtype)
        output = x + self.residual_scale * self.correction(correction)
        if return_diagnostics:
            return output, diagnostics
        return output


# Historical import aliases.
CrossScaleSpectralRouter = ContextGuidedSpectralRedistribution
FreqAttention = ContextGuidedSpectralRedistribution

@register()
class SGDEIMEncoder(nn.Module):
    """
    SG-DEIM encoder with CGSR and CSAI.
    
    Key features:
    - CGSR redistributes spectral energy before encoder projection.
    - Optional C2/P2 bypass injects shallow details into P3.
    - CSAI is applied before FPN/PAN adjacent-scale addition.
    
    Args:
        use_csai: Whether to enable CSAI fusion modules.
        csai_position: Where to apply CSAI ('fpn', 'pan', 'both').
        csai_type: CSAI implementation variant: 'efficient' or 'ultralight'.
        csai_kernel_size: Sequence-mixing kernel size used by CSAI.
    """
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 csp_type='csp',
                 fuse_op='sum',
                 # CSAI parameters.
                 use_csai=True,
                 csai_position='both',
                 csai_type='efficient',  # 'efficient' or 'ultralight'
                 csai_kernel_size=5,
                 csai_reduction=4,
                 # CGSR parameters.
                 use_cgsr=True,
                 cgsr_num_bands=4,
                 cgsr_reduction=4,
                 cgsr_max_strength=0.5,
                 # Training-time context source. ``coarse`` is the proposed
                 # cross-scale design; ``self`` is the controlled ablation.
                 cgsr_context_source='coarse',
                 # Evaluation-only causal intervention for CGSR context.
                 # ``shuffle`` uses a deterministic one-sample batch roll, so
                 # evaluation batches must contain at least two images.
                 cgsr_context_intervention='normal',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op
        self.use_csai = use_csai
        self.csai_position = csai_position
        self.use_cgsr = use_cgsr
        valid_context_sources = {'coarse', 'self'}
        if cgsr_context_source not in valid_context_sources:
            raise ValueError(
                'cgsr_context_source must be one of '
                f'{sorted(valid_context_sources)}, got '
                f'{cgsr_context_source!r}.'
            )
        self.cgsr_context_source = cgsr_context_source
        valid_context_interventions = {'normal', 'zero', 'shuffle'}
        if cgsr_context_intervention not in valid_context_interventions:
            raise ValueError(
                'cgsr_context_intervention must be one of '
                f'{sorted(valid_context_interventions)}, got '
                f'{cgsr_context_intervention!r}.'
            )
        self.cgsr_context_intervention = cgsr_context_intervention

        print("        ### SG-DEIM encoder: CGSR + CSAI ###")
        print(f"        ### CSAI: {use_csai}, position: {csai_position}, type: {csai_type} ###")
        print(f"        ### CGSR: {use_cgsr} ###")
        if use_cgsr:
            print(f"        ### CGSR context source: {cgsr_context_source} ###")
        if cgsr_context_intervention != 'normal':
            print(
                '        ### CGSR context intervention '
                f'(evaluation only): {cgsr_context_intervention} ###'
            )

        # In the proposed model each scale is conditioned by its adjacent
        # coarser scale and the deepest scale is self-conditioned.  The
        # ``self`` source keeps the router structure while replacing every
        # cross-scale link by a same-scale control for a fair ablation.
        if self.use_cgsr:
            print(
                "        ### Using CGSR (context-guided, energy-balanced spectral redistribution) ###"
            )
            self.cgsr_modules = nn.ModuleList([
                ContextGuidedSpectralRedistribution(
                    channel=in_channel,
                    context_channel=(
                        in_channels[min(i + 1, len(in_channels) - 1)]
                        if cgsr_context_source == 'coarse'
                        else in_channel
                    ),
                    num_bands=cgsr_num_bands,
                    reduction=cgsr_reduction,
                    max_strength=cgsr_max_strength,
                )
                for i, in_channel in enumerate(in_channels)
            ])
            self.p2_cgsr = ContextGuidedSpectralRedistribution(
                channel=in_channels[0],
                context_channel=in_channels[0],
                num_bands=cgsr_num_bands,
                reduction=cgsr_reduction,
                max_strength=cgsr_max_strength,
            )
        else:
            self.cgsr_modules = None
            self.p2_cgsr = None

        # Channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # Encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) 
            for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if fuse_op == 'sum' else hidden_dim * 2

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim*2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        
        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type)
        elif version == 'deim':
            Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        else:
            Fuse_Block = CSPLayer(in_channels=c1, out_channels=c2, num_blocks=num_blocks, act=act,
                                  expansion=expansion, bottletype=VGGBlock)
            Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act)
            SCDown_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)

        # Top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # Bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        # CSAI modules before adjacent-scale addition.
        if use_csai:
            if csai_type == 'efficient':
                CSAIClass = CrossScaleAdaptiveInteraction
                csai_kwargs = {
                    'dim': hidden_dim,
                    'kernel_size': csai_kernel_size,
                    'reduction': csai_reduction,
                }
            else:  # ultralight
                CSAIClass = LightweightCrossScaleAdaptiveInteraction
                csai_kwargs = {
                    'dim': hidden_dim,
                    'kernel_size': csai_kernel_size,
                }
            
            # Top-down FPN CSAI modules.
            if csai_position in ['fpn', 'both']:
                self.fpn_csai = nn.ModuleList([
                    CSAIClass(**csai_kwargs) for _ in range(len(in_channels) - 1)
                ])
            else:
                self.fpn_csai = None
            
            # Bottom-up PAN CSAI modules.
            if csai_position in ['pan', 'both']:
                self.pan_csai = nn.ModuleList([
                    CSAIClass(**csai_kwargs) for _ in range(len(in_channels) - 1)
                ])
            else:
                self.pan_csai = None
        else:
            self.fpn_csai = None
            self.pan_csai = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Translate pre-rename encoder keys when loading old checkpoints."""
        legacy_prefixes = {
            'freq_attns.': 'cgsr_modules.',
            'p2_freq_attn.': 'p2_cgsr.',
            'fpn_csmf.': 'fpn_csai.',
            'pan_csmf.': 'pan_csai.',
        }
        for old_name, new_name in legacy_prefixes.items():
            old_prefix = prefix + old_name
            new_prefix = prefix + new_name
            for key in [key for key in state_dict if key.startswith(old_prefix)]:
                new_key = new_prefix + key[len(old_prefix):]
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        # C2/P2 shallow-detail bypass. CGSR operations are computed from
        # the original backbone levels so their semantic conditions are not
        # contaminated by an order-dependent in-place update.
        p2 = None
        if len(feats) == len(self.in_channels) + 1:
            p2 = feats[0]
            feats = list(feats[1:])
        else:
            feats = list(feats)
        assert len(feats) == len(self.in_channels)
        
        if self.use_cgsr:
            if self.training and self.cgsr_context_intervention != 'normal':
                raise RuntimeError(
                    'CGSR context intervention is evaluation-only. '
                    'Use cgsr_context_intervention=normal for training.'
                )

            def context_for_level(index):
                # The deepest level has no adjacent coarser feature and stays
                # self-conditioned under every intervention.  Interventions
                # therefore test only genuine cross-scale links.
                if index == len(feats) - 1:
                    return feats[index]

                context = (
                    feats[index + 1]
                    if self.cgsr_context_source == 'coarse'
                    else feats[index]
                )
                if self.cgsr_context_intervention == 'zero':
                    return torch.zeros_like(context)
                if self.cgsr_context_intervention == 'shuffle':
                    if context.shape[0] < 2:
                        raise ValueError(
                            'CGSR shuffle intervention requires an '
                            'evaluation batch with at least two images.'
                        )
                    return torch.roll(context, shifts=1, dims=0)
                return context

            enhanced_feats = [
                self.cgsr_modules[i](
                    feat,
                    context_for_level(i),
                )
                for i, feat in enumerate(feats)
            ]
            if p2 is not None:
                p2_context = (
                    feats[0]
                    if self.cgsr_context_source == 'coarse'
                    else p2
                )
                if self.cgsr_context_intervention == 'zero':
                    p2_context = torch.zeros_like(p2_context)
                elif self.cgsr_context_intervention == 'shuffle':
                    if p2_context.shape[0] < 2:
                        raise ValueError(
                            'CGSR shuffle intervention requires an '
                            'evaluation batch with at least two images.'
                        )
                    p2_context = torch.roll(p2_context, shifts=1, dims=0)
                p2_feat = self.p2_cgsr(p2, p2_context)
                enhanced_feats[0] = enhanced_feats[0] + F.max_pool2d(
                    p2_feat, kernel_size=2, stride=2
                )
        else:
            enhanced_feats = feats
            if p2 is not None:
                enhanced_feats[0] = enhanced_feats[0] + F.max_pool2d(
                    p2, kernel_size=2, stride=2
                )
        
        # 2. Channel projection
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(enhanced_feats)]

        # Encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # Top-down FPN with CSAI.
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # Apply CSAI before addition.
            if self.fpn_csai is not None:
                csai_idx = len(self.in_channels) - 1 - idx
                upsample_feat = self.fpn_csai[csai_idx](feat_low, upsample_feat)
            
            fused_feat = (upsample_feat + feat_low) \
                if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN with CSAI.
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            
            # Apply CSAI before addition.
            if self.pan_csai is not None:
                downsample_feat = self.pan_csai[idx](feat_high, downsample_feat)
            
            fused_feat = (downsample_feat + feat_high) \
                if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_high], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs


# Backward-compatible import alias. New YAML files use SGDEIMEncoder.
HybridEncoder_ECSMF = SGDEIMEncoder
