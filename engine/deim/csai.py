"""
CSAI: Cross-Scale Adaptive Interaction
======================================
This file implements the paper's CSAI module for adjacent-scale fusion before
FPN/PAN addition.

Historical import aliases are retained for compatibility. Functionally, the
module follows the paper description:
1. Bidirectional sequence mixing with lightweight depthwise 1D convolutions.
2. Context-conditioned cross-scale gated modulation.
3. Learnable balancing and channel fusion of the two directions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAIBidirectionalMixer(nn.Module):
    """
    CSAI bidirectional sequence mixing.

    It captures forward and backward spatial-sequence context with depthwise
    1D convolutions instead of global attention or full SSM scanning.
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        
        # Forward sequence context.
        self.conv_fwd = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        # Backward sequence context.
        self.conv_bwd = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        
        # Feature mixing
        self.mix = nn.Linear(dim * 2, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D] - sequence features
        Returns:
            y: [B, L, D] - bidirectionally enhanced features
        """
        B, L, D = x.shape
        
        # Process in both directions
        x_t = x.transpose(1, 2)  # [B, D, L]
        
        # Forward
        y_fwd = self.conv_fwd(x_t)  # [B, D, L]
        # Backward (flip, conv, flip)
        y_bwd = self.conv_bwd(torch.flip(x_t, dims=[-1]))
        y_bwd = torch.flip(y_bwd, dims=[-1])  # [B, D, L]
        
        # Combine
        y = torch.cat([y_fwd, y_bwd], dim=1)  # [B, 2D, L]
        y = y.transpose(1, 2)  # [B, L, 2D]
        y = self.mix(y)  # [B, L, D]
        y = self.act(self.norm(y + x))  # Residual + activation
        
        return y


class CrossScaleAdaptiveGate(nn.Module):
    """
    CSAI cross-scale gated modulation.

    The context scale generates a gate that modulates the value projection of
    the current scale before residual write-back.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 32)
        
        # Gate generation from the adjacent-scale context.
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
        
        # Value projection for the scale being modulated.
        self.value_proj = nn.Linear(dim, dim, bias=False)
        
        # Paper alpha/gamma-style residual scale; initialized at zero.
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, context):
        """
        Args:
            x: [B, L, D] - features to be gated
            context: [B, L, D] - context features for gate generation
        Returns:
            y: [B, L, D] - gated features
        """
        # Generate the cross-scale gate from context.
        gate = self.gate_proj(context)
        
        # Modulate the current scale's value projection.
        v = self.value_proj(x)
        y = v * gate
        
        # Residual write-back keeps early training close to identity.
        return x + self.gamma * y


class CSAIDirectionalInteraction(nn.Module):
    """
    One CSAI direction: sequence mixing followed by cross-scale gating.

    The class name is retained for compatibility with earlier experiments.
    """
    def __init__(self, dim, kernel_size=5, reduction=4):
        super().__init__()
        
        # Within-scale bidirectional sequence mixing.
        self.bi_conv = CSAIBidirectionalMixer(dim, kernel_size)
        
        # Cross-scale gated modulation.
        self.cross_gate = CrossScaleAdaptiveGate(dim, reduction)
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, context):
        """
        Args:
            x: [B, L, D] - main features
            context: [B, L, D] - cross-scale context
        Returns:
            y: [B, L, D] - enhanced features
        """
        # Bidirectional sequence mixing.
        x_bi = self.bi_conv(x)
        
        # Context-conditioned gated modulation.
        y = self.cross_gate(x_bi, context)
        
        return self.norm(y)


class CrossScaleAdaptiveInteraction(nn.Module):
    """
    Efficient CSAI implementation.
    
    This is the main CSAI module used at FPN/PAN fusion nodes. It computes
    both low-to-high and high-to-low interactions, then fuses the two directions
    with a learnable balance.
    """
    def __init__(self, dim, kernel_size=5, reduction=4, use_bidirectional=True):
        super().__init__()
        self.dim = dim
        self.use_bidirectional = use_bidirectional
        
        # Bidirectional cross-scale interactions.
        self.cross_high_low = CSAIDirectionalInteraction(dim, kernel_size, reduction)
        self.cross_low_high = CSAIDirectionalInteraction(dim, kernel_size, reduction)
        
        # Output fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        
        # Learnable balance between the two interaction directions.
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, feat_high, feat_low):
        """
        Args:
            feat_high: [B, C, H, W] - higher resolution feature
            feat_low: [B, C, H, W] - lower resolution feature
        Returns:
            fused: [B, C, H, W] - fused feature
        """
        B, C, H, W = feat_high.shape
        
        # Ensure alignment
        if feat_high.shape[2:] != feat_low.shape[2:]:
            feat_low = F.interpolate(feat_low, size=(H, W), mode='nearest')
        
        # Flatten to sequence
        high_flat = feat_high.flatten(2).transpose(1, 2)  # [B, H*W, C]
        low_flat = feat_low.flatten(2).transpose(1, 2)    # [B, H*W, C]
        
        # Cross-scale adaptive interactions in both directions.
        high_enhanced = self.cross_low_high(high_flat, low_flat)
        low_enhanced = self.cross_high_low(low_flat, high_flat)
        
        # Reshape back
        high_out = high_enhanced.transpose(1, 2).view(B, C, H, W)
        low_out = low_enhanced.transpose(1, 2).view(B, C, H, W)
        
        # Learnable directional balance followed by channel fusion.
        alpha = torch.sigmoid(self.alpha)
        fused = torch.cat([high_out * alpha, low_out * (1 - alpha)], dim=1)
        fused = self.fusion(fused)
        
        return fused


class LightweightCrossScaleAdaptiveInteraction(nn.Module):
    """
    Ultra-lightweight CSAI variant using only 2D convolutions and gating.

    The class name is retained for compatibility.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        # Shared depthwise conv for local context
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        
        # Cross-scale gate generation.
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        
        # Residual scale
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, feat_high, feat_low):
        """Simple and fast cross-scale adaptive interaction."""
        # Ensure alignment
        if feat_high.shape[2:] != feat_low.shape[2:]:
            feat_low = F.interpolate(feat_low, size=feat_high.shape[2:], mode='nearest')
        
        # Local context enhancement
        high_ctx = self.dwconv(feat_high)
        low_ctx = self.dwconv(feat_low)
        
        # Cross-scale gated modulation.
        combined = torch.cat([high_ctx, low_ctx], dim=1)
        gate = self.gate_conv(combined)
        
        # Gated fusion of the two adjacent scales.
        fused = high_ctx * gate + low_ctx * (1 - gate)
        fused = self.proj(fused)
        
        # Residual addition from high-res features
        return feat_high + self.scale * fused


# Backward-compatible aliases for checkpoints and external imports created
# before the paper terminology was finalized.
EfficientBidirectionalConv = CSAIBidirectionalMixer
CrossScaleGating = CrossScaleAdaptiveGate
EfficientCrossMamba = CSAIDirectionalInteraction
EfficientCSMF = CrossScaleAdaptiveInteraction
LightweightCSMF_V2 = LightweightCrossScaleAdaptiveInteraction


__all__ = [
    'CSAIBidirectionalMixer',
    'CrossScaleAdaptiveGate',
    'CSAIDirectionalInteraction',
    'CrossScaleAdaptiveInteraction',
    'LightweightCrossScaleAdaptiveInteraction',
    'EfficientBidirectionalConv',
    'CrossScaleGating', 
    'EfficientCrossMamba',
    'EfficientCSMF',
    'LightweightCSMF_V2'
]
