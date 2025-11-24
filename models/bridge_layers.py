"""
Bridge Layer Modules for Enhanced MultiCapCLIP
Author: Manus AI
Date: October 30, 2025

This module implements the three strategic bridge layers:
1. Vision Adapter Bridge (Bridge-1)
2. Prompt Enhancement Bridge (Bridge-2)
3. Alignment Projection Bridge (Bridge-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VisionAdapterBridge(nn.Module):
    """
    Bridge-1: Vision Adapter Layer
    
    Adapts frozen CLIP visual features to be more suitable for captioning task.
    Uses a lightweight MLP with residual connection to preserve original CLIP features.
    
    Args:
        input_dim (int): Dimension of input features (default: 768 for CLIP ViT-B/16)
        hidden_dim (int): Dimension of hidden layer (default: 1024)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self, 
        input_dim: int = 768,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Adapter network: 2-layer MLP
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Learnable scaling factor for residual connection
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Adapter Bridge
        
        Args:
            vision_features: Tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            adapted_features: Tensor of shape [batch_size, seq_len, input_dim]
        """
        # Apply adapter with residual connection
        adapted = self.adapter(vision_features)
        
        # Scaled residual connection
        output = vision_features + self.scale * adapted
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output
    
    def get_num_params(self) -> int:
        """Return number of parameters in this module"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PromptEnhancementBridge(nn.Module):
    """
    Bridge-2: Prompt Enhancement Layer
    
    Enriches retrieved concept prompts with visual context through cross-attention.
    Allows prompts to be contextualized based on specific visual content.
    
    Args:
        dim (int): Feature dimension (default: 768)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention: prompts attend to vision features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        prompts: torch.Tensor, 
        vision_features: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Prompt Enhancement Bridge
        
        Args:
            prompts: Tensor of shape [batch_size, num_prompts, dim]
            vision_features: Tensor of shape [batch_size, seq_len, dim]
            prompt_mask: Optional mask for prompts
            
        Returns:
            enhanced_prompts: Tensor of shape [batch_size, num_prompts, dim]
            attention_weights: Tensor of shape [batch_size, num_prompts, seq_len]
        """
        # Cross-attention: prompts attend to vision
        enhanced, attn_weights = self.cross_attn(
            query=prompts,
            key=vision_features,
            value=vision_features,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Residual connection + normalization
        prompts = self.norm1(prompts + self.dropout(enhanced))
        
        # Feed-forward network
        ffn_output = self.ffn(prompts)
        
        # Residual connection + normalization
        prompts = self.norm2(prompts + self.dropout(ffn_output))
        
        return prompts, attn_weights
    
    def get_num_params(self) -> int:
        """Return number of parameters in this module"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlignmentProjectionBridge(nn.Module):
    """
    Bridge-3: Alignment Projection Layer
    
    Projects combined visual and prompt features into decoder's optimal space.
    Uses gated projection for adaptive alignment.
    
    Args:
        input_dim (int): Dimension of input features (default: 768)
        output_dim (int): Dimension of output features (default: 768)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Projection network
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Gating mechanism for adaptive alignment
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Alignment Projection Bridge
        
        Args:
            features: Tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            aligned_features: Tensor of shape [batch_size, seq_len, output_dim]
        """
        # Project features
        projected = self.projection(features)
        
        # Compute gate values
        gate_values = self.gate(features)
        
        # Gated projection
        gated = projected * gate_values
        
        # Apply dropout and layer norm
        output = self.layer_norm(self.dropout(gated))
        
        return output
    
    def get_num_params(self) -> int:
        """Return number of parameters in this module"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveBridge(nn.Module):
    """
    Adaptive variant of bridge layers with learnable routing.
    
    This wrapper adds a routing mechanism that determines how much
    each bridge should contribute based on the input.
    
    Args:
        base_bridge (nn.Module): The base bridge module to wrap
        dim (int): Feature dimension
    """
    
    def __init__(self, base_bridge: nn.Module, dim: int = 768):
        super().__init__()
        
        self.bridge = base_bridge
        
        # Routing network
        self.router = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, *args, **kwargs):
        """
        Forward pass with adaptive routing
        
        Returns:
            output: Adaptively weighted output
            weight: Routing weight (for analysis)
        """
        # Get input features (first argument)
        features = args[0]
        
        # Compute routing weight based on input
        # Use mean pooling over sequence dimension
        if features.dim() == 3:
            pooled = features.mean(dim=1)
        else:
            pooled = features
            
        weight = self.router(pooled).unsqueeze(1)  # [batch_size, 1, 1]
        
        # Apply bridge
        if isinstance(self.bridge, PromptEnhancementBridge):
            bridged, attn = self.bridge(*args, **kwargs)
            # Apply routing weight
            output = args[0] + weight * (bridged - args[0])
            return output, attn
        else:
            bridged = self.bridge(*args, **kwargs)
            # Apply routing weight
            output = args[0] + weight * (bridged - args[0])
            return output
    
    def get_num_params(self) -> int:
        """Return number of parameters including router"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Utility function to create all bridges at once
def create_bridge_layers(
    dim: int = 768,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    dropout: float = 0.1,
    adaptive: bool = False
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Factory function to create all three bridge layers
    
    Args:
        dim: Feature dimension
        hidden_dim: Hidden dimension for Bridge-1
        num_heads: Number of attention heads for Bridge-2
        dropout: Dropout rate
        adaptive: Whether to use adaptive routing
        
    Returns:
        bridge1, bridge2, bridge3: The three bridge modules
    """
    bridge1 = VisionAdapterBridge(
        input_dim=dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    bridge2 = PromptEnhancementBridge(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout
    )
    
    bridge3 = AlignmentProjectionBridge(
        input_dim=dim,
        output_dim=dim,
        dropout=dropout
    )
    
    if adaptive:
        bridge1 = AdaptiveBridge(bridge1, dim)
        bridge2 = AdaptiveBridge(bridge2, dim)
        bridge3 = AdaptiveBridge(bridge3, dim)
    
    return bridge1, bridge2, bridge3


if __name__ == "__main__":
    # Test the bridge layers
    print("Testing Bridge Layers...")
    print("=" * 50)
    
    batch_size = 4
    seq_len = 49  # 7x7 patches for ViT-B/16
    num_prompts = 16
    dim = 768
    
    # Create dummy inputs
    vision_features = torch.randn(batch_size, seq_len, dim)
    prompts = torch.randn(batch_size, num_prompts, dim)
    
    # Test Bridge-1
    print("\n1. Testing Vision Adapter Bridge...")
    bridge1 = VisionAdapterBridge(input_dim=dim)
    adapted = bridge1(vision_features)
    print(f"   Input shape: {vision_features.shape}")
    print(f"   Output shape: {adapted.shape}")
    print(f"   Parameters: {bridge1.get_num_params():,}")
    
    # Test Bridge-2
    print("\n2. Testing Prompt Enhancement Bridge...")
    bridge2 = PromptEnhancementBridge(dim=dim)
    enhanced, attn = bridge2(prompts, vision_features)
    print(f"   Prompt shape: {prompts.shape}")
    print(f"   Vision shape: {vision_features.shape}")
    print(f"   Output shape: {enhanced.shape}")
    print(f"   Attention shape: {attn.shape}")
    print(f"   Parameters: {bridge2.get_num_params():,}")
    
    # Test Bridge-3
    print("\n3. Testing Alignment Projection Bridge...")
    combined = torch.cat([enhanced.mean(dim=1, keepdim=True).expand(-1, seq_len, -1), 
                          adapted], dim=-1)
    bridge3 = AlignmentProjectionBridge(input_dim=dim*2, output_dim=dim)
    aligned = bridge3(combined)
    print(f"   Input shape: {combined.shape}")
    print(f"   Output shape: {aligned.shape}")
    print(f"   Parameters: {bridge3.get_num_params():,}")
    
    # Test factory function
    print("\n4. Testing Factory Function...")
    b1, b2, b3 = create_bridge_layers(dim=dim)
    total_params = b1.get_num_params() + b2.get_num_params() + b3.get_num_params()
    print(f"   Total parameters: {total_params:,}")
    
    # Test adaptive bridges
    print("\n5. Testing Adaptive Bridges...")
    b1_adp, b2_adp, b3_adp = create_bridge_layers(dim=dim, adaptive=True)
    total_params_adp = (b1_adp.get_num_params() + 
                        b2_adp.get_num_params() + 
                        b3_adp.get_num_params())
    print(f"   Total parameters (adaptive): {total_params_adp:,}")
    print(f"   Additional parameters: {total_params_adp - total_params:,}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
