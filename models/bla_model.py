

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .bridge_layers import (
    VisionAdapterBridge,
    PromptEnhancementBridge,
    AlignmentProjectionBridge,
    create_bridge_layers
)



class BLAMultiCapCLIP(nn.Module):
    """
    Bridge-Layer Architecture for Enhanced MultiCapCLIP
    
    This model enhances the original MultiCapCLIP with three strategic bridge layers:
    1. Vision Adapter Bridge: Adapts CLIP features for captioning
    2. Prompt Enhancement Bridge: Contextualizes prompts with visual information
    3. Alignment Projection Bridge: Aligns features for the decoder
    
    Args:
        clip_model: Pre-trained CLIP model (frozen)
        decoder: Multilingual language model decoder (trainable)
        concept_embeddings: Pre-computed concept prompt embeddings
        dim: Feature dimension (default: 768)
        hidden_dim: Hidden dimension for Bridge-1 (default: 1024)
        num_heads: Number of attention heads for Bridge-2 (default: 8)
        num_prompts: Number of prompts to retrieve (default: 16)
        dropout: Dropout rate (default: 0.1)
        adaptive: Whether to use adaptive routing (default: False)
    """
    
    def __init__(
        self,
        clip_model: nn.Module,
        decoder: nn.Module,
        concept_embeddings: torch.Tensor,
        dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_prompts: int = 16,
        dropout: float = 0.1,
        adaptive: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.dim = dim
        self.num_prompts = num_prompts
        self.adaptive = adaptive
        
        # Original MultiCapCLIP components
        self.clip_vision = clip_model.visual  # Frozen
        self.clip_text = clip_model.encode_text  # For concept embeddings
        self.decoder = decoder  # Trainable
        
        # Freeze CLIP
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        # Concept prompt embeddings (frozen)
        self.register_buffer('concept_embeddings', concept_embeddings)
        
        # Create bridge layers
        self.bridge1 = VisionAdapterBridge(
            input_dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.bridge2 = PromptEnhancementBridge(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        # Bridge-3 takes concatenated features (2*dim)
        self.bridge3 = AlignmentProjectionBridge(
            input_dim=dim * 2,
            output_dim=dim,
            dropout=dropout
        )
        
        if adaptive:
            from bridge_layers import AdaptiveBridge
            self.bridge1 = AdaptiveBridge(self.bridge1, dim)
            self.bridge2 = AdaptiveBridge(self.bridge2, dim)
            self.bridge3 = AdaptiveBridge(self.bridge3, dim * 2)
        
        # Projection to combine prompts and vision features
        self.feature_combiner = nn.Linear(dim * 2, dim)
        
    def retrieve_prompts(
        self, 
        features: torch.Tensor, 
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k concept prompts based on cosine similarity
        
        Args:
            features: Visual features [batch_size, dim]
            k: Number of prompts to retrieve (default: self.num_prompts)
            
        Returns:
            selected_prompts: [batch_size, k, dim]
            similarity_scores: [batch_size, k]
        """
        if k is None:
            k = self.num_prompts
        
        # Normalize features
        features_norm = F.normalize(features, dim=-1)
        concepts_norm = F.normalize(self.concept_embeddings, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(features_norm, concepts_norm.T)  # [batch, num_concepts]
        
        # Get top-k
        top_k_scores, top_k_indices = torch.topk(similarity, k, dim=-1)
        
        # Retrieve corresponding embeddings
        batch_size = features.size(0)
        selected_prompts = self.concept_embeddings[top_k_indices]  # [batch, k, dim]
        
        return selected_prompts, top_k_scores
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BLA-MultiCapCLIP
        
        Args:
            images: Input images [batch_size, 3, H, W] (for inference)
            text_features: Text features [batch_size, dim] (for training)
            target_ids: Target token IDs for decoder [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: Decoder output logits
                - loss: Training loss (if target_ids provided)
                - attention: Attention weights (if return_attention=True)
        """
        batch_size = images.size(0) if images is not None else text_features.size(0)
        
        # Step 1: Get visual or text features
        if images is not None:
            # Inference mode: use vision encoder
            with torch.no_grad():
                vision_features = self.clip_vision(images)  # [batch, seq_len, dim]
            
            # Apply Bridge-1: Vision Adapter
            if self.adaptive:
                adapted_features = self.bridge1(vision_features)[0]
            else:
                adapted_features = self.bridge1(vision_features)
            
            # Use global feature for prompt retrieval
            global_feature = adapted_features.mean(dim=1)  # [batch, dim]
            
        else:
            # Training mode: use text features
            adapted_features = text_features.unsqueeze(1)  # [batch, 1, dim]
            global_feature = text_features
        
        # Step 2: Retrieve concept prompts
        prompts, prompt_scores = self.retrieve_prompts(global_feature)  # [batch, k, dim]
        
        # Step 3: Apply Bridge-2: Prompt Enhancement
        if self.adaptive:
            enhanced_prompts, attention_weights = self.bridge2(
                prompts, adapted_features
            )[0], None  # Adaptive returns tuple
        else:
            enhanced_prompts, attention_weights = self.bridge2(
                prompts, adapted_features
            )
        
        # Step 4: Combine prompts and visual features
        # Average prompts
        prompt_feature = enhanced_prompts.mean(dim=1, keepdim=True)  # [batch, 1, dim]
        
        # Expand to match sequence length
        seq_len = adapted_features.size(1)
        prompt_feature = prompt_feature.expand(-1, seq_len, -1)
        
        # Concatenate
        combined = torch.cat([prompt_feature, adapted_features], dim=-1)  # [batch, seq_len, 2*dim]
        
        # Step 5: Apply Bridge-3: Alignment Projection
        if self.adaptive:
            aligned_features = self.bridge3(combined)[0]
        else:
            aligned_features = self.bridge3(combined)
        
        # Step 6: Decode
        decoder_outputs = self.decoder(
            encoder_hidden_states=aligned_features,
            labels=target_ids
        )
        
        # Prepare output
        output = {
            'logits': decoder_outputs.logits if hasattr(decoder_outputs, 'logits') else decoder_outputs,
            'prompt_scores': prompt_scores
        }
        
        if hasattr(decoder_outputs, 'loss') and decoder_outputs.loss is not None:
            output['loss'] = decoder_outputs.loss
        
        if return_attention and attention_weights is not None:
            output['attention'] = attention_weights
        
        return output
    
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 20,
        num_beams: int = 3,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate captions for images
        
        Args:
            images: Input images [batch_size, 3, H, W]
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            **kwargs: Additional arguments for decoder.generate()
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len]
        """
        # Get visual features through bridges
        with torch.no_grad():
            vision_features = self.clip_vision(images)
            adapted_features = self.bridge1(vision_features)
            global_feature = adapted_features.mean(dim=1)
            
            # Retrieve and enhance prompts
            prompts, _ = self.retrieve_prompts(global_feature)
            enhanced_prompts, _ = self.bridge2(prompts, adapted_features)
            
            # Combine and align
            seq_len = adapted_features.size(1)
            prompt_feature = enhanced_prompts.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            combined = torch.cat([prompt_feature, adapted_features], dim=-1)
            aligned_features = self.bridge3(combined)
        
        # Generate using decoder
        generated_ids = self.decoder.generate(
            encoder_outputs={'last_hidden_state': aligned_features},
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            **kwargs
        )
        
        return generated_ids
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (bridges + decoder)"""
        trainable = []
        trainable.extend(self.bridge1.parameters())
        trainable.extend(self.bridge2.parameters())
        trainable.extend(self.bridge3.parameters())
        trainable.extend(self.decoder.parameters())
        trainable.extend(self.feature_combiner.parameters())
        return trainable
    
    def get_bridge_params(self) -> List[nn.Parameter]:
        """Get list of bridge parameters only"""
        bridge_params = []
        bridge_params.extend(self.bridge1.parameters())
        bridge_params.extend(self.bridge2.parameters())
        bridge_params.extend(self.bridge3.parameters())
        return bridge_params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different components"""
        return {
            'clip_vision': sum(p.numel() for p in self.clip_vision.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters() if p.requires_grad),
            'bridge1': self.bridge1.get_num_params() if hasattr(self.bridge1, 'get_num_params') else sum(p.numel() for p in self.bridge1.parameters()),
            'bridge2': self.bridge2.get_num_params() if hasattr(self.bridge2, 'get_num_params') else sum(p.numel() for p in self.bridge2.parameters()),
            'bridge3': self.bridge3.get_num_params() if hasattr(self.bridge3, 'get_num_params') else sum(p.numel() for p in self.bridge3.parameters()),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    print("Testing BLA-MultiCapCLIP Model...")
    print("=" * 60)
    
    # Create dummy components for testing
    class DummyCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = DummyVisualEncoder()
        
        def encode_text(self, text):
            return torch.randn(len(text), 768)
    
    class DummyVisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 768, 16, 16)
            self.proj = nn.Linear(768, 768)
        
        def forward(self, x):
            # x: [batch, 3, 224, 224]
            x = self.conv(x)  # [batch, 768, 14, 14]
            x = x.flatten(2).transpose(1, 2)  # [batch, 196, 768]
            x = self.proj(x)  # [batch, 196, 768]
            return x
    
    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(768, 50000)
        
        def forward(self, encoder_hidden_states, labels=None):
            batch_size, seq_len, _ = encoder_hidden_states.shape
            # Take mean over sequence for simplicity in dummy
            pooled = encoder_hidden_states.mean(dim=1)  # [batch, dim]
            logits = self.lm_head(pooled).unsqueeze(1).expand(-1, labels.size(1) if labels is not None else 20, -1)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, 50000),
                    labels.reshape(-1),
                    ignore_index=-100
                )
            
            class Output:
                pass
            output = Output()
            output.logits = logits
            output.loss = loss
            return output
        
        def generate(self, encoder_outputs, max_length=20, **kwargs):
            batch_size = encoder_outputs['last_hidden_state'].size(0)
            return torch.randint(0, 50000, (batch_size, max_length))
    
    # Create model
    clip_model = DummyCLIP()
    decoder = DummyDecoder()
    concept_embeddings = torch.randn(1000, 768)  # 1000 concepts
    
    model = BLAMultiCapCLIP(
        clip_model=clip_model,
        decoder=decoder,
        concept_embeddings=concept_embeddings,
        dim=768,
        num_prompts=16
    )
    
    print("\n1. Model Architecture:")
    print(f"   - CLIP Vision: Frozen")
    print(f"   - Bridge Layers: 3 modules")
    print(f"   - Decoder: Trainable")
    
    print("\n2. Parameter Count:")
    params = model.count_parameters()
    for name, count in params.items():
        print(f"   - {name}: {count:,}")
    
    print("\n3. Testing Forward Pass (Training Mode):")
    text_features = torch.randn(4, 768)
    target_ids = torch.randint(0, 50000, (4, 20))
    outputs = model(text_features=text_features, target_ids=target_ids)
    print(f"   - Input: {text_features.shape}")
    print(f"   - Output logits: {outputs['logits'].shape}")
    print(f"   - Loss: {outputs['loss'].item():.4f}")
    
    print("\n4. Testing Forward Pass (Inference Mode):")
    images = torch.randn(4, 3, 224, 224)
    outputs = model(images=images, return_attention=True)
    print(f"   - Input: {images.shape}")
    print(f"   - Output logits: {outputs['logits'].shape}")
    if 'attention' in outputs:
        print(f"   - Attention: {outputs['attention'].shape}")
    
    print("\n5. Testing Generation:")
    generated = model.generate(images, max_length=15, num_beams=3)
    print(f"   - Generated IDs: {generated.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
