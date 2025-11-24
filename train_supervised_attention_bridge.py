import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Use torch.cuda.amp for mixed precision (stable on Colab)
from torch.cuda.amp import autocast, GradScaler
import clip
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    data_root = Path("/content/drive/MyDrive/MultiCapCLIP/data/MSCOCO")
    train_images_dir = data_root / "train2014"
    val_images_dir = data_root / "val2014"

    # train
    train_annotations = data_root / "annotations/captions_train2014_subset_clean.json"
    val_annotations = data_root / "annotations/captions_val2014.json"

    output_dir = Path("/content/drive/MyDrive/MultiCapCLIP/supervised_attention_bridge")
    
    # Model parameters
    clip_dim = 512
    mbart_dim = 1024
    num_bridge_tokens = 32
    bridge_layers = 4
    bridge_heads = 8      # Number of attention heads
    bridge_ff_dim = 2048  # Feed-forward dimension
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 3000
    gradient_clip_norm = 1.0
    
    # Logging
    log_interval = 100
    eval_interval = 1000
    save_interval = 2000
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
    
    # Data
    max_caption_length = 128
    num_workers = 2
    
    # Language
    target_lang = "en_XX"  # English

# ============================================================================
# Multi-Layer Attention Bridge Network
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with multi-head attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class AttentionBridge(nn.Module):
    """
    Multi-layer attention bridge that processes CLIP patch features.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection from CLIP to mBART dimension
        self.input_proj = nn.Linear(config.clip_dim, config.mbart_dim)
        
        # Multi-layer Transformer encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.mbart_dim,
                nhead=config.bridge_heads,
                dim_feedforward=config.bridge_ff_dim,
                dropout=config.dropout
            )
            for _ in range(config.bridge_layers)
        ])
        
        # Learnable query tokens for cross-attention pooling
        self.query_tokens = nn.Parameter(torch.randn(1, config.num_bridge_tokens, config.mbart_dim))
        self.cross_attn = nn.MultiheadAttention(
            config.mbart_dim, 
            config.bridge_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(config.mbart_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with smaller values to prevent gradient explosion."""
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
    
    def forward(self, clip_features):
        batch_size = clip_features.shape[0]
        
        # Project to mBART dimension
        x = self.input_proj(clip_features)
        
        # Apply multi-layer Transformer
        for layer in self.layers:
            x = layer(x)
        
        # Cross-attention pooling with learnable queries
        queries = self.query_tokens.expand(batch_size, -1, -1)
        output, _ = self.cross_attn(queries, x, x)
        
        # Output normalization
        output = self.output_norm(output)
        
        return output

# ============================================================================
# Dataset
# ============================================================================

class COCOCaptionDataset(Dataset):
    
    def __init__(self, images_dir, annotations_file, clip_preprocess, tokenizer, config, split='train'):
        self.images_dir = Path(images_dir)
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.image_id_to_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_captions:
                self.image_id_to_captions[img_id] = []
            self.image_id_to_captions[img_id].append(ann['caption'])
        
        self.image_id_to_filename = {}
        for img in data['images']:
            self.image_id_to_filename[img['id']] = img['file_name']
        
        self.samples = []
        for img_id, captions in self.image_id_to_captions.items():
            if img_id in self.image_id_to_filename:
                if split == 'train':
                    for caption in captions:
                        self.samples.append((img_id, caption))
                else:
                    self.samples.append((img_id, captions[0]))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        filename = self.image_id_to_filename[img_id]
        img_path = self.images_dir / filename
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.clip_preprocess(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        caption_with_lang = f"{caption}"
        encoded = self.tokenizer(
            caption_with_lang,
            max_length=self.config.max_caption_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'caption': caption
        }

# ============================================================================
# Training Functions
# ============================================================================

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(bridge, clip_model, mbart_model, tokenizer, train_loader, optimizer, scheduler, scaler, config, epoch, global_step, csv_writer, csv_file):
    """Train for one epoch."""
    bridge.train()
    clip_model.eval()
    mbart_model.eval()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=config.use_amp):
            # Extract CLIP patch features (frozen)
            with torch.no_grad():
                if hasattr(clip_model.visual, 'transformer'):
                    x = clip_model.visual.conv1(images.type(clip_model.dtype))
                    x = x.reshape(x.shape[0], x.shape[1], -1)
                    x = x.permute(0, 2, 1)
                    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                    x = x + clip_model.visual.positional_embedding.to(x.dtype)
                    x = clip_model.visual.ln_pre(x)
                    x = x.permute(1, 0, 2)
                    x = clip_model.visual.transformer(x)
                    x = x.permute(1, 0, 2)
                    clip_patch_features = x.float()
                    clip_patch_features = clip_patch_features.to(clip_model.visual.proj.dtype)
                    clip_patch_features = clip_patch_features @ clip_model.visual.proj.float()
                else:
                    clip_features = clip_model.encode_image(images)
                    clip_patch_features = clip_features.unsqueeze(1).repeat(1, 50, 1).float()
            
            # Bridge network (trainable)
            bridge_output = bridge(clip_patch_features)
            
            # Prepare decoder inputs
            decoder_input_ids = mbart_model.prepare_decoder_input_ids_from_labels(input_ids)
            
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=bridge_output)
            
            # mBART decoder forward
            outputs = mbart_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                return_dict=True
            )
            
            # Calculate loss
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
        
        if not torch.isfinite(loss):
            print(f"\nWarning: Non-finite loss detected at step {global_step}: {loss.item()}. Skipping batch.")
            optimizer.zero_grad()
            continue
        
        # --- ENHANCEMENT: STABLE GRADIENT UPDATE --- #
        # 1. Scale loss and backpropagate
        scaler.scale(loss).backward()
        
        # 2. Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # 3. Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), config.gradient_clip_norm)
        
        # 4. Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update()
        # ------------------------------------------- #
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        avg_loss = total_loss / num_batches
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        if batch_idx % config.log_interval == 0:
            csv_writer.writerow({
                'epoch': epoch,
                'step': global_step,
                'loss': loss.item(),
                'avg_loss': avg_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            csv_file.flush()
        
        if global_step % config.save_interval == 0:
            checkpoint_path = config.output_dir / "checkpoints" / f"step_{global_step}.pt"
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'bridge_state_dict': bridge.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    return total_loss / num_batches, global_step


@torch.no_grad()
def validate(bridge, clip_model, mbart_model, tokenizer, val_loader, config):
    """Validate the model."""
    bridge.eval()
    clip_model.eval()
    mbart_model.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        with autocast(enabled=config.use_amp):
            x = clip_model.visual.conv1(images.type(clip_model.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + clip_model.visual.positional_embedding.to(x.dtype)
            x = clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            clip_patch_features = x.float()
            clip_patch_features = clip_patch_features.to(clip_model.visual.proj.dtype)
            clip_patch_features = clip_patch_features @ clip_model.visual.proj.float()
            
            bridge_output = bridge(clip_patch_features)
            
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=bridge_output)
            
            decoder_input_ids = mbart_model.prepare_decoder_input_ids_from_labels(input_ids)
            outputs = mbart_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    config = Config()
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoints").mkdir(exist_ok=True)
    
    print("="*80)
    print("Supervised Image Captioning Training ")
    print("Multi-Layer Attention Bridge Architecture")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Bridge layers: {config.bridge_layers}")
    print(f"Bridge tokens: {config.num_bridge_tokens}")
    print(f"AMP Enabled: {config.use_amp}")
    print(f"Gradient Clip Norm: {config.gradient_clip_norm}")
    print("="*80)
    
    print("\nLoading models...")
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=config.device)
    clip_model = clip_model.float()
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(config.device)
    mbart_model.eval()
    for param in mbart_model.parameters():
        param.requires_grad = False
    
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    tokenizer.src_lang = config.target_lang
    tokenizer.tgt_lang = config.target_lang
    
    print("Initializing bridge network...")
    bridge = AttentionBridge(config).to(config.device)
    num_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(f"Bridge parameters: {num_params:,}")
    
    print("\nLoading datasets...")
    train_dataset = COCOCaptionDataset(
        config.train_images_dir,
        config.train_annotations,
        clip_preprocess,
        tokenizer,
        config,
        split='train'
    )
    val_dataset = COCOCaptionDataset(
        config.val_images_dir,
        config.val_annotations,
        clip_preprocess,
        tokenizer,
        config,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    
    csv_path = config.output_dir / "training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'step', 'loss', 'avg_loss', 'learning_rate'])
    csv_writer.writeheader()
    
    print("\nStarting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*80}")
        
        train_loss, global_step = train_epoch(
            bridge, clip_model, mbart_model, tokenizer,
            train_loader, optimizer, scheduler, scaler,
            config, epoch, global_step, csv_writer, csv_file
        )
        
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
        
        val_loss, val_ppl = validate(
            bridge, clip_model, mbart_model, tokenizer,
            val_loader, config
        )
        
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = config.output_dir / "checkpoints" / "best.pt"
            torch.save({
                'epoch': epoch,
                'step': global_step,
                'bridge_state_dict': bridge.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, best_path)
            print(f"Saved best model to {best_path}")
        
        epoch_path = config.output_dir / "checkpoints" / f"epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'step': global_step,
            'bridge_state_dict': bridge.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
        }, epoch_path)
    
    csv_file.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Logs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
