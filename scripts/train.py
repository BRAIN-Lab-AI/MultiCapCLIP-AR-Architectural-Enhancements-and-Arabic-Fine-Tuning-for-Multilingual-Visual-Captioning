
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # تحمّل الصور المقطوعة بدل ما تسقط

# إضافة مسار المجلد models
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from bla_model import BLAMultiCapCLIP  # يفترض وجوده لديك

# -------------------------------------------------
# دالة توحيد مفاتيح الباتش (صور/سمات + لابلز)
# -------------------------------------------------
def _extract_inputs(batch, device):
    """
    يُحول أي batch لصيغة موحّدة:
      inputs: image أو text_features
      labels: target_ids
    ويدعم مفاتيح شائعة: pixel_values/image/images/text_features + labels/target_ids/caption_ids/input_ids
    """
    # labels
    for yk in ("target_ids", "labels", "caption_ids", "input_ids"):
        if yk in batch:
            target_ids = batch[yk].to(device)
            break
    else:
        raise KeyError("لم أجد labels/target_ids/caption_ids/input_ids داخل batch.")

    # features أو صور
    if "text_features" in batch:
        return {"text_features": batch["text_features"].to(device), "target_ids": target_ids}

    for xk in ("pixel_values", "image", "images", "pixel"):
        if xk in batch:
            return {"image": batch[xk].to(device), "target_ids": target_ids}

    raise KeyError("لم أجد pixel_values/image/images ولا text_features داخل batch.")

# -------------------------------------------------
# COCO Dataset (مختصر داخل الملف)
# -------------------------------------------------
import torchvision.transforms as T
from transformers import AutoTokenizer

class CocoCapsDataset(torch.utils.data.Dataset):
    """
    داتاست مبسطة لكابتشن COCO 2014:
      - images_dir: مجلد الصور (train2014/val2014)
      - ann_json: ملف أنوتيشن (captions_*.json)
    تُرجع:
      {"pixel_values": Tensor[3,H,W], "target_ids": Tensor[seq_len]}
    """
    def __init__(self, images_dir, ann_json, tokenizer_name="bert-base-uncased",
                 max_len=20, image_size=224):
        import json as _json

        if not os.path.isfile(ann_json):
            raise FileNotFoundError(f"Annotation JSON غير موجود: {ann_json}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"مجلد الصور غير موجود: {images_dir}")

        with open(ann_json, "r") as f:
            ann = _json.load(f)

        # map: image_id -> file_name
        id2name = {img["id"]: img["file_name"] for img in ann["images"]}

        # كل تعليق يُعدّ عيّنة (صورة + نص)
        samples = []
        miss_cnt = 0
        for a in ann["annotations"]:
            img_id = a["image_id"]
            if img_id not in id2name:
                continue
            p = os.path.join(images_dir, id2name[img_id])
            if os.path.isfile(p):
                samples.append((p, a.get("caption", "")))
            else:
                miss_cnt += 1

        self.samples = samples
        if miss_cnt:
            print(f"[COCO] تم تخطي {miss_cnt} عنصر لعدم وجود الصورة فعلياً.")

        self.tx  = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tok.pad_token is None:
            # تعيين pad token إذا مفقود
            self.tok.pad_token = self.tok.eos_token or self.tok.sep_token or "[PAD]"
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cap = self.samples[i]
        # فتح الصورة (قد تكون ناقصة — لدينا LOAD_TRUNCATED_IMAGES=True)
        img = Image.open(path).convert("RGB")
        px  = self.tx(img)  # [3,H,W]

        tok = self.tok(
            cap,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "pixel_values": px,                         # صور
            "target_ids":  tok["input_ids"].squeeze(0)  # لابلز
        }

def build_coco_dataloaders(config: Dict):
    """بناء DataLoader للتدريب والتحقق من config['data']"""
    data = config.get("data", {})
    train_images = data["train_images"]
    val_images   = data["val_images"]
    train_ann    = data["train_ann"]
    val_ann      = data["val_ann"]

    max_len    = data.get("max_caption_length", 20)
    image_size = data.get("image_size", 224)
    # Colab يشتكي من عدد وركرز كبير — نخليه صغير
    num_workers = int(data.get("num_workers", 0))
    if num_workers > 2:
        num_workers = 2

    # batch_size: من الجذر أو من قسم training
    batch_size = int(config.get("batch_size", config.get("training", {}).get("batch_size", 32)))
    tok_name   = config.get("tokenizer_name", "bert-base-uncased")

    print("\nBuilding COCO dataloaders...")
    print("[COCO] Using paths:")
    print(f" train_images: {train_images} | exists: {os.path.isdir(train_images)}")
    print(f" val_images:   {val_images}   | exists: {os.path.isdir(val_images)}")
    print(f" train_ann:    {train_ann} | exists: {os.path.isfile(train_ann)}")
    print(f" val_ann:      {val_ann}   | exists: {os.path.isfile(val_ann)}")

    train_ds = CocoCapsDataset(train_images, train_ann, tok_name, max_len, image_size)
    val_ds   = CocoCapsDataset(val_images,   val_ann,   tok_name, max_len, image_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False
    )
    return train_loader, val_loader

# -------------------------------------------------
# BLATrainer (مع دعم الاستكمال)
# -------------------------------------------------
class BLATrainer:
    """
    Trainer class for BLA-MultiCapCLIP

    يدعم:
      1) bridge_pretrain  (تدريب الجسور فقط)
      2) end_to_end       (تدريب كامل)
      3) Resume from checkpoint (استكمال التدريب)
    """
    def __init__(
        self,
        model: BLAMultiCapCLIP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # إخراج
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

        # حالة التدريب
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 1  # للاستكمال

        # إسقاط احتياطي للصورة -> 768 (للنماذج التي لا تملك image=...)
        # (AdaptiveAvgPool2d → Flatten → Linear(3->768))
        self.fallback_img_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 768)
        ).to(device)

    def setup_optimizer(self, phase: str = 'end_to_end'):
        """تهيئة الأوبتميزر (حسب المرحلة)"""
        if phase == 'bridge_pretrain':
            params = self.model.get_bridge_params()
            print(f"Bridge pre-training: {sum(p.numel() for p in params):,} parameters")
        else:
            params = self.model.get_trainable_params()
            print(f"End-to-end training: {sum(p.numel() for p in params):,} parameters")

        opt_name = self.config.get('optimizer', 'adamw').lower()
        if opt_name == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif opt_name == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Scheduler
        if self.config.get('use_scheduler', True):
            steps_per_epoch = max(1, len(self.train_loader))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'] * steps_per_epoch,
                eta_min=self.config['learning_rate'] * 0.1
            )
        else:
            self.scheduler = None

    def load_checkpoint(self, checkpoint_path: str):
        """
        تحميل checkpoint لاستكمال التدريب
        
        Args:
            checkpoint_path: مسار ملف الـ checkpoint
        """
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model state from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state")
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Loaded scheduler state")
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✓ Resuming from epoch {self.start_epoch}")
        print(f"✓ Global step: {self.global_step}")
        print(f"✓ Best validation loss so far: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return checkpoint

    def _forward_model(self, inp: Dict):
        """
        يحاول استدعاء الموديل بشكل مرن:
          - أولاً: image=... إن كان مدعوماً
          - وإلا: يسقط الصورة إلى text_features (3→768) ويمررها كـ text_features
        يجب أن يعيد كائن فيه outputs['loss'].
        """
        y = inp["target_ids"]
        if "text_features" in inp:
            return self.model(text_features=inp["text_features"], target_ids=y)

        # نملك صورة فقط
        img = inp["image"]
        # نحاول image=...
        try:
            return self.model(image=img, target_ids=y)
        except TypeError:
            # fallback: إسقاط بسيط للصورة → 768 ثم نمررها كأنها text_features
            # img: [B,3,H,W] → [B,3] → Linear(3,768)
            tf = self.fallback_img_proj(img)
            return self.model(text_features=tf, target_ids=y)

    def train_epoch(self, epoch: int) -> float:
        """تدريب لحقبة واحدة"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            inp = _extract_inputs(batch, self.device)

            outputs = self._forward_model(inp)
            loss = outputs['loss']

            self.optimizer.zero_grad()
            loss.backward()

            # قص الجريدينت
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            # لوجينغ
            if self.global_step % self.config.get('log_interval', 100) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'avg_loss': f'{avg_loss:.4f}',
                                  'lr': f'{lr:.6f}'})
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """تحقق"""
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Validation")
        for batch in pbar:
            inp = _extract_inputs(batch, self.device)
            outputs = self._forward_model(inp)
            loss = outputs['loss']
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / max(1, len(self.val_loader))
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """حفظ نقاط التفتيش"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        ckpt_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def train(self, phase: str = 'end_to_end', resume_from: Optional[str] = None):
        """
        حلقة التدريب العامة
        
        Args:
            phase: 'bridge_pretrain' or 'end_to_end'
            resume_from: مسار checkpoint للاستكمال (اختياري)
        """
        print(f"\n{'='*60}")
        print(f"Starting {phase} training")
        print(f"{'='*60}\n")

        # Optimizer
        self.setup_optimizer(phase)
        
        # تحميل checkpoint إن وُجد
        if resume_from:
            self.load_checkpoint(resume_from)

        num_epochs = self.config.get(f'{phase}_epochs', self.config['num_epochs'])
        
        # استخدام self.start_epoch للاستكمال من الإيبوك الصحيح
        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)

            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")

            if epoch % self.config.get('save_interval', 1) == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

        print(f"\n{'='*60}")
        print(f"Finished {phase} training")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        self.writer.close()

# -------------------------------------------------
# Dummy components (للاختبار فقط)
# -------------------------------------------------
class DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Identity()
    def encode_text(self, text):
        return torch.randn(len(text), 768)

class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(768, 50000)
    def forward(self, encoder_hidden_states, labels=None):
        batch_size, seq_len, _ = encoder_hidden_states.shape
        pooled = encoder_hidden_states.mean(dim=1)
        logits = self.lm_head(pooled).unsqueeze(1).expand(
            -1, labels.size(1) if labels is not None else 20, -1
        )
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 50000),
                labels.reshape(-1),
                ignore_index=-100
            )
        class Output: pass
        out = Output()
        out.logits = logits
        out.loss   = loss
        return out

def build_model_from_config(config: Dict) -> BLAMultiCapCLIP:
    """
    يبني الموديل. إن لم يكن لديك طريقة تهيئة جاهزة،
    نستعمل DummyCLIP + DummyDecoder لتجارب البايبلاين.
    """
    # إن كان لديك checkpoint/دالة from_config في bla_model فاستخدميها هنا.
    # وإلا نعود للدمي:
    clip_model = DummyCLIP()
    decoder = DummyDecoder()
    dim = config.get("model", {}).get("dim", 768)
    num_prompts = config.get("model", {}).get("num_prompts", 16)
    concept_embeddings = torch.randn(1000, dim)

    model = BLAMultiCapCLIP(
        clip_model=clip_model,
        decoder=decoder,
        concept_embeddings=concept_embeddings,
        dim=dim,
        num_prompts=num_prompts
    )
    return model

# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train BLA-MultiCapCLIP')
    parser.add_argument('--config', type=str, default='configs/train_config.json',
                        help='Path to config file')
    parser.add_argument('--phase', type=str, default='end_to_end',
                        choices=['bridge_pretrain', 'end_to_end'],
                        help='Training phase')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    # إضافة معامل الاستكمال
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/bla_multicapclip/checkpoint_epoch_7.pt)')
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "experiment_name": "bla_multicapclip_baseline",
            'output_dir': 'outputs/bla_multicapclip',
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 20,
            'bridge_pretrain_epochs': 3,
            'batch_size': 32,
            'max_grad_norm': 1.0,
            'log_interval': 100,
            'save_interval': 1,
            'use_scheduler': True,
            "data": {
                # عدّلي هذه المسارات لمساراتك الفعلية
                "train_images": "/content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/train2014",
                "val_images":   "/content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/val2014",
                "train_ann":    "/content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/annotations/captions_train2014.json",
                "val_ann":      "/content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/annotations/captions_val2014.json",
                "max_caption_length": 20,
                "num_workers": 0,
                "image_size": 224
            },
            "model": {
                "dim": 768,
                "num_prompts": 16
            }
        }
        print("Using default config")

    print("Configuration:")
    print(json.dumps(config, indent=2))

    # Data - دائماً نبني COCO dataloader
    print("\nCreating model and dataloaders...")
    train_loader, val_loader = build_coco_dataloaders(config)

    # Model
    model = build_model_from_config(config)

    # Trainer
    trainer = BLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # (اختياري) طباعة مفاتيح أول باتش
    first = next(iter(train_loader))
    print("Batch keys (sample):", list(first.keys()))

    # Train (مع دعم الاستكمال)
    trainer.train(phase=args.phase, resume_from=args.resume)

if __name__ == '__main__':
    main()