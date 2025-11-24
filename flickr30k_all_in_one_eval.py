
import os
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

import ruamel.yaml as yaml
from torchvision import transforms

import models
import configs

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def simple_tokenize(text):
    return text.lower().strip().split()


def lcs_length(a, b):

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def compute_rouge_l(references, hypotheses):
    assert len(references) == len(hypotheses)
    scores = []

    for refs, hyp in zip(references, hypotheses):
        if len(hyp) == 0:
            scores.append(0.0)
            continue

        best_f1 = 0.0
        for ref in refs:
            if len(ref) == 0:
                continue
            lcs = lcs_length(ref, hyp)
            prec = lcs / len(hyp)
            rec = lcs / len(ref)
            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = (2 * prec * rec) / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)

    return sum(scores) / max(len(scores), 1)


def compute_meteor(references, hypotheses):

    assert len(references) == len(hypotheses)
    scores = []
    for refs, hyp in zip(references, hypotheses):
        if len(hyp) == 0:
            scores.append(0.0)
            continue
        try:
            score = meteor_score(refs, hyp)
        except Exception:
            score = 0.0
        scores.append(score)

    return sum(scores) / max(len(scores), 1)


def load_flickr_json(path):
    print(f">>> Loading Flickr30k JSON: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    samples = []


    if isinstance(data, dict) and "test_samples" in data:
        print("[INFO] Detected format: dict['test_samples']")
        for item in data["test_samples"]:
            samples.append({
                "image_id": item.get("image_id", item.get("id")),
                "filename": item["filename"],
                "captions": item.get("captions", []),
            })


    elif isinstance(data, list):
        print("[INFO] Detected format: list of objects")
        for item in data:
            samples.append({
                "image_id": item.get("image_id", item.get("id")),
                "filename": item.get("filename"),
                "captions": item.get("captions", []),
            })


    elif isinstance(data, dict):
        print("[INFO] Detected format: dict[id] -> [captions]")
        for image_id, caps in data.items():
            filename = image_id
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filename = f"{image_id}.jpg"
            samples.append({
                "image_id": image_id,
                "filename": filename,
                "captions": caps if isinstance(caps, list) else [caps],
            })
    else:
        raise ValueError("Unknown Flickr30k JSON format!")

    print(f"[INFO] Loaded {len(samples)} Flickr30k samples")
    return samples


def load_model_and_preprocess(checkpoint_path, config_path=None, device="cuda", mode="adapt"):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")

    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    print(f"[INFO] Building MultiCapCLIP model (mode='{mode}')...")
    model = models.build_model(cfg, mode=mode)
    model = model.to(device)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]} ...")
    print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.eval()

    # CLIP-style image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    return model, cfg, transform


@torch.no_grad()
def generate_caption_for_image(model, image_tensor, max_length=40, num_beams=3):
    image_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)

    result = model.generate(
        image=image_tensor,
        max_length=max_length,
        num_beams=num_beams,
        repetition_penalty=1.1,
    )

    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to MultiCapCLIP checkpoint (.pth)")
    parser.add_argument("--flickr_json", required=True, help="Path to Flickr30k JSON (GT captions)")
    parser.add_argument("--flickr_imgs", required=True, help="Path to Flickr30k images directory")
    parser.add_argument("--output_json", required=True, help="Where to save predictions JSON")
    parser.add_argument("--config", default=None, help="Optional config.yaml path (if not next to checkpoint)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--mode", default="adapt", help="Model mode (default: adapt)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f">>> Device: {device}")


    flickr_samples = load_flickr_json(args.flickr_json)


    print(">>> Loading model...")
    model, cfg, transform = load_model_and_preprocess(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        mode=args.mode,
    )

    predictions = []
    hyps_tokens = []
    refs_tokens = []

    print(">>> Starting evaluation over Flickr30k...")
    for item in tqdm(flickr_samples, desc="Evaluating Flickr30k"):
        img_file = item["filename"]
        img_path = os.path.join(args.flickr_imgs, img_file)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image file: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image)

        hyp = generate_caption_for_image(model, image_tensor)
        if isinstance(hyp, torch.Tensor):
            hyp = hyp[0]
        if not isinstance(hyp, str):
            hyp = str(hyp)

        refs = item.get("captions", [])
        if refs is None:
            refs = []

        predictions.append({
            "image_id": item.get("image_id"),
            "filename": img_file,
            "generated": hyp,
            "gt": refs,
        })

        # tokenization for metrics
        hyp_tok = simple_tokenize(hyp)
        ref_toks = [simple_tokenize(r) for r in refs if isinstance(r, str) and len(r.strip()) > 0]

        if len(ref_toks) == 0:
            ref_toks = [[]]

        hyps_tokens.append(hyp_tok)
        refs_tokens.append(ref_toks)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(predictions)} predictions to: {args.output_json}")

    if len(predictions) == 0:
        print("[WARN] No predictions to evaluate. Check image paths / JSON format.")
        return

    print("\n===== Evaluation Metrics on Flickr30k =====")
    smoothie = SmoothingFunction().method1


    bleu1 = corpus_bleu(refs_tokens, hyps_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs_tokens, hyps_tokens, weights=(1/3, 1/3, 1/3, 0.0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    meteor = compute_meteor(refs_tokens, hyps_tokens)
    rouge_l = compute_rouge_l(refs_tokens, hyps_tokens)

    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")


# ============================================================
if __name__ == "__main__":
    main()
