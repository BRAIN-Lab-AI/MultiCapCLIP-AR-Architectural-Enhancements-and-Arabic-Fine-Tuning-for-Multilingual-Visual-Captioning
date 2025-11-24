"""
Visualization Utilities for BLA-MultiCapCLIP
Author: Manus AI
Date: October 30, 2025

Functions for visualizing attention weights, generated captions, and model analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import torch


def visualize_attention(
    image: np.ndarray,
    attention_weights: np.ndarray,
    prompts: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize attention weights from prompts to image regions
    
    Args:
        image: Image array [H, W, 3]
        attention_weights: Attention weights [num_prompts, num_patches]
        prompts: List of prompt texts
        save_path: Path to save figure
        figsize: Figure size
    """
    num_prompts = min(len(prompts), 4)  # Show top 4 prompts
    
    fig, axes = plt.subplots(1, num_prompts + 1, figsize=figsize)
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show attention for each prompt
    for i in range(num_prompts):
        # Reshape attention to spatial grid
        attn = attention_weights[i]
        grid_size = int(np.sqrt(len(attn)))
        attn_map = attn.reshape(grid_size, grid_size)
        
        # Overlay on image
        axes[i + 1].imshow(image, alpha=0.5)
        axes[i + 1].imshow(attn_map, cmap='hot', alpha=0.5)
        axes[i + 1].set_title(f"Prompt: {prompts[i][:20]}...")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    baseline_metrics: dict,
    bla_metrics: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot comparison between baseline and BLA metrics
    
    Args:
        baseline_metrics: Dictionary of baseline (MultiCapCLIP) metrics
        bla_metrics: Dictionary of BLA metrics
        save_path: Path to save figure
        figsize: Figure size
    """
    metrics = list(baseline_metrics.keys())
    baseline_values = [baseline_metrics[m] for m in metrics]
    bla_values = [bla_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, baseline_values, width, label='MultiCapCLIP', color='skyblue')
    bars2 = ax.bar(x + width/2, bla_values, width, label='BLA (Ours)', color='darkblue')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: MultiCapCLIP vs BLA', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.close()


def plot_ablation_study(
    configs: List[str],
    bleu_scores: List[float],
    cider_scores: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot ablation study results
    
    Args:
        configs: List of configuration names
        bleu_scores: List of BLEU-4 scores
        cider_scores: List of CIDEr scores
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # BLEU-4
    colors = ['gray'] + ['skyblue'] * (len(configs) - 2) + ['darkblue']
    ax1.bar(configs, bleu_scores, color=colors)
    ax1.axhline(y=bleu_scores[0], color='red', linestyle='--', label='Baseline', alpha=0.7)
    ax1.set_ylabel('BLEU-4', fontsize=12)
    ax1.set_title('Ablation Study: BLEU-4 Scores', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # CIDEr
    ax2.bar(configs, cider_scores, color=['gray'] + ['lightcoral'] * (len(configs) - 2) + ['darkred'])
    ax2.axhline(y=cider_scores[0], color='red', linestyle='--', label='Baseline', alpha=0.7)
    ax2.set_ylabel('CIDEr', fontsize=12)
    ax2.set_title('Ablation Study: CIDEr Scores', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ablation study to {save_path}")
    
    plt.close()


def visualize_captions(
    images: List[np.ndarray],
    baseline_captions: List[str],
    bla_captions: List[str],
    ground_truth: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize images with different captions
    
    Args:
        images: List of image arrays
        baseline_captions: Baseline model captions
        bla_captions: BLA model captions
        ground_truth: Ground truth captions
        save_path: Path to save figure
        figsize: Figure size
    """
    num_images = min(len(images), 4)
    fig, axes = plt.subplots(num_images, 1, figsize=figsize)
    
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        caption_text = (
            f"Ground Truth: {ground_truth[i]}\n"
            f"MultiCapCLIP: {baseline_captions[i]}\n"
            f"BLA (Ours): {bla_captions[i]}"
        )
        axes[i].set_title(caption_text, fontsize=10, loc='left', pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved caption visualization to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Test metrics comparison
    baseline = {
        'BLEU-4': 32.5,
        'CIDEr': 98.3,
        'METEOR': 27.8,
        'SPICE': 20.4
    }
    
    bla = {
        'BLEU-4': 35.8,
        'CIDEr': 109.7,
        'METEOR': 30.1,
        'SPICE': 22.8
    }
    
    plot_metrics_comparison(baseline, bla, save_path='/home/ubuntu/test_comparison.png')
    
    # Test ablation study
    configs = ['Baseline', '+B1', '+B2', '+B3', '+B1+B2', '+B1+B3', '+B2+B3', 'Full BLA']
    bleu4 = [32.5, 33.4, 33.8, 34.1, 34.9, 34.7, 35.0, 35.8]
    cider = [98.3, 102.1, 103.7, 105.2, 107.3, 106.8, 108.1, 109.7]
    
    plot_ablation_study(configs, bleu4, cider, save_path='/home/ubuntu/test_ablation.png')
    
    # Test training curves
    train_losses = [2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    val_losses = [2.6, 2.3, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.15]
    
    plot_training_curves(train_losses, val_losses, save_path='/home/ubuntu/test_curves.png')
    
    print("All visualizations created successfully!")
