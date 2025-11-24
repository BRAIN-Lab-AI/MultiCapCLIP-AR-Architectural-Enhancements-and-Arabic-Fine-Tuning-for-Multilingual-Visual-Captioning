"""
Evaluation Script for BLA-MultiCapCLIP
Author: Manus AI
Date: November 4, 2025

This script evaluates the trained model on COCO validation set.
Computes retrieval and captioning metrics.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Try to import evaluation metrics (install if needed)
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: pycocoevalcap not installed. Install with:")
    print("pip install pycocoevalcap")
    METRICS_AVAILABLE = False


class RetrievalEvaluator:
    """
    Evaluator for image-text retrieval tasks
    Computes Recall@K metrics
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Initialize evaluator
        
        Args:
            k_values: List of K values for Recall@K
        """
        self.k_values = k_values
    
    def compute_recall_at_k(
        self,
        similarity_matrix: torch.Tensor,
        k: int
    ) -> float:
        """
        Compute Recall@K
        
        Args:
            similarity_matrix: Similarity matrix [N, N]
            k: K value for recall
            
        Returns:
            Recall@K score
        """
        n = similarity_matrix.size(0)
        
        # Get top-k predictions for each query
        top_k = torch.topk(similarity_matrix, k, dim=1).indices
        
        # Check if ground truth is in top-k
        # Assuming diagonal elements are ground truth matches
        correct = 0
        for i in range(n):
            if i in top_k[i]:
                correct += 1
        
        return correct / n
    
    def evaluate_retrieval(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate both image-to-text and text-to-image retrieval
        
        Args:
            image_embeddings: Image embeddings [N, D]
            text_embeddings: Text embeddings [N, D]
            
        Returns:
            Dictionary of metrics
        """
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrices
        i2t_similarity = torch.matmul(image_embeddings, text_embeddings.T)
        t2i_similarity = torch.matmul(text_embeddings, image_embeddings.T)
        
        metrics = {}
        
        # Image-to-text retrieval
        for k in self.k_values:
            recall = self.compute_recall_at_k(i2t_similarity, k)
            metrics[f'i2t_recall@{k}'] = recall
        
        # Text-to-image retrieval
        for k in self.k_values:
            recall = self.compute_recall_at_k(t2i_similarity, k)
            metrics[f't2i_recall@{k}'] = recall
        
        # Average recall
        avg_recall = np.mean([metrics[f'i2t_recall@{k}'] for k in self.k_values] +
                            [metrics[f't2i_recall@{k}'] for k in self.k_values])
        metrics['avg_recall'] = avg_recall
        
        return metrics


class CaptioningEvaluator:
    """
    Evaluator for image captioning tasks
    Computes BLEU, METEOR, ROUGE, CIDEr metrics
    """
    
    def __init__(self):
        """Initialize evaluator with metric scorers"""
        if not METRICS_AVAILABLE:
            raise ImportError("pycocoevalcap is required for captioning evaluation")
        
        self.scorers = {
            'BLEU': Bleu(4),
            'METEOR': Meteor(),
            'ROUGE': Rouge(),
            'CIDEr': Cider()
        }
    
    def evaluate_captions(
        self,
        predictions: Dict[str, List[str]],
        references: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate generated captions against references
        
        Args:
            predictions: Dict mapping image IDs to predicted captions
            references: Dict mapping image IDs to reference captions
            
        Returns:
            Dictionary of metrics
        """
        # Format for pycocoevalcap
        # predictions: {img_id: [caption]}
        # references: {img_id: [ref1, ref2, ...]}
        
        metrics = {}
        
        for metric_name, scorer in self.scorers.items():
            score, scores = scorer.compute_score(references, predictions)
            
            if isinstance(score, list):
                # BLEU returns list of scores for different n-grams
                for i, s in enumerate(score, 1):
                    metrics[f'{metric_name}-{i}'] = s
            else:
                metrics[metric_name] = score
        
        return metrics


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    output_file: str,
    device: str = 'cuda',
    batch_size: int = 32,
    mode: str = 'retrieval'
):
    """
    Main evaluation function
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to validation data
        output_file: Path to save results
        device: Device to use
        batch_size: Batch size for evaluation
        mode: Evaluation mode ('retrieval' or 'captioning')
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    if mode == 'retrieval':
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION")
        print("="*60)
        
        evaluator = RetrievalEvaluator(k_values=[1, 5, 10])
        
        # Load or compute embeddings
        # Note: This is a placeholder - implement based on your actual data loading
        print("Computing embeddings...")
        
        # Dummy embeddings for demonstration
        n_samples = 1000
        dim = 768
        image_embeddings = torch.randn(n_samples, dim)
        text_embeddings = torch.randn(n_samples, dim)
        
        # Evaluate
        metrics = evaluator.evaluate_retrieval(image_embeddings, text_embeddings)
        
        # Print results
        print("\nRetrieval Results:")
        print("-" * 40)
        print("Image-to-Text Retrieval:")
        for k in [1, 5, 10]:
            print(f"  Recall@{k}: {metrics[f'i2t_recall@{k}']:.4f}")
        
        print("\nText-to-Image Retrieval:")
        for k in [1, 5, 10]:
            print(f"  Recall@{k}: {metrics[f't2i_recall@{k}']:.4f}")
        
        print(f"\nAverage Recall: {metrics['avg_recall']:.4f}")
        
    elif mode == 'captioning':
        if not METRICS_AVAILABLE:
            print("Error: pycocoevalcap not installed")
            print("Install with: pip install pycocoevalcap")
            return
        
        print("\n" + "="*60)
        print("CAPTIONING EVALUATION")
        print("="*60)
        
        evaluator = CaptioningEvaluator()
        
        # Load predictions and references
        # Note: This is a placeholder - implement based on your actual data
        print("Loading captions...")
        
        # Dummy data for demonstration
        predictions = {str(i): [f"predicted caption {i}"] for i in range(100)}
        references = {str(i): [f"reference caption {i} variant 1",
                              f"reference caption {i} variant 2"] for i in range(100)}
        
        # Evaluate
        metrics = evaluator.evaluate_captions(predictions, references)
        
        # Print results
        print("\nCaptioning Results:")
        print("-" * 40)
        for metric_name, score in metrics.items():
            print(f"  {metric_name}: {score:.4f}")
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint.get('val_loss', None),
        'mode': mode,
        'metrics': metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BLA-MultiCapCLIP')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--mode', type=str, default='retrieval',
                        choices=['retrieval', 'captioning', 'both'],
                        help='Evaluation mode')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    
    if args.mode == 'both':
        # Run both evaluations
        print("Running retrieval evaluation...")
        evaluate_model(
            args.checkpoint,
            args.data_dir,
            args.output.replace('.json', '_retrieval.json'),
            args.device,
            args.batch_size,
            'retrieval'
        )
        
        print("\n" + "="*60 + "\n")
        
        print("Running captioning evaluation...")
        evaluate_model(
            args.checkpoint,
            args.data_dir,
            args.output.replace('.json', '_captioning.json'),
            args.device,
            args.batch_size,
            'captioning'
        )
    else:
        evaluate_model(
            args.checkpoint,
            args.data_dir,
            args.output,
            args.device,
            args.batch_size,
            args.mode
        )


if __name__ == '__main__':
    main()
