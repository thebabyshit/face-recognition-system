"""Metrics and evaluation utilities for face recognition."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional
import torch


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return accuracy_score(labels, predictions)


def calculate_precision_recall_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = 'macro'
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return precision, recall, f1


def calculate_face_verification_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate face verification metrics (TAR, FAR, EER).
    
    Args:
        similarities: Similarity scores between face pairs
        labels: Binary labels (1 for same person, 0 for different)
        thresholds: Thresholds to evaluate (if None, use automatic)
        
    Returns:
        Dictionary with verification metrics
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 1000)
    
    tars = []  # True Accept Rate (TPR)
    fars = []  # False Accept Rate (FPR)
    
    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        
        # Calculate TAR and FAR
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        tar = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tars.append(tar)
        fars.append(far)
    
    tars = np.array(tars)
    fars = np.array(fars)
    
    # Find Equal Error Rate (EER)
    eer_idx = np.argmin(np.abs(tars - (1 - fars)))
    eer = (fars[eer_idx] + (1 - tars[eer_idx])) / 2
    eer_threshold = thresholds[eer_idx]
    
    # Calculate AUC
    roc_auc = auc(fars, tars)
    
    return {
        'thresholds': thresholds,
        'tar': tars,
        'far': fars,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'auc': roc_auc
    }


def calculate_rank_accuracy(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    ranks: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Calculate rank-k accuracy for face identification.
    
    Args:
        query_features: Query feature embeddings [num_queries, feat_dim]
        gallery_features: Gallery feature embeddings [num_gallery, feat_dim]
        query_labels: Query labels [num_queries]
        gallery_labels: Gallery labels [num_gallery]
        ranks: List of ranks to evaluate
        
    Returns:
        Dictionary mapping rank to accuracy
    """
    # Calculate similarity matrix
    similarities = np.dot(query_features, gallery_features.T)
    
    # Get sorted indices for each query
    sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]
    
    rank_accuracies = {}
    
    for rank in ranks:
        correct = 0
        for i, query_label in enumerate(query_labels):
            # Get top-k gallery labels
            top_k_indices = sorted_indices[i, :rank]
            top_k_labels = gallery_labels[top_k_indices]
            
            # Check if query label is in top-k
            if query_label in top_k_labels:
                correct += 1
        
        rank_accuracies[rank] = correct / len(query_labels)
    
    return rank_accuracies


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Limit display for large matrices
    if cm.shape[0] > 50:
        # Show only a subset for visualization
        subset_size = 50
        cm_subset = cm[:subset_size, :subset_size]
        sns.heatmap(cm_subset, annot=False, fmt=fmt, cmap='Blues', ax=ax)
        ax.set_title(f'{title} (showing first {subset_size} classes)')
    else:
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)
        ax.set_title(title)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    similarities: np.ndarray,
    labels: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve for face verification.
    
    Args:
        similarities: Similarity scores
        labels: Binary labels
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        features: Feature embeddings (optional)
        class_names: List of class names (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = calculate_accuracy(predictions, labels)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, labels)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1_score': f1_per_class,
        'support': support
    }
    
    # Feature-based metrics (if features provided)
    if features is not None:
        # Calculate intra-class and inter-class distances
        unique_labels = np.unique(labels)
        intra_distances = []
        inter_distances = []
        
        for label in unique_labels:
            class_features = features[labels == label]
            if len(class_features) > 1:
                # Intra-class distances
                for i in range(len(class_features)):
                    for j in range(i + 1, len(class_features)):
                        dist = np.linalg.norm(class_features[i] - class_features[j])
                        intra_distances.append(dist)
            
            # Inter-class distances
            other_features = features[labels != label]
            if len(other_features) > 0:
                for class_feat in class_features:
                    for other_feat in other_features[:100]:  # Limit for efficiency
                        dist = np.linalg.norm(class_feat - other_feat)
                        inter_distances.append(dist)
        
        if intra_distances and inter_distances:
            metrics['intra_class_distance'] = {
                'mean': np.mean(intra_distances),
                'std': np.std(intra_distances)
            }
            metrics['inter_class_distance'] = {
                'mean': np.mean(inter_distances),
                'std': np.std(inter_distances)
            }
            
            # Separability ratio
            metrics['separability_ratio'] = np.mean(inter_distances) / np.mean(intra_distances)
    
    return metrics


def print_metrics_summary(metrics: Dict):
    """Print a formatted summary of metrics."""
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    if 'separability_ratio' in metrics:
        print(f"Separability Ratio: {metrics['separability_ratio']:.4f}")
    
    if 'intra_class_distance' in metrics:
        print(f"Intra-class Distance: {metrics['intra_class_distance']['mean']:.4f} ± "
              f"{metrics['intra_class_distance']['std']:.4f}")
    
    if 'inter_class_distance' in metrics:
        print(f"Inter-class Distance: {metrics['inter_class_distance']['mean']:.4f} ± "
              f"{metrics['inter_class_distance']['std']:.4f}")
    
    print("="*50)