"""Model evaluation script for face recognition."""

import argparse
import sys
from pathlib import Path
import json
import time
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.face_recognition import FaceRecognitionModel, FaceRecognizer
from data.dataset import create_data_loaders, FacecapDataset
from utils.metrics import (
    calculate_metrics, calculate_face_verification_metrics,
    calculate_rank_accuracy, plot_confusion_matrix, plot_roc_curve,
    print_metrics_summary
)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        device: str = 'auto',
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_root: Root directory of dataset
            device: Device to use for evaluation
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
        """
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
    
    def _load_model(self) -> FaceRecognitionModel:
        """Load trained model."""
        print(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_config = config.get('model', {})
            
            num_classes = model_config.get('num_classes', 500)
            backbone = model_config.get('backbone', 'resnet50')
            feature_dim = model_config.get('feature_dim', 512)
            dropout_rate = model_config.get('dropout_rate', 0.6)
            arcface_scale = model_config.get('arcface_scale', 64.0)
            arcface_margin = model_config.get('arcface_margin', 0.5)
            
            backbone_depth = int(backbone.replace('resnet', ''))
        else:
            # Use default values if config not available
            num_classes = 500
            backbone_depth = 50
            feature_dim = 512
            dropout_rate = 0.6
            arcface_scale = 64.0
            arcface_margin = 0.5
        
        # Create model
        model = FaceRecognitionModel(
            num_classes=num_classes,
            backbone_depth=backbone_depth,
            feat_dim=feature_dim,
            drop_ratio=dropout_rate,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"  Classes: {num_classes}")
        print(f"  Backbone: resnet{backbone_depth}")
        print(f"  Feature dim: {feature_dim}")
        
        return model
    
    def _load_data(self):
        """Load evaluation datasets."""
        print(f"Loading dataset from: {self.data_root}")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=self.data_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            target_size=(112, 112)
        )
        
        print(f"Dataset loaded:")
        print(f"  Train: {len(train_loader.dataset):,} samples")
        print(f"  Val: {len(val_loader.dataset):,} samples")
        print(f"  Test: {len(test_loader.dataset):,} samples")
        
        return train_loader, val_loader, test_loader
    
    def evaluate_classification(self, data_loader, split_name: str = "test") -> dict:
        """Evaluate classification performance."""
        print(f"\nEvaluating {split_name} set classification...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_features = []
        all_confidences = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                features, logits = self.model(images, labels)
                
                # Get predictions and confidences
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.append(features.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        evaluation_time = time.time() - start_time
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_features = np.vstack(all_features)
        all_confidences = np.array(all_confidences)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_predictions, all_labels, all_features)
        
        # Add timing and confidence metrics
        metrics['evaluation_time'] = evaluation_time
        metrics['avg_confidence'] = np.mean(all_confidences)
        metrics['min_confidence'] = np.min(all_confidences)
        metrics['max_confidence'] = np.max(all_confidences)
        
        # Add raw data for further analysis
        metrics['predictions'] = all_predictions
        metrics['labels'] = all_labels
        metrics['features'] = all_features
        metrics['confidences'] = all_confidences
        
        print(f"Classification evaluation completed in {evaluation_time:.2f}s")
        
        return metrics
    
    def evaluate_verification(self, num_pairs: int = 10000) -> dict:
        """Evaluate face verification performance."""
        print(f"\nEvaluating face verification with {num_pairs} pairs...")
        
        # Use test set for verification evaluation
        test_dataset = self.test_loader.dataset
        
        # Generate pairs
        positive_pairs = []
        negative_pairs = []
        
        # Get samples per class
        class_samples = {}
        for idx, (_, label) in enumerate(test_dataset):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(idx)
        
        # Generate positive pairs (same person)
        num_positive = num_pairs // 2
        for _ in range(num_positive):
            # Select random class with at least 2 samples
            valid_classes = [cls for cls, samples in class_samples.items() if len(samples) >= 2]
            if not valid_classes:
                break
            
            cls = np.random.choice(valid_classes)
            idx1, idx2 = np.random.choice(class_samples[cls], 2, replace=False)
            positive_pairs.append((idx1, idx2, 1))
        
        # Generate negative pairs (different persons)
        num_negative = num_pairs - len(positive_pairs)
        for _ in range(num_negative):
            cls1, cls2 = np.random.choice(list(class_samples.keys()), 2, replace=False)
            idx1 = np.random.choice(class_samples[cls1])
            idx2 = np.random.choice(class_samples[cls2])
            negative_pairs.append((idx1, idx2, 0))
        
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
        
        # Extract features for all pairs
        similarities = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for idx1, idx2, label in all_pairs:
                # Get images
                img1, _ = test_dataset[idx1]
                img2, _ = test_dataset[idx2]
                
                # Add batch dimension and move to device
                img1 = img1.unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).to(self.device)
                
                # Extract features
                feat1 = self.model.extract_features(img1).squeeze()
                feat2 = self.model.extract_features(img2).squeeze()
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(feat1, feat2, dim=0).item()
                
                similarities.append(similarity)
                labels.append(label)
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Calculate verification metrics
        verification_metrics = calculate_face_verification_metrics(similarities, labels)
        
        print(f"Verification evaluation completed")
        print(f"  EER: {verification_metrics['eer']:.4f}")
        print(f"  AUC: {verification_metrics['auc']:.4f}")
        
        return verification_metrics
    
    def evaluate_identification(self, gallery_size: int = 1000) -> dict:
        """Evaluate face identification performance."""
        print(f"\nEvaluating face identification with gallery size {gallery_size}...")
        
        test_dataset = self.test_loader.dataset
        
        # Select gallery and query sets
        all_indices = list(range(len(test_dataset)))
        np.random.shuffle(all_indices)
        
        gallery_indices = all_indices[:gallery_size]
        query_indices = all_indices[gallery_size:gallery_size + min(500, len(all_indices) - gallery_size)]
        
        print(f"Gallery size: {len(gallery_indices)}")
        print(f"Query size: {len(query_indices)}")
        
        # Extract features for gallery
        gallery_features = []
        gallery_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in gallery_indices:
                img, label = test_dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                
                features = self.model.extract_features(img).squeeze()
                gallery_features.append(features.cpu().numpy())
                gallery_labels.append(label)
        
        gallery_features = np.array(gallery_features)
        gallery_labels = np.array(gallery_labels)
        
        # Extract features for queries
        query_features = []
        query_labels = []
        
        with torch.no_grad():
            for idx in query_indices:
                img, label = test_dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                
                features = self.model.extract_features(img).squeeze()
                query_features.append(features.cpu().numpy())
                query_labels.append(label)
        
        query_features = np.array(query_features)
        query_labels = np.array(query_labels)
        
        # Calculate rank accuracies
        rank_accuracies = calculate_rank_accuracy(
            query_features, gallery_features,
            query_labels, gallery_labels,
            ranks=[1, 5, 10, 20]
        )
        
        print(f"Identification evaluation completed")
        for rank, acc in rank_accuracies.items():
            print(f"  Rank-{rank} accuracy: {acc:.4f}")
        
        return {
            'rank_accuracies': rank_accuracies,
            'gallery_size': len(gallery_indices),
            'query_size': len(query_indices)
        }
    
    def generate_visualizations(self, metrics: dict, output_dir: str):
        """Generate evaluation visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # Confusion matrix
        if 'predictions' in metrics and 'labels' in metrics:
            try:
                fig = plot_confusion_matrix(
                    metrics['labels'],
                    metrics['predictions'],
                    normalize=True,
                    title='Normalized Confusion Matrix'
                )
                plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("  ✓ Confusion matrix saved")
            except Exception as e:
                print(f"  ✗ Error generating confusion matrix: {e}")
        
        # Confidence distribution
        if 'confidences' in metrics:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(metrics['confidences'], bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Prediction Confidence')
                plt.ylabel('Frequency')
                plt.title('Prediction Confidence Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✓ Confidence distribution saved")
            except Exception as e:
                print(f"  ✗ Error generating confidence distribution: {e}")
    
    def run_full_evaluation(self, output_dir: str = "evaluation_results") -> dict:
        """Run comprehensive evaluation."""
        print("="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        # Classification evaluation
        test_metrics = self.evaluate_classification(self.test_loader, "test")
        results['classification'] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'evaluation_time': test_metrics['evaluation_time'],
            'avg_confidence': test_metrics['avg_confidence']
        }
        
        # Print classification summary
        print_metrics_summary(test_metrics)
        
        # Verification evaluation
        try:
            verification_metrics = self.evaluate_verification(num_pairs=5000)
            results['verification'] = verification_metrics
        except Exception as e:
            print(f"Verification evaluation failed: {e}")
            results['verification'] = None
        
        # Identification evaluation
        try:
            identification_metrics = self.evaluate_identification(gallery_size=500)
            results['identification'] = identification_metrics
        except Exception as e:
            print(f"Identification evaluation failed: {e}")
            results['identification'] = None
        
        # Generate visualizations
        try:
            self.generate_visualizations(test_metrics, output_dir)
        except Exception as e:
            print(f"Visualization generation failed: {e}")
        
        # Save results
        output_path = Path(output_dir) / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        if isinstance(v, np.ndarray):
                            serializable_results[key][k] = v.tolist()
                        else:
                            serializable_results[key][k] = v
                    elif isinstance(v, (int, float, str, bool)) or v is None:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEvaluation results saved to: {output_path}")
        
        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate face recognition model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-root', type=str, default='../facecap',
                        help='Root directory of dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--verification-pairs', type=int, default=10000,
                        help='Number of pairs for verification evaluation')
    parser.add_argument('--gallery-size', type=int, default=1000,
                        help='Gallery size for identification evaluation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    if not Path(args.data_root).exists():
        print(f"Error: Data root not found: {args.data_root}")
        return 1
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            data_root=args.data_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation(args.output_dir)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'classification' in results:
            cls_results = results['classification']
            print(f"Classification Accuracy: {cls_results['accuracy']:.4f}")
            print(f"Average Confidence: {cls_results['avg_confidence']:.4f}")
        
        if 'verification' in results and results['verification']:
            ver_results = results['verification']
            print(f"Verification EER: {ver_results['eer']:.4f}")
            print(f"Verification AUC: {ver_results['auc']:.4f}")
        
        if 'identification' in results and results['identification']:
            id_results = results['identification']
            rank_accs = id_results['rank_accuracies']
            print(f"Rank-1 Accuracy: {rank_accs[1]:.4f}")
            print(f"Rank-5 Accuracy: {rank_accs[5]:.4f}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())