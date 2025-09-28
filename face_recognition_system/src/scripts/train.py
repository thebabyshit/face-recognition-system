"""Training script for face recognition model."""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_data_loaders, FacecapDataset
from models.face_recognition import FaceRecognitionModel
from utils.metrics import calculate_metrics, plot_confusion_matrix


class ModelTrainer:
    """Face recognition model trainer."""
    
    def __init__(
        self,
        model: FaceRecognitionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        log_dir: str = 'logs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Face recognition model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            log_dir: Directory for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def setup_optimizer(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        optimizer_type: str = 'adam'
    ):
        """Setup optimizer and learning rate scheduler."""
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            features, logits = self.model(images, labels)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 100 == 0:
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features, logits = self.model(images, labels)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False,
        save_optimizer: bool = True
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = 'models/checkpoints',
        save_every: int = 5,
        early_stopping_patience: int = 10
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch}/{num_epochs-1} - '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, '
                  f'Time: {elapsed_time:.1f}s')
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % save_every == 0 or is_best:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {patience_counter} epochs without improvement')
                break
        
        print(f'Training completed! Best validation accuracy: {self.best_val_acc:.4f}')
        self.writer.close()
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features, logits = self.model(images, labels)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.append(features.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'features': np.vstack(all_features)
        }
        
        print(f'Test Accuracy: {accuracy:.4f}')
        
        return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--data-root', type=str, default='../facecap',
                        help='Root directory of facecap dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--save-dir', type=str, default='models/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading dataset...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get number of classes from dataset
        train_dataset = train_loader.dataset
        num_classes = len(train_dataset.labels)
        print(f"Dataset loaded: {num_classes} classes")
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Val: {len(val_loader.dataset)} samples")
        print(f"Test: {len(test_loader.dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Create model
    print("Creating model...")
    model = FaceRecognitionModel(num_classes=num_classes)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=args.log_dir
    )
    
    # Setup optimizer
    trainer.setup_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    
    # Save final results
    results_path = Path(args.save_dir) / 'test_results.json'
    import json
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'accuracy': test_results['accuracy'],
            'classification_report': test_results['classification_report']
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    return 0


if __name__ == '__main__':
    exit(main())