"""Advanced trainer for face recognition model."""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import configurations and models
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import ExperimentConfig
from models.face_recognition import FaceRecognitionModel
from utils.metrics import calculate_metrics, plot_confusion_matrix, print_metrics_summary


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class AdvancedTrainer:
    """Advanced trainer with comprehensive features."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: FaceRecognitionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration
            model: Face recognition model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Setup logging
        self.writer = self._setup_tensorboard()
        self.checkpoint_dir = Path(config.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                mode='max'
            )
        
        # Set random seeds
        self._set_random_seeds(config.seed)
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        params = self.model.parameters()
        
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.training.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.training.patience,
                factor=self.config.training.gamma
            )
        elif self.config.training.scheduler.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.gamma
            )
        else:
            return None
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion."""
        if self.config.training.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=self.config.training.label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def _setup_tensorboard(self) -> SummaryWriter:
        """Setup TensorBoard logging."""
        log_dir = Path(self.config.logging.tensorboard_dir) / self.config.experiment_name
        return SummaryWriter(log_dir=str(log_dir))
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.training.use_amp and self.scaler is not None:
                with autocast():
                    features, logits = self.model(images, labels)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                features, logits = self.model(images, labels)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            
            # Logging
            if batch_idx % self.config.logging.log_frequency == 0:
                batch_time = time.time() - batch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'LR: {current_lr:.6f}, '
                      f'Time: {batch_time:.2f}s')
                
                # TensorBoard logging
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        epoch_time = time.time() - epoch_start_time
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time
        }
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.config.training.use_amp:
                    with autocast():
                        features, logits = self.model(images, labels)
                        loss = self.criterion(logits, labels)
                else:
                    features, logits = self.model(images, labels)
                    loss = self.criterion(logits, labels)
                
                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions and features
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.append(features.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Additional metrics
        try:
            all_features_np = np.vstack(all_features)
            detailed_metrics = calculate_metrics(
                np.array(all_predictions),
                np.array(all_labels),
                all_features_np
            )
        except Exception as e:
            print(f"Warning: Could not calculate detailed metrics: {e}")
            detailed_metrics = {}
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'features': all_features,
            **detailed_metrics
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if load_optimizer and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print("="*60)
        print(f"Starting training: {self.config.experiment_name}")
        print("="*60)
        print(f"Model: {self.config.model.backbone}")
        print(f"Dataset: {self.config.data.data_root}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Device: {self.device}")
        print("="*60)
        
        # Save configuration
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        training_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            if epoch % self.config.training.validation_frequency == 0:
                val_metrics = self.validate_epoch()
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                
                # TensorBoard logging
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['accuracy'])
                    else:
                        self.scheduler.step()
                
                # Check for best model
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                
                # Print progress
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{self.config.training.num_epochs-1}")
                print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                print(f"Time: {epoch_time:.1f}s, Best Val Acc: {self.best_val_acc:.4f}")
                
                # Save checkpoint
                if epoch % self.config.training.save_frequency == 0 or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping
                if self.early_stopping and self.early_stopping(val_metrics['accuracy']):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        training_time = time.time() - training_start_time
        
        print(f"\nTraining completed!")
        print(f"Total time: {training_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Final evaluation
        final_results = {}
        if self.test_loader:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_loader)
            final_results['test_metrics'] = test_metrics
            print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        final_results.update({
            'training_time': training_time,
            'best_val_acc': self.best_val_acc,
            'final_epoch': self.current_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        })
        
        return final_results
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on given dataset."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_features = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                features, logits = self.model(images, labels)
                loss = self.criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.append(features.cpu().numpy())
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / total_samples
        all_features_np = np.vstack(all_features)
        
        metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            all_features_np
        )
        
        metrics['loss'] = avg_loss
        metrics['predictions'] = all_predictions
        metrics['labels'] = all_labels
        
        return metrics