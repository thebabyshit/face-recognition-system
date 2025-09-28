"""Advanced training script for face recognition model."""

import argparse
import sys
import json
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import ExperimentConfig, create_default_config, create_quick_test_config
from models.face_recognition import FaceRecognitionModel
from data.dataset import create_data_loaders
from training.trainer import AdvancedTrainer


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)


def save_config_to_file(config: ExperimentConfig, config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def create_model_from_config(config: ExperimentConfig) -> FaceRecognitionModel:
    """Create model from configuration."""
    return FaceRecognitionModel(
        num_classes=config.model.num_classes,
        backbone_depth=int(config.model.backbone.replace('resnet', '')),
        feat_dim=config.model.feature_dim,
        drop_ratio=config.model.dropout_rate,
        scale=config.model.arcface_scale,
        margin=config.model.arcface_margin,
        easy_margin=config.model.easy_margin
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Advanced face recognition training')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Path to save configuration')
    
    # Quick options (override config)
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory of dataset')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu, cuda, auto)')
    
    # Preset configurations
    parser.add_argument('--quick-test', action='store_true',
                        help='Use quick test configuration')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Data options
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--quality-threshold', type=float, default=None,
                        help='Image quality threshold')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_file(args.config)
    elif args.quick_test:
        print("Using quick test configuration")
        config = create_quick_test_config()
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override configuration with command line arguments
    if args.data_root:
        config.data.data_root = args.data_root
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.device = args.device
    if args.num_workers:
        config.training.num_workers = args.num_workers
    if args.quality_threshold:
        config.data.quality_threshold = args.quality_threshold
    
    # Save configuration if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
        print(f"Configuration saved to: {args.save_config}")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Data root: {config.data.data_root}")
    print(f"Model: {config.model.backbone}")
    print(f"Classes: {config.model.num_classes}")
    print(f"Feature dim: {config.model.feature_dim}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Scheduler: {config.training.scheduler}")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.training.use_amp}")
    print(f"Early stopping: {config.training.early_stopping}")
    print("="*60)
    
    # Validate data root
    data_root = Path(config.data.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return 1
    
    # Create data loaders
    print("\nLoading dataset...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=config.data.data_root,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            target_size=config.model.input_size
        )
        
        print(f"Dataset loaded successfully:")
        print(f"  Train: {len(train_loader.dataset):,} samples")
        print(f"  Val: {len(val_loader.dataset):,} samples")
        print(f"  Test: {len(test_loader.dataset):,} samples")
        
        # Update number of classes in config
        actual_num_classes = len(train_loader.dataset.labels)
        if actual_num_classes != config.model.num_classes:
            print(f"Updating num_classes: {config.model.num_classes} -> {actual_num_classes}")
            config.model.num_classes = actual_num_classes
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Create model
    print("\nCreating model...")
    try:
        model = create_model_from_config(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully:")
        print(f"  Architecture: {config.model.backbone}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return 1
    
    # Create trainer
    print("\nInitializing trainer...")
    try:
        trainer = AdvancedTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        print("Trainer initialized successfully")
        
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return 1
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 1
    
    # Start training
    print("\nStarting training...")
    try:
        results = trainer.train()
        
        # Save final results
        results_path = Path(config.logging.checkpoint_dir) / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                serializable_results[key] = value
            else:
                # Skip non-serializable values
                continue
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nTraining results saved to: {results_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
        print(f"Training time: {results['training_time']/3600:.2f} hours")
        print(f"Final epoch: {results['final_epoch']}")
        
        if 'test_metrics' in results:
            print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())