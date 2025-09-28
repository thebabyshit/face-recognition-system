"""Training configuration for face recognition model."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    backbone: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101, resnet152
    feature_dim: int = 512
    num_classes: int = 500
    dropout_rate: float = 0.6
    
    # ArcFace parameters
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    easy_margin: bool = False
    
    # Input
    input_size: Tuple[int, int] = (112, 112)  # (height, width)
    input_channels: int = 3


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    
    # Optimizer
    optimizer: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9  # for SGD
    
    # Learning rate scheduler
    scheduler: str = "step"  # step, cosine, plateau, exponential
    step_size: int = 10  # for StepLR
    gamma: float = 0.1  # for StepLR and ExponentialLR
    patience: int = 5  # for ReduceLROnPlateau
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Validation
    validation_frequency: int = 1  # validate every N epochs
    save_frequency: int = 5  # save checkpoint every N epochs
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0  # 0.0 means no mixup
    cutmix_alpha: float = 0.0  # 0.0 means no cutmix


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Paths
    data_root: str = "../facecap"
    processed_data_root: Optional[str] = None
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_prob: float = 0.3
    rotation_degrees: float = 10.0
    
    # Normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Quality filtering
    quality_threshold: Optional[float] = None
    enhance_images: bool = False


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    
    # Directories
    log_dir: str = "logs"
    checkpoint_dir: str = "models/checkpoints"
    tensorboard_dir: str = "runs"
    
    # Logging frequency
    log_frequency: int = 100  # log every N batches
    image_log_frequency: int = 1000  # log images every N batches
    
    # Metrics to track
    track_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "loss", "learning_rate", "epoch_time"
    ])
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    save_model_graph: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment info
    experiment_name: str = "facecap_arcface_resnet50"
    description: str = "Face recognition training on facecap dataset"
    tags: List[str] = field(default_factory=lambda: ["face_recognition", "arcface", "resnet50"])
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Update model num_classes based on data
        if self.data.data_root:
            try:
                # Try to determine number of classes from labels.txt
                labels_file = Path(self.data.data_root) / "labels.txt"
                if labels_file.exists():
                    with open(labels_file, 'r') as f:
                        num_classes = len([line.strip() for line in f if line.strip()])
                    self.model.num_classes = num_classes
            except Exception:
                pass  # Keep default value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "seed": self.seed,
            "device": self.device,
            "model": {
                "backbone": self.model.backbone,
                "feature_dim": self.model.feature_dim,
                "num_classes": self.model.num_classes,
                "dropout_rate": self.model.dropout_rate,
                "arcface_scale": self.model.arcface_scale,
                "arcface_margin": self.model.arcface_margin,
                "easy_margin": self.model.easy_margin,
                "input_size": self.model.input_size,
                "input_channels": self.model.input_channels,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "optimizer": self.training.optimizer,
                "scheduler": self.training.scheduler,
                "early_stopping": self.training.early_stopping,
                "use_amp": self.training.use_amp,
                "label_smoothing": self.training.label_smoothing,
            },
            "data": {
                "data_root": self.data.data_root,
                "use_augmentation": self.data.use_augmentation,
                "quality_threshold": self.data.quality_threshold,
                "enhance_images": self.data.enhance_images,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "checkpoint_dir": self.logging.checkpoint_dir,
                "tensorboard_dir": self.logging.tensorboard_dir,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update basic fields
        for key in ["experiment_name", "description", "tags", "seed", "device"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Update model config
        if "model" in config_dict:
            model_dict = config_dict["model"]
            for key, value in model_dict.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update training config
        if "training" in config_dict:
            training_dict = config_dict["training"]
            for key, value in training_dict.items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # Update data config
        if "data" in config_dict:
            data_dict = config_dict["data"]
            for key, value in data_dict.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Update logging config
        if "logging" in config_dict:
            logging_dict = config_dict["logging"]
            for key, value in logging_dict.items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        return config


def create_default_config() -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig()


def create_quick_test_config() -> ExperimentConfig:
    """Create configuration for quick testing."""
    config = ExperimentConfig()
    config.experiment_name = "quick_test"
    config.training.num_epochs = 5
    config.training.batch_size = 16
    config.training.early_stopping_patience = 3
    config.logging.log_frequency = 10
    return config


def create_production_config() -> ExperimentConfig:
    """Create configuration for production training."""
    config = ExperimentConfig()
    config.experiment_name = "production_training"
    config.training.num_epochs = 200
    config.training.batch_size = 64
    config.training.learning_rate = 0.0005
    config.training.early_stopping_patience = 20
    config.data.use_augmentation = True
    config.data.quality_threshold = 0.4
    return config