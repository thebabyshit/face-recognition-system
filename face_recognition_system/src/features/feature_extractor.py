"""Advanced feature extraction for face recognition."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.face_recognition import FaceRecognitionModel, FaceRecognizer
from models.face_detector import FaceDetector
from utils.image_quality import ImageQualityAssessor


class BatchFeatureExtractor:
    """High-performance batch feature extraction."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        batch_size: int = 32,
        num_workers: int = 4,
        use_face_detection: bool = True,
        quality_threshold: float = 0.3
    ):
        """
        Initialize batch feature extractor.
        
        Args:
            model_path: Path to trained model
            device: Device to use for inference
            batch_size: Batch size for feature extraction
            num_workers: Number of data loading workers
            use_face_detection: Whether to use face detection
            quality_threshold: Minimum image quality threshold
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_face_detection = use_face_detection
        self.quality_threshold = quality_threshold
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Feature extractor using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize face detector if needed
        self.face_detector = None
        if use_face_detection:
            try:
                self.face_detector = FaceDetector(device=str(self.device))
                print("Face detector initialized")
            except Exception as e:
                print(f"Warning: Could not initialize face detector: {e}")
                self.use_face_detection = False
        
        # Initialize quality assessor
        self.quality_assessor = ImageQualityAssessor()
    
    def _load_model(self) -> FaceRecognitionModel:
        """Load trained model."""
        print(f"Loading model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_config = config.get('model', {})
            
            num_classes = model_config.get('num_classes', 500)
            backbone = model_config.get('backbone', 'resnet50')
            feature_dim = model_config.get('feature_dim', 512)
            dropout_rate = model_config.get('dropout_rate', 0.6)
            
            backbone_depth = int(backbone.replace('resnet', ''))
        else:
            # Default values
            num_classes = 500
            backbone_depth = 50
            feature_dim = 512
            dropout_rate = 0.6
        
        # Create and load model
        model = FaceRecognitionModel(
            num_classes=num_classes,
            backbone_depth=backbone_depth,
            feat_dim=feature_dim,
            drop_ratio=dropout_rate
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {backbone_depth}-layer ResNet, {feature_dim}D features")
        return model
    
    def preprocess_image(
        self,
        image: Union[np.ndarray, str, Path],
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[torch.Tensor]:
        """
        Preprocess single image for feature extraction.
        
        Args:
            image: Input image (array, file path, or PIL Image)
            target_size: Target image size (width, height)
            
        Returns:
            Preprocessed image tensor or None if processing failed
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # Face detection and cropping
            if self.use_face_detection and self.face_detector:
                try:
                    face_boxes = self.face_detector.detect_faces(image)
                    if face_boxes:
                        # Use the largest face
                        largest_face = max(face_boxes, key=lambda x: x.area)
                        image = self.face_detector.align_face(
                            image, largest_face, output_size=target_size
                        )
                        if image is None:
                            return None
                    else:
                        # No face detected, resize original image
                        image = cv2.resize(image, target_size)
                except Exception:
                    # Fall back to resize if face detection fails
                    image = cv2.resize(image, target_size)
            else:
                # Direct resize
                image = cv2.resize(image, target_size)
            
            # Quality assessment
            if self.quality_threshold > 0:
                quality_scores = self.quality_assessor.assess_overall_quality(image)
                if quality_scores['overall_score'] < self.quality_threshold:
                    return None
            
            # Convert to tensor and normalize
            image = image.astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_features_single(self, image: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
        """
        Extract features from single image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector or None if extraction failed
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        if image_tensor is None:
            return None
        
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.extract_features(image_batch)
            features = features.cpu().numpy().squeeze()
        
        return features
    
    def extract_features_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract features from batch of images.
        
        Args:
            images: List of input images
            show_progress: Whether to show progress
            
        Returns:
            Tuple of (features_list, valid_indices)
        """
        features_list = []
        valid_indices = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_indices = list(range(i, min(i + self.batch_size, len(images))))
            
            # Preprocess batch
            batch_tensors = []
            batch_valid_indices = []
            
            for j, img in enumerate(batch_images):
                tensor = self.preprocess_image(img)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_valid_indices.append(batch_indices[j])
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.extract_features(batch_tensor)
                batch_features = batch_features.cpu().numpy()
            
            # Collect results
            for k, features in enumerate(batch_features):
                features_list.append(features)
                valid_indices.append(batch_valid_indices[k])
            
            if show_progress and (i // self.batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_images)}/{len(images)} images")
        
        return features_list, valid_indices
    
    def extract_features_from_directory(
        self,
        image_dir: Union[str, Path],
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        recursive: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from all images in directory.
        
        Args:
            image_dir: Directory containing images
            extensions: Valid image extensions
            recursive: Whether to search recursively
            
        Returns:
            Dictionary mapping image paths to feature vectors
        """
        image_dir = Path(image_dir)
        
        # Find all image files
        image_paths = []
        if recursive:
            for ext in extensions:
                image_paths.extend(image_dir.rglob(f"*{ext}"))
                image_paths.extend(image_dir.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                image_paths.extend(image_dir.glob(f"*{ext}"))
                image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        if not image_paths:
            return {}
        
        # Extract features
        start_time = time.time()
        features_list, valid_indices = self.extract_features_batch(
            [str(p) for p in image_paths]
        )
        
        extraction_time = time.time() - start_time
        
        # Create results dictionary
        results = {}
        for i, features in enumerate(features_list):
            original_idx = valid_indices[i]
            image_path = str(image_paths[original_idx])
            results[image_path] = features
        
        print(f"Feature extraction completed:")
        print(f"  Processed: {len(features_list)}/{len(image_paths)} images")
        print(f"  Time: {extraction_time:.2f}s")
        print(f"  Speed: {len(features_list)/extraction_time:.1f} images/s")
        
        return results
    
    def save_features(
        self,
        features_dict: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        format: str = 'npz'
    ):
        """
        Save extracted features to file.
        
        Args:
            features_dict: Dictionary of features
            output_path: Output file path
            format: Save format ('npz', 'npy', 'pkl')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            # Save as compressed numpy archive
            np.savez_compressed(output_path, **features_dict)
        elif format == 'npy':
            # Save as single numpy array (features only)
            features_array = np.array(list(features_dict.values()))
            np.save(output_path, features_array)
        elif format == 'pkl':
            # Save as pickle
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(features_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Features saved to: {output_path}")
    
    def load_features(
        self,
        input_path: Union[str, Path],
        format: str = 'npz'
    ) -> Dict[str, np.ndarray]:
        """
        Load features from file.
        
        Args:
            input_path: Input file path
            format: File format ('npz', 'npy', 'pkl')
            
        Returns:
            Dictionary of features
        """
        input_path = Path(input_path)
        
        if format == 'npz':
            data = np.load(input_path)
            return {key: data[key] for key in data.files}
        elif format == 'npy':
            features_array = np.load(input_path)
            # Return as numbered dictionary
            return {f"feature_{i}": feat for i, feat in enumerate(features_array)}
        elif format == 'pkl':
            import pickle
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")


class FeatureDatabase:
    """Database for storing and querying face features."""
    
    def __init__(self, feature_dim: int = 512):
        """
        Initialize feature database.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        self.feature_dim = feature_dim
        self.features = []
        self.labels = []
        self.metadata = []
        self.feature_index = {}  # Map from label to feature indices
    
    def add_features(
        self,
        features: Union[np.ndarray, List[np.ndarray]],
        labels: Union[int, str, List[Union[int, str]]],
        metadata: Optional[Union[Dict, List[Dict]]] = None
    ):
        """
        Add features to database.
        
        Args:
            features: Feature vectors
            labels: Labels for features
            metadata: Optional metadata for features
        """
        # Normalize inputs
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = [features]
            else:
                features = list(features)
        
        if not isinstance(labels, list):
            labels = [labels]
        
        if metadata is None:
            metadata = [{}] * len(features)
        elif not isinstance(metadata, list):
            metadata = [metadata]
        
        # Add to database
        for feat, label, meta in zip(features, labels, metadata):
            # Normalize feature
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            
            idx = len(self.features)
            self.features.append(feat)
            self.labels.append(label)
            self.metadata.append(meta)
            
            # Update index
            if label not in self.feature_index:
                self.feature_index[label] = []
            self.feature_index[label].append(idx)
    
    def search_similar(
        self,
        query_features: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[int, float, Union[int, str], Dict]]:
        """
        Search for similar features.
        
        Args:
            query_features: Query feature vector
            top_k: Number of top results to return
            threshold: Similarity threshold
            
        Returns:
            List of (index, similarity, label, metadata) tuples
        """
        if not self.features:
            return []
        
        # Normalize query
        query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        # Calculate similarities
        similarities = []
        for i, feat in enumerate(self.features):
            similarity = np.dot(query_features, feat)
            if similarity >= threshold:
                similarities.append((i, similarity, self.labels[i], self.metadata[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_features_by_label(self, label: Union[int, str]) -> List[np.ndarray]:
        """Get all features for a specific label."""
        if label not in self.feature_index:
            return []
        
        indices = self.feature_index[label]
        return [self.features[i] for i in indices]
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.features:
            return {
                'num_features': 0,
                'num_labels': 0,
                'feature_dim': self.feature_dim
            }
        
        features_array = np.array(self.features)
        
        return {
            'num_features': len(self.features),
            'num_labels': len(self.feature_index),
            'feature_dim': features_array.shape[1],
            'mean_norm': np.mean(np.linalg.norm(features_array, axis=1)),
            'labels': list(self.feature_index.keys())
        }
    
    def save(self, filepath: Union[str, Path]):
        """Save database to file."""
        import pickle
        
        data = {
            'features': self.features,
            'labels': self.labels,
            'metadata': self.metadata,
            'feature_index': self.feature_index,
            'feature_dim': self.feature_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Feature database saved to: {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """Load database from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features']
        self.labels = data['labels']
        self.metadata = data['metadata']
        self.feature_index = data['feature_index']
        self.feature_dim = data['feature_dim']
        
        print(f"Feature database loaded from: {filepath}")
        print(f"  Features: {len(self.features)}")
        print(f"  Labels: {len(self.feature_index)}")


def create_feature_extractor(
    model_path: str,
    device: str = 'auto',
    **kwargs
) -> BatchFeatureExtractor:
    """
    Create feature extractor with default settings.
    
    Args:
        model_path: Path to trained model
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured BatchFeatureExtractor
    """
    return BatchFeatureExtractor(
        model_path=model_path,
        device=device,
        **kwargs
    )