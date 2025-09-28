"""Face recognition model implementation using ResNet50 with ArcFace loss."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Initialize ArcFace layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Number of classes
            scale: Feature scale
            margin: Angular margin penalty
            easy_margin: Use easy margin
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cosine and sine of margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ArcFace layer.
        
        Args:
            input: Input features [batch_size, in_features]
            label: Ground truth labels [batch_size]
            
        Returns:
            Output logits [batch_size, out_features]
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate phi (cosine with margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class ResNetBackbone(nn.Module):
    """ResNet backbone for face recognition."""
    
    def __init__(
        self,
        depth: int = 50,
        drop_ratio: float = 0.6,
        net_mode: str = 'ir',
        feat_dim: int = 512
    ):
        """
        Initialize ResNet backbone.
        
        Args:
            depth: ResNet depth (18, 34, 50, 101, 152)
            drop_ratio: Dropout ratio
            net_mode: Network mode ('ir' for improved ResNet)
            feat_dim: Feature dimension
        """
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet
        if depth == 18:
            resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            resnet = models.resnet34(pretrained=True)
        elif depth == 50:
            resnet = models.resnet50(pretrained=True)
        elif depth == 101:
            resnet = models.resnet101(pretrained=True)
        elif depth == 152:
            resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add batch normalization and dropout
        self.bn1 = nn.BatchNorm2d(resnet.fc.in_features)
        self.dropout = nn.Dropout(drop_ratio)
        
        # Feature projection layer
        self.fc = nn.Linear(resnet.fc.in_features, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet backbone.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Feature embeddings [batch_size, feat_dim]
        """
        # Extract features
        x = self.features(x)
        x = self.bn1(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout and feature projection
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)
        
        return x


class FaceRecognitionModel(nn.Module):
    """Complete face recognition model with ResNet backbone and ArcFace head."""
    
    def __init__(
        self,
        num_classes: int,
        backbone_depth: int = 50,
        feat_dim: int = 512,
        drop_ratio: float = 0.6,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Initialize face recognition model.
        
        Args:
            num_classes: Number of identity classes
            backbone_depth: ResNet backbone depth
            feat_dim: Feature embedding dimension
            drop_ratio: Dropout ratio
            scale: ArcFace scale parameter
            margin: ArcFace margin parameter
            easy_margin: Use easy margin in ArcFace
        """
        super(FaceRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Backbone network
        self.backbone = ResNetBackbone(
            depth=backbone_depth,
            drop_ratio=drop_ratio,
            feat_dim=feat_dim
        )
        
        # ArcFace head
        self.head = ArcMarginProduct(
            in_features=feat_dim,
            out_features=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin
        )
    
    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            label: Ground truth labels [batch_size] (required for training)
            
        Returns:
            Tuple of (features, logits)
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply ArcFace head if labels provided (training mode)
        if label is not None:
            logits = self.head(features, label)
        else:
            # Inference mode: compute cosine similarity with class centers
            logits = F.linear(F.normalize(features), F.normalize(self.head.weight))
        
        return features, logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized feature embeddings.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Normalized feature embeddings [batch_size, feat_dim]
        """
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)


class FaceRecognizer:
    """High-level interface for face recognition."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 500,
        feat_dim: int = 512,
        device: str = 'cpu'
    ):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to trained model weights
            num_classes: Number of identity classes
            feat_dim: Feature embedding dimension
            device: Device to run inference on
        """
        self.device = device
        self.feat_dim = feat_dim
        
        # Initialize model
        self.model = FaceRecognitionModel(
            num_classes=num_classes,
            feat_dim=feat_dim
        ).to(device)
        
        # Load pretrained weights if provided
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load model weights from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from face images.
        
        Args:
            images: Input face images [batch_size, 3, height, width]
            
        Returns:
            Normalized feature embeddings [batch_size, feat_dim]
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model.extract_features(images)
            return features.cpu()
    
    def compare_features(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> float:
        """
        Compare two feature embeddings using cosine similarity.
        
        Args:
            feat1: First feature embedding [feat_dim]
            feat2: Second feature embedding [feat_dim]
            
        Returns:
            Cosine similarity score
        """
        # Ensure features are normalized
        feat1 = F.normalize(feat1, p=2, dim=-1)
        feat2 = F.normalize(feat2, p=2, dim=-1)
        
        # Calculate cosine similarity
        similarity = torch.dot(feat1, feat2).item()
        return similarity
    
    def identify_person(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_labels: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[Optional[int], float]:
        """
        Identify person from gallery using query features.
        
        Args:
            query_features: Query feature embedding [feat_dim]
            gallery_features: Gallery feature embeddings [num_gallery, feat_dim]
            gallery_labels: Gallery labels [num_gallery]
            threshold: Similarity threshold for identification
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        # Calculate similarities with all gallery features
        query_features = F.normalize(query_features, p=2, dim=-1)
        gallery_features = F.normalize(gallery_features, p=2, dim=-1)
        
        similarities = torch.mm(query_features.unsqueeze(0), gallery_features.t()).squeeze(0)
        
        # Find best match
        max_similarity, max_idx = torch.max(similarities, dim=0)
        
        if max_similarity.item() >= threshold:
            predicted_label = gallery_labels[max_idx].item()
            confidence = max_similarity.item()
            return predicted_label, confidence
        else:
            return None, max_similarity.item()


def create_face_recognition_model(
    num_classes: int = 500,
    backbone_depth: int = 50,
    feat_dim: int = 512,
    pretrained: bool = True
) -> FaceRecognitionModel:
    """
    Create a face recognition model with default settings.
    
    Args:
        num_classes: Number of identity classes
        backbone_depth: ResNet backbone depth
        feat_dim: Feature embedding dimension
        pretrained: Use pretrained backbone
        
    Returns:
        Configured FaceRecognitionModel
    """
    return FaceRecognitionModel(
        num_classes=num_classes,
        backbone_depth=backbone_depth,
        feat_dim=feat_dim
    )