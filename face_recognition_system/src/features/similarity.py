"""Similarity computation and matching for face recognition."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class SimilarityMetric(Enum):
    """Supported similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    INNER_PRODUCT = "inner_product"
    NORMALIZED_EUCLIDEAN = "normalized_euclidean"


@dataclass
class MatchResult:
    """Result of a face matching operation."""
    
    query_id: Union[int, str]
    matched_id: Union[int, str]
    similarity: float
    confidence: float
    distance: float
    is_match: bool
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'query_id': self.query_id,
            'matched_id': self.matched_id,
            'similarity': self.similarity,
            'confidence': self.confidence,
            'distance': self.distance,
            'is_match': self.is_match,
            'metadata': self.metadata or {}
        }


class SimilarityCalculator:
    """Advanced similarity calculation with multiple metrics."""
    
    def __init__(self, metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE):
        """
        Initialize similarity calculator.
        
        Args:
            metric: Similarity metric to use
        """
        if isinstance(metric, str):
            metric = SimilarityMetric(metric.lower())
        
        self.metric = metric
        self._similarity_func = self._get_similarity_function()
    
    def _get_similarity_function(self) -> Callable:
        """Get the appropriate similarity function."""
        if self.metric == SimilarityMetric.COSINE:
            return self._cosine_similarity
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity
        elif self.metric == SimilarityMetric.MANHATTAN:
            return self._manhattan_similarity
        elif self.metric == SimilarityMetric.INNER_PRODUCT:
            return self._inner_product_similarity
        elif self.metric == SimilarityMetric.NORMALIZED_EUCLIDEAN:
            return self._normalized_euclidean_similarity
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """Calculate cosine similarity and distance."""
        # Normalize features
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0, 2.0  # Maximum distance for zero vectors
        
        feat1_norm = feat1 / norm1
        feat2_norm = feat2 / norm2
        
        similarity = np.dot(feat1_norm, feat2_norm)
        distance = 1.0 - similarity  # Cosine distance
        
        return float(similarity), float(distance)
    
    def _euclidean_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """Calculate Euclidean-based similarity and distance."""
        distance = np.linalg.norm(feat1 - feat2)
        # Convert distance to similarity (0-1 range)
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity), float(distance)
    
    def _manhattan_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """Calculate Manhattan-based similarity and distance."""
        distance = np.sum(np.abs(feat1 - feat2))
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity), float(distance)
    
    def _inner_product_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """Calculate inner product similarity."""
        similarity = np.dot(feat1, feat2)
        distance = -similarity  # Negative inner product as distance
        
        return float(similarity), float(distance)
    
    def _normalized_euclidean_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """Calculate normalized Euclidean similarity."""
        # Normalize features first
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        distance = np.linalg.norm(feat1_norm - feat2_norm)
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity), float(distance)
    
    def calculate(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate similarity and distance between two features.
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Tuple of (similarity, distance)
        """
        return self._similarity_func(feat1, feat2)
    
    def batch_calculate(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate similarities between query and gallery features.
        
        Args:
            query_features: Query feature vectors [N, D]
            gallery_features: Gallery feature vectors [M, D]
            
        Returns:
            Tuple of (similarities [N, M], distances [N, M])
        """
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        if gallery_features.ndim == 1:
            gallery_features = gallery_features.reshape(1, -1)
        
        n_queries = query_features.shape[0]
        n_gallery = gallery_features.shape[0]
        
        similarities = np.zeros((n_queries, n_gallery))
        distances = np.zeros((n_queries, n_gallery))
        
        for i in range(n_queries):
            for j in range(n_gallery):
                sim, dist = self.calculate(query_features[i], gallery_features[j])
                similarities[i, j] = sim
                distances[i, j] = dist
        
        return similarities, distances


class AdaptiveThreshold:
    """Adaptive threshold calculation for face matching."""
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ):
        """
        Initialize adaptive threshold.
        
        Args:
            initial_threshold: Initial threshold value
            adaptation_rate: Rate of threshold adaptation
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Statistics for adaptation
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.total_comparisons = 0
    
    def update_statistics(self, similarity: float, is_same_person: bool, predicted_match: bool):
        """Update statistics for threshold adaptation."""
        self.total_comparisons += 1
        
        if is_same_person and predicted_match:
            self.true_positives += 1
        elif is_same_person and not predicted_match:
            self.false_negatives += 1
        elif not is_same_person and predicted_match:
            self.false_positives += 1
        else:
            self.true_negatives += 1
    
    def adapt_threshold(self):
        """Adapt threshold based on current statistics."""
        if self.total_comparisons < 10:
            return  # Need more data
        
        # Calculate current metrics
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        
        # Adjust threshold based on precision-recall balance
        if precision < 0.8 and self.false_positives > 0:
            # Too many false positives, increase threshold
            adjustment = self.adaptation_rate
        elif recall < 0.8 and self.false_negatives > 0:
            # Too many false negatives, decrease threshold
            adjustment = -self.adaptation_rate
        else:
            adjustment = 0
        
        # Apply adjustment
        new_threshold = self.threshold + adjustment
        self.threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)
    
    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.threshold
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        if self.total_comparisons == 0:
            return {
                'threshold': self.threshold,
                'total_comparisons': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'threshold': self.threshold,
            'total_comparisons': self.total_comparisons,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


class FaceMatcher:
    """High-level face matching with confidence estimation."""
    
    def __init__(
        self,
        similarity_metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
        threshold: float = 0.5,
        use_adaptive_threshold: bool = False,
        confidence_method: str = 'sigmoid'
    ):
        """
        Initialize face matcher.
        
        Args:
            similarity_metric: Similarity metric to use
            threshold: Matching threshold
            use_adaptive_threshold: Whether to use adaptive thresholding
            confidence_method: Method for confidence calculation
        """
        self.similarity_calc = SimilarityCalculator(similarity_metric)
        self.base_threshold = threshold
        self.confidence_method = confidence_method
        
        # Adaptive threshold
        if use_adaptive_threshold:
            self.adaptive_threshold = AdaptiveThreshold(threshold)
        else:
            self.adaptive_threshold = None
    
    def _calculate_confidence(self, similarity: float, distance: float) -> float:
        """Calculate confidence score from similarity."""
        if self.confidence_method == 'sigmoid':
            # Sigmoid-based confidence
            return 1.0 / (1.0 + np.exp(-10 * (similarity - 0.5)))
        elif self.confidence_method == 'linear':
            # Linear mapping
            return np.clip(similarity, 0.0, 1.0)
        elif self.confidence_method == 'exponential':
            # Exponential confidence
            return 1.0 - np.exp(-5 * similarity)
        else:
            return similarity
    
    def match_features(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        query_id: Union[int, str] = None,
        gallery_ids: List[Union[int, str]] = None,
        return_all: bool = False
    ) -> Union[MatchResult, List[MatchResult]]:
        """
        Match query features against gallery.
        
        Args:
            query_features: Query feature vector
            gallery_features: Gallery feature vectors
            query_id: Query identifier
            gallery_ids: Gallery identifiers
            return_all: Whether to return all matches or just the best
            
        Returns:
            MatchResult or list of MatchResults
        """
        if gallery_features.ndim == 1:
            gallery_features = gallery_features.reshape(1, -1)
        
        if gallery_ids is None:
            gallery_ids = list(range(len(gallery_features)))
        
        # Calculate similarities
        similarities, distances = self.similarity_calc.batch_calculate(
            query_features.reshape(1, -1), gallery_features
        )
        
        similarities = similarities[0]  # Remove batch dimension
        distances = distances[0]
        
        # Get current threshold
        current_threshold = self.base_threshold
        if self.adaptive_threshold:
            current_threshold = self.adaptive_threshold.get_threshold()
        
        # Create match results
        results = []
        for i, (sim, dist, gal_id) in enumerate(zip(similarities, distances, gallery_ids)):
            confidence = self._calculate_confidence(sim, dist)
            is_match = sim >= current_threshold
            
            result = MatchResult(
                query_id=query_id,
                matched_id=gal_id,
                similarity=sim,
                confidence=confidence,
                distance=dist,
                is_match=is_match,
                metadata={'gallery_index': i}
            )
            
            results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        if return_all:
            return results
        else:
            return results[0] if results else None
    
    def match_one_to_one(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        id1: Union[int, str] = None,
        id2: Union[int, str] = None
    ) -> MatchResult:
        """
        Match two feature vectors one-to-one.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            id1: First identifier
            id2: Second identifier
            
        Returns:
            MatchResult
        """
        similarity, distance = self.similarity_calc.calculate(features1, features2)
        confidence = self._calculate_confidence(similarity, distance)
        
        current_threshold = self.base_threshold
        if self.adaptive_threshold:
            current_threshold = self.adaptive_threshold.get_threshold()
        
        is_match = similarity >= current_threshold
        
        return MatchResult(
            query_id=id1,
            matched_id=id2,
            similarity=similarity,
            confidence=confidence,
            distance=distance,
            is_match=is_match
        )
    
    def update_threshold(self, similarity: float, is_same_person: bool):
        """Update adaptive threshold with feedback."""
        if self.adaptive_threshold:
            predicted_match = similarity >= self.adaptive_threshold.get_threshold()
            self.adaptive_threshold.update_statistics(similarity, is_same_person, predicted_match)
            self.adaptive_threshold.adapt_threshold()
    
    def get_threshold_statistics(self) -> Dict:
        """Get threshold adaptation statistics."""
        if self.adaptive_threshold:
            return self.adaptive_threshold.get_statistics()
        else:
            return {'threshold': self.base_threshold, 'adaptive': False}


class MatchingPipeline:
    """Complete face matching pipeline with preprocessing and post-processing."""
    
    def __init__(
        self,
        matcher: FaceMatcher,
        feature_normalizer: Optional[str] = 'l2',
        outlier_detection: bool = True,
        quality_threshold: float = 0.3
    ):
        """
        Initialize matching pipeline.
        
        Args:
            matcher: Face matcher instance
            feature_normalizer: Feature normalization method
            outlier_detection: Whether to detect outlier features
            quality_threshold: Minimum feature quality threshold
        """
        self.matcher = matcher
        self.feature_normalizer = feature_normalizer
        self.outlier_detection = outlier_detection
        self.quality_threshold = quality_threshold
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if self.feature_normalizer == 'l2':
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            return features / (norms + 1e-8)
        elif self.feature_normalizer == 'l1':
            norms = np.sum(np.abs(features), axis=-1, keepdims=True)
            return features / (norms + 1e-8)
        elif self.feature_normalizer == 'standardize':
            mean = np.mean(features, axis=-1, keepdims=True)
            std = np.std(features, axis=-1, keepdims=True)
            return (features - mean) / (std + 1e-8)
        else:
            return features
    
    def _detect_outliers(self, features: np.ndarray) -> np.ndarray:
        """Detect outlier features using statistical methods."""
        if features.ndim == 1:
            return np.array([False])
        
        # Calculate feature norms
        norms = np.linalg.norm(features, axis=1)
        
        # Use IQR method for outlier detection
        q1 = np.percentile(norms, 25)
        q3 = np.percentile(norms, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (norms < lower_bound) | (norms > upper_bound)
        
        return outliers
    
    def process_match(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        query_id: Union[int, str] = None,
        gallery_ids: List[Union[int, str]] = None,
        return_all: bool = False
    ) -> Union[MatchResult, List[MatchResult]]:
        """
        Process complete matching pipeline.
        
        Args:
            query_features: Query feature vector
            gallery_features: Gallery feature vectors
            query_id: Query identifier
            gallery_ids: Gallery identifiers
            return_all: Whether to return all matches
            
        Returns:
            MatchResult or list of MatchResults
        """
        # Normalize features
        if self.feature_normalizer:
            query_features = self._normalize_features(query_features)
            gallery_features = self._normalize_features(gallery_features)
        
        # Outlier detection
        if self.outlier_detection and gallery_features.ndim > 1:
            outliers = self._detect_outliers(gallery_features)
            if np.any(outliers):
                # Filter out outliers
                valid_indices = ~outliers
                gallery_features = gallery_features[valid_indices]
                if gallery_ids:
                    gallery_ids = [gallery_ids[i] for i in range(len(gallery_ids)) if valid_indices[i]]
        
        # Perform matching
        results = self.matcher.match_features(
            query_features, gallery_features, query_id, gallery_ids, return_all=True
        )
        
        # Post-process results
        filtered_results = []
        for result in results:
            # Apply quality threshold
            if result.confidence >= self.quality_threshold:
                filtered_results.append(result)
        
        if not filtered_results:
            # Return best result even if below threshold, but mark as no match
            if results:
                best_result = results[0]
                best_result.is_match = False
                filtered_results = [best_result]
        
        if return_all:
            return filtered_results
        else:
            return filtered_results[0] if filtered_results else None


def create_face_matcher(
    similarity_metric: str = 'cosine',
    threshold: float = 0.5,
    adaptive: bool = False,
    **kwargs
) -> FaceMatcher:
    """
    Create face matcher with default settings.
    
    Args:
        similarity_metric: Similarity metric to use
        threshold: Matching threshold
        adaptive: Whether to use adaptive thresholding
        **kwargs: Additional arguments
        
    Returns:
        Configured FaceMatcher
    """
    return FaceMatcher(
        similarity_metric=similarity_metric,
        threshold=threshold,
        use_adaptive_threshold=adaptive,
        **kwargs
    )