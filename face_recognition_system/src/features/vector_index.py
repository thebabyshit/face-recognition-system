"""Vector indexing system using Faiss for fast similarity search."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json

# Faiss import with fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: Faiss not available. Install with: pip install faiss-cpu")


class FaissVectorIndex:
    """High-performance vector index using Faiss."""
    
    def __init__(
        self,
        dimension: int = 512,
        index_type: str = 'flat',
        metric: str = 'cosine',
        use_gpu: bool = False
    ):
        """
        Initialize Faiss vector index.
        
        Args:
            dimension: Feature vector dimension
            index_type: Index type ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'ip')
            use_gpu: Whether to use GPU acceleration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is required for vector indexing")
        
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Create index
        self.index = self._create_index()
        
        # Metadata storage
        self.id_to_label = {}  # Map from internal ID to label
        self.label_to_ids = {}  # Map from label to list of internal IDs
        self.metadata = {}  # Additional metadata for each ID
        self.next_id = 0
        
        print(f"Faiss index created: {self.index_type}, {self.metric}, dim={dimension}")
        if self.use_gpu:
            print("GPU acceleration enabled")
    
    def _create_index(self):
        """Create Faiss index based on configuration."""
        if self.metric == 'cosine':
            # For cosine similarity, use inner product with normalized vectors
            if self.index_type == 'flat':
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        
        elif self.metric == 'l2':
            if self.index_type == 'flat':
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        
        elif self.metric == 'ip':
            # Inner product
            if self.index_type == 'flat':
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for cosine similarity."""
        if self.metric == 'cosine':
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            return features / norms
        return features
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        labels: Union[List[Union[int, str]], np.ndarray],
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Feature vectors to add [N, D]
            labels: Labels for each vector
            metadata: Optional metadata for each vector
            
        Returns:
            List of internal IDs assigned to the vectors
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if not isinstance(labels, list):
            labels = list(labels)
        
        if metadata is None:
            metadata = [{}] * len(vectors)
        
        # Normalize vectors if needed
        vectors = self._normalize_features(vectors.astype(np.float32))
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(vectors) >= 100:  # Need enough data for training
                self.index.train(vectors)
            else:
                print("Warning: Not enough vectors for IVF training")
        
        # Add vectors to index
        start_id = self.next_id
        self.index.add(vectors)
        
        # Update metadata
        assigned_ids = []
        for i, (label, meta) in enumerate(zip(labels, metadata)):
            internal_id = start_id + i
            assigned_ids.append(internal_id)
            
            # Update mappings
            self.id_to_label[internal_id] = label
            if label not in self.label_to_ids:
                self.label_to_ids[label] = []
            self.label_to_ids[label].append(internal_id)
            self.metadata[internal_id] = meta
        
        self.next_id += len(vectors)
        
        return assigned_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float, Union[int, str], Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector [D]
            k: Number of nearest neighbors to return
            threshold: Optional similarity threshold
            
        Returns:
            List of (internal_id, similarity, label, metadata) tuples
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query vector
        query_vector = self._normalize_features(query_vector.astype(np.float32))
        
        # Search
        similarities, indices = self.index.search(query_vector, k)
        
        # Process results
        results = []
        for i in range(len(indices[0])):
            internal_id = indices[0][i]
            similarity = float(similarities[0][i])
            
            # Skip invalid results
            if internal_id == -1:
                continue
            
            # Apply threshold if specified
            if threshold is not None:
                if self.metric == 'l2':
                    # For L2, smaller is better, so invert threshold logic
                    if similarity > threshold:
                        continue
                else:
                    # For cosine/IP, larger is better
                    if similarity < threshold:
                        continue
            
            # Get metadata
            label = self.id_to_label.get(internal_id, internal_id)
            metadata = self.metadata.get(internal_id, {})
            
            results.append((internal_id, similarity, label, metadata))
        
        return results
    
    def search_by_label(
        self,
        query_vector: np.ndarray,
        target_labels: List[Union[int, str]],
        k: int = 10
    ) -> List[Tuple[int, float, Union[int, str], Dict]]:
        """
        Search for vectors with specific labels.
        
        Args:
            query_vector: Query vector
            target_labels: Labels to search within
            k: Number of results per label
            
        Returns:
            List of results sorted by similarity
        """
        all_results = []
        
        # Search in full index first
        candidates = self.search(query_vector, k * len(target_labels) * 2)
        
        # Filter by target labels
        label_counts = {label: 0 for label in target_labels}
        
        for result in candidates:
            internal_id, similarity, label, metadata = result
            
            if label in target_labels and label_counts[label] < k:
                all_results.append(result)
                label_counts[label] += 1
                
                # Stop if we have enough results for all labels
                if all(count >= k for count in label_counts.values()):
                    break
        
        return all_results
    
    def remove_by_label(self, label: Union[int, str]) -> int:
        """
        Remove all vectors with a specific label.
        Note: This requires rebuilding the index.
        
        Args:
            label: Label to remove
            
        Returns:
            Number of vectors removed
        """
        if label not in self.label_to_ids:
            return 0
        
        # Get IDs to remove
        ids_to_remove = set(self.label_to_ids[label])
        
        # Collect remaining vectors and metadata
        remaining_vectors = []
        remaining_labels = []
        remaining_metadata = []
        
        # This is inefficient but necessary since Faiss doesn't support removal
        print("Warning: Removing vectors requires rebuilding the index")
        
        # We would need to store original vectors to rebuild
        # For now, just update metadata
        for internal_id in ids_to_remove:
            if internal_id in self.id_to_label:
                del self.id_to_label[internal_id]
            if internal_id in self.metadata:
                del self.metadata[internal_id]
        
        del self.label_to_ids[label]
        
        return len(ids_to_remove)
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'num_labels': len(self.label_to_ids),
            'use_gpu': self.use_gpu,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
    
    def save(self, filepath: Union[str, Path]):
        """Save index and metadata to files."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        index_path = filepath.with_suffix('.faiss')
        if self.use_gpu:
            # Move to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = filepath.with_suffix('.json')
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'use_gpu': self.use_gpu,
            'next_id': self.next_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()},
            'label_to_ids': {str(k): v for k, v in self.label_to_ids.items()},
            'metadata': {str(k): v for k, v in self.metadata.items()}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
    
    def load(self, filepath: Union[str, Path]):
        """Load index and metadata from files."""
        filepath = Path(filepath)
        
        # Load Faiss index
        index_path = filepath.with_suffix('.faiss')
        self.index = faiss.read_index(str(index_path))
        
        # Move to GPU if requested
        if self.use_gpu and not hasattr(self.index, 'getDevice'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.next_id = metadata['next_id']
        
        # Convert string keys back to integers where appropriate
        self.id_to_label = {int(k): v for k, v in metadata['id_to_label'].items()}
        self.label_to_ids = {
            (int(k) if k.isdigit() else k): v 
            for k, v in metadata['label_to_ids'].items()
        }
        self.metadata = {int(k): v for k, v in metadata['metadata'].items()}
        
        print(f"Index loaded from: {index_path}")
        print(f"Metadata loaded from: {metadata_path}")
        print(f"Total vectors: {self.index.ntotal}")


class SimpleVectorIndex:
    """Simple vector index implementation without Faiss dependency."""
    
    def __init__(self, dimension: int = 512, metric: str = 'cosine'):
        """
        Initialize simple vector index.
        
        Args:
            dimension: Feature vector dimension
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.metric = metric.lower()
        
        # Storage
        self.vectors = []
        self.labels = []
        self.metadata = []
        self.label_to_indices = {}
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        labels: Union[List[Union[int, str]], np.ndarray],
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add vectors to the index."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if not isinstance(labels, list):
            labels = list(labels)
        
        if metadata is None:
            metadata = [{}] * len(vectors)
        
        # Normalize vectors for cosine similarity
        if self.metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vectors = vectors / norms
        
        # Add to storage
        assigned_ids = []
        for i, (vector, label, meta) in enumerate(zip(vectors, labels, metadata)):
            idx = len(self.vectors)
            assigned_ids.append(idx)
            
            self.vectors.append(vector)
            self.labels.append(label)
            self.metadata.append(meta)
            
            # Update label index
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        return assigned_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float, Union[int, str], Dict]]:
        """Search for similar vectors."""
        if not self.vectors:
            return []
        
        # Normalize query vector
        if self.metric == 'cosine':
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 1e-8:
                query_vector = query_vector / query_norm
        
        # Calculate similarities
        vectors_array = np.array(self.vectors)
        
        if self.metric == 'cosine' or self.metric == 'ip':
            similarities = np.dot(vectors_array, query_vector)
        elif self.metric == 'l2':
            similarities = -np.linalg.norm(vectors_array - query_vector, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Get top-k indices
        if len(similarities) <= k:
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -k)[-k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        
        # Build results
        results = []
        for idx in indices:
            similarity = float(similarities[idx])
            
            # Apply threshold
            if threshold is not None:
                if self.metric == 'l2':
                    if -similarity > threshold:  # L2 distance
                        continue
                else:
                    if similarity < threshold:
                        continue
            
            results.append((
                int(idx),
                similarity,
                self.labels[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'total_vectors': len(self.vectors),
            'dimension': self.dimension,
            'metric': self.metric,
            'num_labels': len(self.label_to_indices)
        }
    
    def save(self, filepath: Union[str, Path]):
        """Save index to file."""
        import pickle
        
        data = {
            'dimension': self.dimension,
            'metric': self.metric,
            'vectors': self.vectors,
            'labels': self.labels,
            'metadata': self.metadata,
            'label_to_indices': self.label_to_indices
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Simple index saved to: {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """Load index from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.vectors = data['vectors']
        self.labels = data['labels']
        self.metadata = data['metadata']
        self.label_to_indices = data['label_to_indices']
        
        print(f"Simple index loaded from: {filepath}")
        print(f"Total vectors: {len(self.vectors)}")


def create_vector_index(
    dimension: int = 512,
    index_type: str = 'auto',
    metric: str = 'cosine',
    use_gpu: bool = False
):
    """
    Create vector index with automatic fallback.
    
    Args:
        dimension: Feature vector dimension
        index_type: Index type ('auto', 'faiss', 'simple')
        metric: Distance metric
        use_gpu: Whether to use GPU (Faiss only)
        
    Returns:
        Vector index instance
    """
    if index_type == 'auto':
        if FAISS_AVAILABLE:
            index_type = 'faiss'
        else:
            index_type = 'simple'
    
    if index_type == 'faiss':
        if not FAISS_AVAILABLE:
            print("Faiss not available, falling back to simple index")
            return SimpleVectorIndex(dimension, metric)
        return FaissVectorIndex(dimension, 'flat', metric, use_gpu)
    elif index_type == 'simple':
        return SimpleVectorIndex(dimension, metric)
    else:
        raise ValueError(f"Unknown index type: {index_type}")