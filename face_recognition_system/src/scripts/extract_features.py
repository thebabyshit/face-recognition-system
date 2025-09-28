"""Feature extraction script for face recognition."""

import argparse
import sys
from pathlib import Path
import json
import time
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from features.feature_extractor import BatchFeatureExtractor, FeatureDatabase
    from features.vector_index import create_vector_index
    from features.similarity import create_face_matcher
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


def extract_features_from_dataset(
    model_path: str,
    data_root: str,
    output_dir: str,
    batch_size: int = 32,
    use_face_detection: bool = True,
    quality_threshold: float = 0.3
):
    """Extract features from entire dataset."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not available")
        return False
    
    print("="*60)
    print("FEATURE EXTRACTION FROM DATASET")
    print("="*60)
    
    # Create feature extractor
    extractor = BatchFeatureExtractor(
        model_path=model_path,
        batch_size=batch_size,
        use_face_detection=use_face_detection,
        quality_threshold=quality_threshold
    )
    
    # Extract features from each person directory
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = {}
    person_features = {}
    
    # Process each person directory
    person_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()]
    person_dirs = sorted(person_dirs)[:10]  # Limit to first 10 for demo
    
    print(f"Processing {len(person_dirs)} person directories...")
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        print(f"\nProcessing person {person_id}...")
        
        # Extract features for this person
        features_dict = extractor.extract_features_from_directory(
            person_dir, recursive=False
        )
        
        if features_dict:
            # Store features
            all_features.update(features_dict)
            person_features[person_id] = list(features_dict.values())
            
            print(f"  Extracted {len(features_dict)} features for person {person_id}")
        else:
            print(f"  No valid features extracted for person {person_id}")
    
    # Save all features
    if all_features:
        features_path = output_dir / "all_features.npz"
        extractor.save_features(all_features, features_path, format='npz')
        
        # Save person-wise features
        person_features_path = output_dir / "person_features.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_person_features = {}
        for person_id, features_list in person_features.items():
            serializable_person_features[person_id] = [
                feat.tolist() for feat in features_list
            ]
        
        with open(person_features_path, 'w') as f:
            json.dump(serializable_person_features, f, indent=2)
        
        print(f"\nFeature extraction completed:")
        print(f"  Total features: {len(all_features)}")
        print(f"  People processed: {len(person_features)}")
        print(f"  Features saved to: {features_path}")
        print(f"  Person mapping saved to: {person_features_path}")
        
        return True
    else:
        print("No features extracted!")
        return False


def build_feature_index(
    features_file: str,
    person_mapping_file: str,
    output_path: str,
    index_type: str = 'simple'
):
    """Build vector index from extracted features."""
    print("\n" + "="*60)
    print("BUILDING FEATURE INDEX")
    print("="*60)
    
    # Load features
    print(f"Loading features from: {features_file}")
    features_data = np.load(features_file)
    
    # Load person mapping
    print(f"Loading person mapping from: {person_mapping_file}")
    with open(person_mapping_file, 'r') as f:
        person_features = json.load(f)
    
    # Create vector index
    feature_dim = list(features_data.values())[0].shape[0]
    index = create_vector_index(
        dimension=feature_dim,
        index_type=index_type,
        metric='cosine'
    )
    
    # Add features to index
    all_vectors = []
    all_labels = []
    all_metadata = []
    
    for person_id, features_list in person_features.items():
        for i, features in enumerate(features_list):
            all_vectors.append(np.array(features))
            all_labels.append(person_id)
            all_metadata.append({'person_id': person_id, 'feature_index': i})
    
    if all_vectors:
        vectors_array = np.array(all_vectors)
        index.add_vectors(vectors_array, all_labels, all_metadata)
        
        # Save index
        index.save(output_path)
        
        stats = index.get_statistics()
        print(f"Index built successfully:")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Number of people: {stats['num_labels']}")
        print(f"  Index saved to: {output_path}")
        
        return True
    else:
        print("No vectors to add to index!")
        return False


def test_feature_matching(
    model_path: str,
    index_path: str,
    test_image: str,
    top_k: int = 5
):
    """Test feature matching with a query image."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not available")
        return False
    
    print("\n" + "="*60)
    print("TESTING FEATURE MATCHING")
    print("="*60)
    
    # Create feature extractor
    extractor = BatchFeatureExtractor(model_path=model_path)
    
    # Extract features from test image
    print(f"Extracting features from: {test_image}")
    query_features = extractor.extract_features_single(test_image)
    
    if query_features is None:
        print("Failed to extract features from test image!")
        return False
    
    print(f"Query features shape: {query_features.shape}")
    
    # Load index
    print(f"Loading index from: {index_path}")
    index = create_vector_index(dimension=query_features.shape[0])
    index.load(index_path)
    
    # Search for similar features
    print(f"Searching for top-{top_k} matches...")
    results = index.search(query_features, k=top_k, threshold=0.3)
    
    print(f"\nMatching results:")
    for i, (idx, similarity, label, metadata) in enumerate(results):
        print(f"  {i+1}. Person {label}: similarity={similarity:.3f}")
    
    # Create face matcher for additional analysis
    matcher = create_face_matcher(similarity_metric='cosine', threshold=0.5)
    
    if results:
        best_match = results[0]
        print(f"\nBest match: Person {best_match[2]} (similarity: {best_match[1]:.3f})")
        
        # Calculate confidence
        match_result = matcher.match_one_to_one(
            query_features, 
            np.random.randn(query_features.shape[0]),  # Placeholder
            id1='query', 
            id2=best_match[2]
        )
        
        print(f"Match confidence: {match_result.confidence:.3f}")
        print(f"Is match: {match_result.is_match}")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Feature extraction and matching')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract features from dataset')
    extract_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model')
    extract_parser.add_argument('--data-root', type=str, default='../facecap',
                               help='Root directory of dataset')
    extract_parser.add_argument('--output-dir', type=str, default='features',
                               help='Output directory for features')
    extract_parser.add_argument('--batch-size', type=int, default=32,
                               help='Batch size for feature extraction')
    extract_parser.add_argument('--no-face-detection', action='store_true',
                               help='Disable face detection')
    extract_parser.add_argument('--quality-threshold', type=float, default=0.3,
                               help='Image quality threshold')
    
    # Build index command
    index_parser = subparsers.add_parser('index', help='Build feature index')
    index_parser.add_argument('--features-file', type=str, required=True,
                             help='Path to features file (.npz)')
    index_parser.add_argument('--person-mapping', type=str, required=True,
                             help='Path to person mapping file (.json)')
    index_parser.add_argument('--output-path', type=str, default='feature_index',
                             help='Output path for index')
    index_parser.add_argument('--index-type', type=str, default='simple',
                             choices=['simple', 'faiss'], help='Index type')
    
    # Test matching command
    test_parser = subparsers.add_parser('test', help='Test feature matching')
    test_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model')
    test_parser.add_argument('--index-path', type=str, required=True,
                            help='Path to feature index')
    test_parser.add_argument('--test-image', type=str, required=True,
                            help='Path to test image')
    test_parser.add_argument('--top-k', type=int, default=5,
                            help='Number of top matches to return')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        success = extract_features_from_dataset(
            model_path=args.model_path,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            use_face_detection=not args.no_face_detection,
            quality_threshold=args.quality_threshold
        )
        return 0 if success else 1
    
    elif args.command == 'index':
        success = build_feature_index(
            features_file=args.features_file,
            person_mapping_file=args.person_mapping,
            output_path=args.output_path,
            index_type=args.index_type
        )
        return 0 if success else 1
    
    elif args.command == 'test':
        success = test_feature_matching(
            model_path=args.model_path,
            index_path=args.index_path,
            test_image=args.test_image,
            top_k=args.top_k
        )
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())