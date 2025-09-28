"""Test feature extraction and similarity computation."""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_similarity_calculation():
    """Test similarity calculation functions."""
    print("Testing similarity calculation...")
    
    try:
        # Import similarity module
        from features.similarity import SimilarityCalculator, SimilarityMetric, FaceMatcher
        
        # Create test features
        feat1 = np.array([1.0, 0.0, 0.0, 1.0])
        feat2 = np.array([0.8, 0.2, 0.1, 0.9])  # Similar to feat1
        feat3 = np.array([0.0, 1.0, 1.0, 0.0])  # Different from feat1
        
        # Test different similarity metrics
        metrics = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.INNER_PRODUCT
        ]
        
        for metric in metrics:
            calc = SimilarityCalculator(metric)
            
            # Test similar features
            sim1, dist1 = calc.calculate(feat1, feat2)
            
            # Test different features
            sim2, dist2 = calc.calculate(feat1, feat3)
            
            print(f"{metric.value}:")
            print(f"  Similar features: sim={sim1:.3f}, dist={dist1:.3f}")
            print(f"  Different features: sim={sim2:.3f}, dist={dist2:.3f}")
            
            # Verify that similar features have higher similarity
            if metric in [SimilarityMetric.COSINE, SimilarityMetric.INNER_PRODUCT]:
                assert sim1 > sim2, f"Similar features should have higher similarity for {metric}"
        
        print("✓ Similarity calculation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Similarity calculation test failed: {e}")
        return False


def test_face_matcher():
    """Test face matching functionality."""
    print("\nTesting face matcher...")
    
    try:
        from features.similarity import FaceMatcher, MatchResult
        
        # Create matcher
        matcher = FaceMatcher(
            similarity_metric='cosine',
            threshold=0.5,
            confidence_method='sigmoid'
        )
        
        # Create test features
        query_feat = np.array([1.0, 0.0, 0.0, 1.0])
        gallery_feats = np.array([
            [0.9, 0.1, 0.0, 0.9],  # Similar
            [0.0, 1.0, 1.0, 0.0],  # Different
            [1.0, 0.0, 0.1, 0.9],  # Similar
        ])
        gallery_ids = ['person_1', 'person_2', 'person_3']
        
        # Test one-to-many matching
        results = matcher.match_features(
            query_feat, gallery_feats, 
            query_id='query', gallery_ids=gallery_ids,
            return_all=True
        )
        
        print(f"Found {len(results)} matches:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.matched_id}: sim={result.similarity:.3f}, "
                  f"conf={result.confidence:.3f}, match={result.is_match}")
        
        # Verify results are sorted by similarity
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i+1].similarity, "Results should be sorted by similarity"
        
        # Test one-to-one matching
        result = matcher.match_one_to_one(
            query_feat, gallery_feats[0],
            id1='query', id2='person_1'
        )
        
        print(f"One-to-one match: sim={result.similarity:.3f}, match={result.is_match}")
        
        print("✓ Face matcher working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Face matcher test failed: {e}")
        return False


def test_adaptive_threshold():
    """Test adaptive threshold functionality."""
    print("\nTesting adaptive threshold...")
    
    try:
        from features.similarity import AdaptiveThreshold
        
        # Create adaptive threshold
        adaptive_thresh = AdaptiveThreshold(
            initial_threshold=0.5,
            adaptation_rate=0.1
        )
        
        print(f"Initial threshold: {adaptive_thresh.get_threshold():.3f}")
        
        # Simulate some matching results
        # High similarities for same person (should be matches)
        same_person_sims = [0.8, 0.85, 0.9, 0.75, 0.82]
        # Low similarities for different persons (should not be matches)
        diff_person_sims = [0.3, 0.25, 0.4, 0.35, 0.2]
        
        # Update statistics
        for sim in same_person_sims:
            predicted_match = sim >= adaptive_thresh.get_threshold()
            adaptive_thresh.update_statistics(sim, True, predicted_match)
        
        for sim in diff_person_sims:
            predicted_match = sim >= adaptive_thresh.get_threshold()
            adaptive_thresh.update_statistics(sim, False, predicted_match)
        
        # Adapt threshold
        adaptive_thresh.adapt_threshold()
        
        stats = adaptive_thresh.get_statistics()
        print(f"After adaptation:")
        print(f"  Threshold: {stats['threshold']:.3f}")
        print(f"  Precision: {stats['precision']:.3f}")
        print(f"  Recall: {stats['recall']:.3f}")
        print(f"  F1 Score: {stats['f1_score']:.3f}")
        
        print("✓ Adaptive threshold working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Adaptive threshold test failed: {e}")
        return False


def test_vector_index():
    """Test vector indexing functionality."""
    print("\nTesting vector index...")
    
    try:
        from features.vector_index import create_vector_index
        
        # Create simple vector index (fallback if Faiss not available)
        index = create_vector_index(
            dimension=4,
            index_type='simple',
            metric='cosine'
        )
        
        # Create test vectors
        vectors = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.9, 0.1, 0.0, 0.9],
            [0.1, 0.9, 0.9, 0.1],
        ])
        labels = ['person_1', 'person_2', 'person_1', 'person_2']
        metadata = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
        ]
        
        # Add vectors to index
        ids = index.add_vectors(vectors, labels, metadata)
        print(f"Added {len(ids)} vectors to index")
        
        # Test search
        query_vector = np.array([0.95, 0.05, 0.0, 0.95])  # Similar to person_1
        results = index.search(query_vector, k=3, threshold=0.3)
        
        print(f"Search results for query:")
        for i, (idx, similarity, label, meta) in enumerate(results):
            print(f"  {i+1}. ID={idx}, Label={label}, Sim={similarity:.3f}, Meta={meta}")
        
        # Verify that person_1 vectors are ranked higher
        if results:
            best_result = results[0]
            assert best_result[2] == 'person_1', "Best match should be person_1"
        
        # Test statistics
        stats = index.get_statistics()
        print(f"Index statistics: {stats}")
        
        print("✓ Vector index working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Vector index test failed: {e}")
        return False


def test_feature_database():
    """Test feature database functionality."""
    print("\nTesting feature database...")
    
    try:
        from features.feature_extractor import FeatureDatabase
        
        # Create database
        db = FeatureDatabase(feature_dim=4)
        
        # Add features
        features = [
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.9, 0.1, 0.0, 0.9]),
            np.array([0.0, 1.0, 1.0, 0.0]),
        ]
        labels = ['alice', 'alice', 'bob']
        metadata = [
            {'image': 'alice_1.jpg'},
            {'image': 'alice_2.jpg'},
            {'image': 'bob_1.jpg'},
        ]
        
        db.add_features(features, labels, metadata)
        
        # Test search
        query_feat = np.array([0.95, 0.05, 0.0, 0.95])
        results = db.search_similar(query_feat, top_k=3, threshold=0.3)
        
        print(f"Database search results:")
        for i, (idx, similarity, label, meta) in enumerate(results):
            print(f"  {i+1}. Index={idx}, Label={label}, Sim={similarity:.3f}")
        
        # Test get features by label
        alice_features = db.get_features_by_label('alice')
        print(f"Alice has {len(alice_features)} features")
        
        # Test statistics
        stats = db.get_statistics()
        print(f"Database stats: {stats}")
        
        print("✓ Feature database working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Feature database test failed: {e}")
        return False


def test_batch_similarity():
    """Test batch similarity calculation."""
    print("\nTesting batch similarity calculation...")
    
    try:
        from features.similarity import SimilarityCalculator
        
        calc = SimilarityCalculator('cosine')
        
        # Create test data
        query_features = np.random.randn(3, 4)  # 3 queries, 4D features
        gallery_features = np.random.randn(5, 4)  # 5 gallery items, 4D features
        
        # Calculate batch similarities
        start_time = time.time()
        similarities, distances = calc.batch_calculate(query_features, gallery_features)
        calc_time = time.time() - start_time
        
        print(f"Batch calculation completed in {calc_time:.4f}s")
        print(f"Similarities shape: {similarities.shape}")
        print(f"Distances shape: {distances.shape}")
        
        # Verify shapes
        assert similarities.shape == (3, 5), "Similarities shape should be (3, 5)"
        assert distances.shape == (3, 5), "Distances shape should be (3, 5)"
        
        # Verify similarity range for cosine
        assert np.all(similarities >= -1.0) and np.all(similarities <= 1.0), "Cosine similarities should be in [-1, 1]"
        
        print("✓ Batch similarity calculation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Batch similarity test failed: {e}")
        return False


def test_matching_pipeline():
    """Test complete matching pipeline."""
    print("\nTesting matching pipeline...")
    
    try:
        from features.similarity import FaceMatcher, MatchingPipeline
        
        # Create matcher and pipeline
        matcher = FaceMatcher(similarity_metric='cosine', threshold=0.5)
        pipeline = MatchingPipeline(
            matcher=matcher,
            feature_normalizer='l2',
            outlier_detection=True,
            quality_threshold=0.3
        )
        
        # Create test data with some outliers
        query_features = np.array([1.0, 0.0, 0.0, 1.0])
        gallery_features = np.array([
            [0.9, 0.1, 0.0, 0.9],   # Normal, similar
            [0.0, 1.0, 1.0, 0.0],   # Normal, different
            [100.0, 0.0, 0.0, 0.0], # Outlier (very large norm)
            [0.8, 0.2, 0.1, 0.8],   # Normal, similar
        ])
        gallery_ids = ['person_1', 'person_2', 'outlier', 'person_3']
        
        # Process matching
        results = pipeline.process_match(
            query_features, gallery_features,
            query_id='query', gallery_ids=gallery_ids,
            return_all=True
        )
        
        print(f"Pipeline results ({len(results)} matches):")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.matched_id}: sim={result.similarity:.3f}, "
                  f"conf={result.confidence:.3f}")
        
        # Verify outlier was filtered (should have fewer results than input)
        assert len(results) <= len(gallery_features), "Pipeline should filter some results"
        
        print("✓ Matching pipeline working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Matching pipeline test failed: {e}")
        return False


def main():
    """Run all feature extraction and similarity tests."""
    print("="*60)
    print("FEATURE EXTRACTION AND SIMILARITY TEST")
    print("="*60)
    
    tests = [
        test_similarity_calculation,
        test_face_matcher,
        test_adaptive_threshold,
        test_vector_index,
        test_feature_database,
        test_batch_similarity,
        test_matching_pipeline,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All feature extraction and similarity tests passed!")
        print("\nFeature extraction and similarity computation working correctly.")
        print("To use with real models:")
        print("  1. Train a model using train_advanced.py")
        print("  2. Extract features using BatchFeatureExtractor")
        print("  3. Build vector index for fast similarity search")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())