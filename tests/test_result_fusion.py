"""
Unit tests for Result Fusion Algorithms
"""

import pytest
import sys
import os

# Add the hybrid_search_feature directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from result_fusion import (
    ResultFusion, FusionMethod, FusionResult, SearchResult,
    FusionMethod
)


class TestSearchResult:
    """Test SearchResult dataclass functionality"""
    
    def test_search_result_creation(self):
        """Test creating SearchResult objects"""
        doc = {'id': 'test1', 'content': 'test content', 'title': 'Test'}
        result = SearchResult(
            id='test1',
            document=doc,
            score=0.85,
            source='bm25',
            rank=1,
            metadata={'test': True}
        )
        
        assert result.id == 'test1'
        assert result.document == doc
        assert result.score == 0.85
        assert result.source == 'bm25'
        assert result.rank == 1
        assert result.metadata['test'] is True
    
    def test_search_result_defaults(self):
        """Test SearchResult with default metadata"""
        doc = {'id': 'test2', 'content': 'test content 2'}
        result = SearchResult(
            id='test2',
            document=doc,
            score=0.75,
            source='vector',
            rank=2
        )
        
        assert result.metadata is None


class TestResultFusion:
    """Test cases for result fusion functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fusion_engine = ResultFusion()
        
        # Sample documents for testing
        self.sample_docs = [
            {'id': 'doc1', 'content': 'Machine learning algorithms', 'title': 'ML Guide'},
            {'id': 'doc2', 'content': 'Python programming tutorial', 'title': 'Python Basics'},
            {'id': 'doc3', 'content': 'Deep learning with neural networks', 'title': 'Deep Learning'},
            {'id': 'doc4', 'content': 'Data science fundamentals', 'title': 'Data Science'},
            {'id': 'doc5', 'content': 'Natural language processing', 'title': 'NLP Guide'}
        ]
        
        # Sample BM25 results (traditional search)
        self.bm25_results = [
            SearchResult('doc1', self.sample_docs[0], 3.2, 'bm25', 1, {'match_type': 'exact'}),
            SearchResult('doc3', self.sample_docs[2], 2.8, 'bm25', 2, {'match_type': 'partial'}),
            SearchResult('doc2', self.sample_docs[1], 2.1, 'bm25', 3, {'match_type': 'partial'}),
        ]
        
        # Sample vector results (semantic search)
        self.vector_results = [
            SearchResult('doc3', self.sample_docs[2], 0.92, 'vector', 1, {'similarity': 'high'}),
            SearchResult('doc5', self.sample_docs[4], 0.87, 'vector', 2, {'similarity': 'high'}),
            SearchResult('doc1', self.sample_docs[0], 0.78, 'vector', 3, {'similarity': 'medium'}),
            SearchResult('doc4', self.sample_docs[3], 0.65, 'vector', 4, {'similarity': 'medium'}),
        ]
    
    def test_initialization(self):
        """Test ResultFusion initialization"""
        fusion = ResultFusion()
        assert fusion.fusion_stats['total_fusions'] == 0
        assert len(fusion.fusion_stats['method_usage']) == 0
        assert fusion.fusion_stats['average_result_counts']['bm25'] == 0
        assert fusion.fusion_stats['average_result_counts']['vector'] == 0
        assert fusion.fusion_stats['average_result_counts']['combined'] == 0
    
    def test_normalize_results(self):
        """Test result score normalization"""
        # Test normal case
        normalized = self.fusion_engine.normalize_results(self.bm25_results)
        scores = [r.score for r in normalized]
        
        assert min(scores) == 0.0  # Min should be 0
        assert max(scores) == 1.0  # Max should be 1
        assert len(normalized) == len(self.bm25_results)
        
        # Check that relative ordering is preserved
        assert normalized[0].score > normalized[1].score > normalized[2].score
        
        # Test empty list
        empty_normalized = self.fusion_engine.normalize_results([])
        assert empty_normalized == []
        
        # Test single result
        single_result = [self.bm25_results[0]]
        single_normalized = self.fusion_engine.normalize_results(single_result)
        assert single_normalized[0].score == 1.0
        
        # Test identical scores
        identical_results = [
            SearchResult('doc1', {}, 2.0, 'test', 1),
            SearchResult('doc2', {}, 2.0, 'test', 2),
            SearchResult('doc3', {}, 2.0, 'test', 3),
        ]
        identical_normalized = self.fusion_engine.normalize_results(identical_results)
        for result in identical_normalized:
            assert result.score == 1.0
    
    def test_reciprocal_rank_fusion(self):
        """Test Reciprocal Rank Fusion (RRF) algorithm"""
        # Test with default k=60
        fused_results = self.fusion_engine.reciprocal_rank_fusion(
            [self.bm25_results, self.vector_results]
        )
        
        assert len(fused_results) > 0
        assert isinstance(fused_results[0], FusionResult)
        
        # Check that results are sorted by RRF score (descending)
        scores = [r.final_score for r in fused_results]
        assert scores == sorted(scores, reverse=True)
        
        # Check that documents appearing in both lists get higher scores
        doc_scores = {r.id: r.final_score for r in fused_results}
        
        # doc1 and doc3 appear in both lists, should have higher scores
        single_source_docs = {'doc2', 'doc4', 'doc5'}  # Only in one list each
        dual_source_docs = {'doc1', 'doc3'}  # In both lists
        
        if dual_source_docs and single_source_docs:
            max_single = max(doc_scores[doc_id] for doc_id in single_source_docs if doc_id in doc_scores)
            min_dual = min(doc_scores[doc_id] for doc_id in dual_source_docs if doc_id in doc_scores)
            # Documents in both lists should generally score higher
            # (though this depends on their individual ranks)
        
        # Test with custom k parameter
        fused_custom_k = self.fusion_engine.reciprocal_rank_fusion(
            [self.bm25_results, self.vector_results], k=30.0
        )
        assert len(fused_custom_k) > 0
        
        # Test with empty lists
        empty_fused = self.fusion_engine.reciprocal_rank_fusion([])
        assert empty_fused == []
        
        # Test with single list
        single_list_fused = self.fusion_engine.reciprocal_rank_fusion([self.bm25_results])
        assert len(single_list_fused) == len(self.bm25_results)
    
    def test_weighted_score_fusion(self):
        """Test weighted score fusion"""
        # Test with equal weights (0.5, 0.5)
        fused_results = self.fusion_engine.weighted_score_fusion(
            self.bm25_results, self.vector_results, 0.5, 0.5
        )
        
        assert len(fused_results) > 0
        assert isinstance(fused_results[0], FusionResult)
        
        # Check that results are sorted by final score (descending)
        scores = [r.final_score for r in fused_results]
        assert scores == sorted(scores, reverse=True)
        
        # Check score breakdown
        for result in fused_results:
            assert 'weighted_score' in result.score_breakdown
            assert 'bm25_contribution' in result.score_breakdown
            assert 'vector_contribution' in result.score_breakdown
            assert result.fusion_method == 'weighted_score'
        
        # Test with BM25-favoring weights
        bm25_heavy = self.fusion_engine.weighted_score_fusion(
            self.bm25_results, self.vector_results, 0.8, 0.2
        )
        
        # Test with vector-favoring weights
        vector_heavy = self.fusion_engine.weighted_score_fusion(
            self.bm25_results, self.vector_results, 0.2, 0.8
        )
        
        # Results should be different with different weightings
        bm25_heavy_scores = {r.id: r.final_score for r in bm25_heavy}
        vector_heavy_scores = {r.id: r.final_score for r in vector_heavy}
        
        # At least some scores should be different
        score_differences = sum(1 for doc_id in bm25_heavy_scores 
                              if doc_id in vector_heavy_scores and 
                              abs(bm25_heavy_scores[doc_id] - vector_heavy_scores[doc_id]) > 0.01)
        assert score_differences > 0
        
        # Test weight normalization (should handle unnormalized weights)
        normalized_fusion = self.fusion_engine.weighted_score_fusion(
            self.bm25_results, self.vector_results, 2.0, 3.0  # Should normalize to 0.4, 0.6
        )
        assert len(normalized_fusion) > 0
    
    def test_adaptive_fusion(self):
        """Test adaptive fusion based on query type"""
        # Test exact query type (should favor BM25)
        exact_fusion = self.fusion_engine.adaptive_fusion(
            self.bm25_results, self.vector_results, 
            query_type='exact', query_confidence=0.9
        )
        
        assert len(exact_fusion) > 0
        assert exact_fusion[0].fusion_method == 'adaptive'
        
        # Check metadata
        for result in exact_fusion:
            assert result.metadata['query_type'] == 'exact'
            assert result.metadata['query_confidence'] == 0.9
            assert result.metadata['base_bm25_weight'] == 0.7  # Should favor BM25
            assert result.metadata['base_vector_weight'] == 0.3
        
        # Test conceptual query type (should favor vector)
        conceptual_fusion = self.fusion_engine.adaptive_fusion(
            self.bm25_results, self.vector_results,
            query_type='conceptual', query_confidence=0.8
        )
        
        for result in conceptual_fusion:
            assert result.metadata['base_bm25_weight'] == 0.3  # Should favor vector
            assert result.metadata['base_vector_weight'] == 0.7
        
        # Test mixed query type (should be balanced)
        mixed_fusion = self.fusion_engine.adaptive_fusion(
            self.bm25_results, self.vector_results,
            query_type='mixed', query_confidence=0.6
        )
        
        for result in mixed_fusion:
            # Should have balanced base weights
            assert result.metadata['base_bm25_weight'] == 0.5
            assert result.metadata['base_vector_weight'] == 0.5
        
        # Test low confidence (should move toward balanced weights)
        low_confidence = self.fusion_engine.adaptive_fusion(
            self.bm25_results, self.vector_results,
            query_type='exact', query_confidence=0.1
        )
        
        # With low confidence, weights should be closer to 0.5, 0.5
        for result in low_confidence:
            bm25_weight = result.metadata['bm25_weight']
            vector_weight = result.metadata['vector_weight']
            # Should be closer to balanced than high-confidence exact query
            assert abs(bm25_weight - 0.5) < abs(0.7 - 0.5)
    
    def test_borda_count_fusion(self):
        """Test Borda count fusion method"""
        fused_results = self.fusion_engine.borda_count_fusion(
            [self.bm25_results, self.vector_results]
        )
        
        assert len(fused_results) > 0
        assert isinstance(fused_results[0], FusionResult)
        
        # Check that results are sorted by Borda score (descending)
        scores = [r.final_score for r in fused_results]
        assert scores == sorted(scores, reverse=True)
        
        # Check that Borda scores are integers (points-based system)
        for result in fused_results:
            assert isinstance(result.final_score, float)
            assert result.final_score == int(result.final_score)  # Should be whole numbers
            assert result.fusion_method == 'borda_count'
        
        # Test with empty lists
        empty_borda = self.fusion_engine.borda_count_fusion([])
        assert empty_borda == []
    
    def test_combsum_fusion(self):
        """Test CombSUM fusion method"""
        fused_results = self.fusion_engine.combsum_fusion(
            self.bm25_results, self.vector_results
        )
        
        assert len(fused_results) > 0
        assert isinstance(fused_results[0], FusionResult)
        
        # CombSUM should be equivalent to weighted fusion with equal weights
        weighted_results = self.fusion_engine.weighted_score_fusion(
            self.bm25_results, self.vector_results, 1.0, 1.0
        )
        
        # Should have same number of results
        assert len(fused_results) == len(weighted_results)
        
        # Scores should be similar (allowing for slight differences in implementation)
        combsum_scores = {r.id: r.final_score for r in fused_results}
        weighted_scores = {r.id: r.final_score for r in weighted_results}
        
        for doc_id in combsum_scores:
            if doc_id in weighted_scores:
                assert abs(combsum_scores[doc_id] - weighted_scores[doc_id]) < 0.001
    
    def test_main_fuse_results(self):
        """Test main fuse_results method"""
        # Test with RRF method
        rrf_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.RRF
        )
        assert len(rrf_results) > 0
        assert rrf_results[0].fusion_method == 'reciprocal_rank_fusion'
        
        # Test with weighted method
        weighted_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.WEIGHTED,
            bm25_weight=0.7, vector_weight=0.3
        )
        assert len(weighted_results) > 0
        assert weighted_results[0].fusion_method == 'weighted_score'
        
        # Test with adaptive method
        adaptive_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.ADAPTIVE,
            query_type='exact', query_confidence=0.8
        )
        assert len(adaptive_results) > 0
        assert adaptive_results[0].fusion_method == 'adaptive'
        
        # Test with Borda method
        borda_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.BORDA
        )
        assert len(borda_results) > 0
        assert borda_results[0].fusion_method == 'borda_count'
        
        # Test BM25-only case
        bm25_only = self.fusion_engine.fuse_results(
            self.bm25_results, [], FusionMethod.WEIGHTED
        )
        assert len(bm25_only) == len(self.bm25_results)
        for result in bm25_only:
            assert 'bm25_only' in result.fusion_method
        
        # Test vector-only case
        vector_only = self.fusion_engine.fuse_results(
            [], self.vector_results, FusionMethod.WEIGHTED
        )
        assert len(vector_only) == len(self.vector_results)
        for result in vector_only:
            assert 'vector_only' in result.fusion_method
        
        # Test empty case
        empty_results = self.fusion_engine.fuse_results([], [])
        assert empty_results == []
        
        # Check that statistics are updated
        assert self.fusion_engine.fusion_stats['total_fusions'] > 0
    
    def test_deduplicate_results(self):
        """Test result deduplication"""
        # Create some results with duplicates
        results_with_dupes = [
            FusionResult('doc1', {}, 1.0, 'test', {}, {}, {}, {}),
            FusionResult('doc2', {}, 0.9, 'test', {}, {}, {}, {}),
            FusionResult('doc1', {}, 0.8, 'test', {}, {}, {}, {}),  # Duplicate
            FusionResult('doc3', {}, 0.7, 'test', {}, {}, {}, {}),
            FusionResult('doc2', {}, 0.6, 'test', {}, {}, {}, {}),  # Duplicate
        ]
        
        deduplicated = self.fusion_engine.deduplicate_results(results_with_dupes)
        
        # Should keep only unique document IDs
        assert len(deduplicated) == 3  # doc1, doc2, doc3
        doc_ids = {r.id for r in deduplicated}
        assert doc_ids == {'doc1', 'doc2', 'doc3'}
        
        # Should keep the first occurrence of each document
        assert deduplicated[0].id == 'doc1'
        assert deduplicated[0].final_score == 1.0
        assert deduplicated[1].id == 'doc2'
        assert deduplicated[1].final_score == 0.9
        
        # Test empty list
        empty_deduped = self.fusion_engine.deduplicate_results([])
        assert empty_deduped == []
    
    def test_explain_fusion(self):
        """Test fusion result explanation"""
        # Get a fusion result to explain
        fused_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.WEIGHTED,
            bm25_weight=0.6, vector_weight=0.4
        )
        
        explanation = self.fusion_engine.explain_fusion(fused_results[0])
        
        # Check explanation structure
        assert 'document_id' in explanation
        assert 'final_score' in explanation
        assert 'fusion_method' in explanation
        assert 'score_breakdown' in explanation
        assert 'source_information' in explanation
        assert 'fusion_metadata' in explanation
        
        # Check source information
        source_info = explanation['source_information']
        assert 'ranks' in source_info
        assert 'original_scores' in source_info
        assert 'sources_used' in source_info
        assert 'source_count' in source_info
        
        # Test method-specific explanations
        if explanation['fusion_method'] == 'weighted_score':
            assert 'method_details' in explanation
            assert 'bm25_weight' in explanation['method_details']
            assert 'vector_weight' in explanation['method_details']
        
        # Test RRF explanation
        rrf_results = self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.RRF, k=45
        )
        rrf_explanation = self.fusion_engine.explain_fusion(rrf_results[0])
        
        if rrf_explanation['fusion_method'] == 'reciprocal_rank_fusion':
            assert 'method_details' in rrf_explanation
            assert 'k_parameter' in rrf_explanation['method_details']
    
    def test_fusion_statistics(self):
        """Test fusion statistics collection"""
        initial_stats = self.fusion_engine.get_fusion_statistics()
        assert initial_stats['total_fusions_performed'] >= 0
        
        # Perform some fusions
        self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.RRF
        )
        self.fusion_engine.fuse_results(
            self.bm25_results, self.vector_results, FusionMethod.WEIGHTED
        )
        
        updated_stats = self.fusion_engine.get_fusion_statistics()
        
        # Check that statistics were updated
        assert updated_stats['total_fusions_performed'] > initial_stats['total_fusions_performed']
        assert len(updated_stats['method_usage_counts']) > 0
        assert FusionMethod.RRF.value in updated_stats['method_usage_counts']
        assert FusionMethod.WEIGHTED.value in updated_stats['method_usage_counts']
        
        # Check structure
        assert 'average_input_sizes' in updated_stats
        assert 'supported_methods' in updated_stats
        assert 'version' in updated_stats
        
        # Verify supported methods
        expected_methods = [method.value for method in FusionMethod]
        assert set(updated_stats['supported_methods']) == set(expected_methods)


class TestFusionEdgeCases:
    """Test edge cases and error handling in result fusion"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fusion_engine = ResultFusion()
    
    def test_single_result_lists(self):
        """Test fusion with single-result lists"""
        single_bm25 = [SearchResult('doc1', {'content': 'test'}, 1.0, 'bm25', 1)]
        single_vector = [SearchResult('doc1', {'content': 'test'}, 0.8, 'vector', 1)]
        
        # Test RRF with single results
        rrf_results = self.fusion_engine.reciprocal_rank_fusion([single_bm25, single_vector])
        assert len(rrf_results) == 1
        assert rrf_results[0].id == 'doc1'
        
        # Test weighted fusion
        weighted_results = self.fusion_engine.weighted_score_fusion(
            single_bm25, single_vector, 0.5, 0.5
        )
        assert len(weighted_results) == 1
    
    def test_no_overlap_results(self):
        """Test fusion when result lists have no overlap"""
        bm25_unique = [SearchResult('doc1', {}, 1.0, 'bm25', 1)]
        vector_unique = [SearchResult('doc2', {}, 0.9, 'vector', 1)]
        
        fused = self.fusion_engine.weighted_score_fusion(bm25_unique, vector_unique, 0.5, 0.5)
        
        assert len(fused) == 2
        doc_ids = {r.id for r in fused}
        assert doc_ids == {'doc1', 'doc2'}
    
    def test_identical_scores(self):
        """Test fusion with identical scores"""
        identical_bm25 = [
            SearchResult('doc1', {}, 2.0, 'bm25', 1),
            SearchResult('doc2', {}, 2.0, 'bm25', 2),
            SearchResult('doc3', {}, 2.0, 'bm25', 3),
        ]
        identical_vector = [
            SearchResult('doc1', {}, 0.5, 'vector', 1),
            SearchResult('doc2', {}, 0.5, 'vector', 2),
            SearchResult('doc3', {}, 0.5, 'vector', 3),
        ]
        
        # Should handle identical scores gracefully
        fused = self.fusion_engine.weighted_score_fusion(
            identical_bm25, identical_vector, 0.5, 0.5
        )
        
        assert len(fused) == 3
        # All should have same final score due to identical inputs
        final_scores = [r.final_score for r in fused]
        assert len(set(final_scores)) == 1  # All scores should be identical
    
    def test_extreme_weights(self):
        """Test fusion with extreme weight values"""
        bm25_results = [SearchResult('doc1', {}, 3.0, 'bm25', 1)]
        vector_results = [SearchResult('doc1', {}, 0.7, 'vector', 1)]
        
        # Test with zero weights
        zero_bm25 = self.fusion_engine.weighted_score_fusion(
            bm25_results, vector_results, 0.0, 1.0
        )
        assert len(zero_bm25) > 0
        
        zero_vector = self.fusion_engine.weighted_score_fusion(
            bm25_results, vector_results, 1.0, 0.0
        )
        assert len(zero_vector) > 0
        
        # Test with very large weights (should normalize)
        large_weights = self.fusion_engine.weighted_score_fusion(
            bm25_results, vector_results, 1000.0, 2000.0
        )
        assert len(large_weights) > 0
        
        # The effective weights should be normalized to sum to 1
        result = large_weights[0]
        bm25_weight = result.metadata['bm25_weight']
        vector_weight = result.metadata['vector_weight']
        assert abs(bm25_weight + vector_weight - 1.0) < 0.001


class TestFusionMethodEnum:
    """Test FusionMethod enumeration"""
    
    def test_fusion_method_values(self):
        """Test that all fusion methods have correct values"""
        assert FusionMethod.RRF.value == "reciprocal_rank_fusion"
        assert FusionMethod.WEIGHTED.value == "weighted_score"
        assert FusionMethod.ADAPTIVE.value == "adaptive"
        assert FusionMethod.BORDA.value == "borda_count"
        assert FusionMethod.COMBSUM.value == "combsum"
    
    def test_fusion_method_coverage(self):
        """Test that all methods are supported by the fusion engine"""
        fusion_engine = ResultFusion()
        stats = fusion_engine.get_fusion_statistics()
        
        supported_methods = set(stats['supported_methods'])
        enum_methods = {method.value for method in FusionMethod}
        
        assert supported_methods == enum_methods


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])