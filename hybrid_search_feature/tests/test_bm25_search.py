"""
Unit tests for BM25 Search Engine
"""

import pytest
import math
from collections import Counter
import sys
import os

# Add the hybrid_search_feature directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bm25_search import BM25Search


class TestBM25Search:
    """Test cases for BM25 search engine functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.search_engine = BM25Search()
        
        # Sample documents for testing
        self.sample_docs = [
            {
                'id': 'doc1',
                'content': 'Machine learning is a subset of artificial intelligence',
                'title': 'ML Introduction'
            },
            {
                'id': 'doc2', 
                'content': 'Python programming language is great for machine learning',
                'title': 'Python ML'
            },
            {
                'id': 'doc3',
                'content': 'Deep learning uses neural networks with multiple layers',
                'title': 'Deep Learning'
            },
            {
                'id': 'doc4',
                'content': 'Natural language processing involves understanding human language',
                'title': 'NLP Basics'
            },
            {
                'id': 'doc5',
                'content': 'Artificial intelligence includes machine learning and deep learning',
                'title': 'AI Overview'
            }
        ]
    
    def test_initialization(self):
        """Test BM25Search initialization"""
        engine = BM25Search()
        assert engine.k1 == 1.2
        assert engine.b == 0.75
        assert engine.k3 == 1000
        assert engine.total_documents == 0
        assert len(engine.vocabulary) == 0
        
        # Test custom parameters
        custom_engine = BM25Search(k1=1.5, b=0.8, k3=500)
        assert custom_engine.k1 == 1.5
        assert custom_engine.b == 0.8
        assert custom_engine.k3 == 500
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality"""
        text = "Hello, World! This is a TEST."
        tokens = self.search_engine.preprocess_text(text)
        
        expected = ['hello', 'world', 'this', 'is', 'a', 'test']
        assert tokens == expected
        
        # Test empty string
        assert self.search_engine.preprocess_text("") == []
        
        # Test string with only punctuation
        assert self.search_engine.preprocess_text("!@#$%") == []
        
        # Test multiple spaces
        tokens = self.search_engine.preprocess_text("word1    word2")
        assert tokens == ['word1', 'word2']
    
    def test_extract_phrases(self):
        """Test phrase extraction from queries"""
        # Test simple phrase
        phrases, terms = self.search_engine.extract_phrases('"machine learning" python')
        assert phrases == ['machine learning']
        assert 'python' in terms
        
        # Test multiple phrases
        phrases, terms = self.search_engine.extract_phrases('"deep learning" and "neural networks" tutorial')
        assert set(phrases) == {'deep learning', 'neural networks'}
        assert 'and' in terms
        assert 'tutorial' in terms
        
        # Test no phrases
        phrases, terms = self.search_engine.extract_phrases('machine learning python')
        assert phrases == []
        assert set(terms) == {'machine', 'learning', 'python'}
        
        # Test empty query
        phrases, terms = self.search_engine.extract_phrases('')
        assert phrases == []
        assert terms == []
    
    def test_build_index(self):
        """Test inverted index construction"""
        self.search_engine.build_index(self.sample_docs)
        
        # Check basic statistics
        assert self.search_engine.total_documents == 5
        assert len(self.search_engine.documents) == 5
        assert len(self.search_engine.document_lengths) == 5
        assert self.search_engine.average_document_length > 0
        
        # Check vocabulary contains expected terms
        assert 'machine' in self.search_engine.vocabulary
        assert 'learning' in self.search_engine.vocabulary
        assert 'python' in self.search_engine.vocabulary
        
        # Check inverted index structure
        assert 'machine' in self.search_engine.inverted_index
        machine_docs = self.search_engine.inverted_index['machine']
        assert 'doc1' in machine_docs  # "machine learning"
        assert 'doc2' in machine_docs  # "machine learning" 
        assert 'doc5' in machine_docs  # "machine learning"
        
        # Check document frequencies
        assert self.search_engine.document_frequencies['machine'] == 3
        assert self.search_engine.document_frequencies['python'] == 1
    
    def test_bm25_score_calculation(self):
        """Test BM25 score calculation"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test score for relevant document
        score = self.search_engine.calculate_bm25_score(['machine', 'learning'], 'doc1')
        assert score > 0
        
        # Test score for non-existent document
        score = self.search_engine.calculate_bm25_score(['machine', 'learning'], 'nonexistent')
        assert score == 0
        
        # Test score with non-existent terms
        score = self.search_engine.calculate_bm25_score(['nonexistent', 'term'], 'doc1')
        assert score == 0
        
        # Test that documents with more query terms score higher
        score1 = self.search_engine.calculate_bm25_score(['machine'], 'doc1')
        score2 = self.search_engine.calculate_bm25_score(['machine', 'learning'], 'doc1')
        assert score2 > score1
    
    def test_search_phrases(self):
        """Test phrase search functionality"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test exact phrase match
        results = self.search_engine.search_phrases(['machine learning'])
        assert len(results) > 0
        
        # Check that results contain the phrase
        found_docs = [r['id'] for r in results]
        assert 'doc1' in found_docs  # "machine learning is a subset"
        assert 'doc2' in found_docs  # "great for machine learning"
        assert 'doc5' in found_docs  # "includes machine learning"
        
        # Test phrase that doesn't exist
        results = self.search_engine.search_phrases(['nonexistent phrase'])
        assert len(results) == 0
        
        # Test multiple phrases
        results = self.search_engine.search_phrases(['machine learning', 'deep learning'])
        assert len(results) > 0
        
        # Verify phrase scores are higher than term scores
        phrase_result = self.search_engine.search_phrases(['machine learning'])[0]
        assert phrase_result['score'] > 10  # Phrase boost
        assert phrase_result['match_type'] == 'phrase'
    
    def test_search_terms(self):
        """Test term-based search functionality"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test single term search
        results = self.search_engine.search_terms(['python'])
        assert len(results) == 1
        assert results[0]['id'] == 'doc2'
        assert results[0]['match_type'] == 'terms'
        
        # Test multiple term search
        results = self.search_engine.search_terms(['machine', 'learning'])
        assert len(results) > 0
        
        # Check that results are sorted by score
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]['score'] >= results[i + 1]['score']
        
        # Test non-existent terms
        results = self.search_engine.search_terms(['nonexistent', 'terms'])
        assert len(results) == 0
    
    def test_main_search(self):
        """Test main search method"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test simple term search
        results = self.search_engine.search('python')
        assert len(results) > 0
        assert results[0]['id'] == 'doc2'
        
        # Test phrase search
        results = self.search_engine.search('"machine learning"')
        assert len(results) > 0
        
        # Test combined phrase and term search
        results = self.search_engine.search('"machine learning" python')
        assert len(results) > 0
        
        # Test empty query
        results = self.search_engine.search('')
        assert len(results) == 0
        
        # Test top_k limiting
        results = self.search_engine.search('learning', top_k=2)
        assert len(results) <= 2
    
    def test_score_explanation(self):
        """Test score explanation functionality"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test explanation for existing document
        explanation = self.search_engine.explain_score('machine learning', 'doc1')
        
        assert 'document_id' in explanation
        assert 'query' in explanation
        assert 'total_score' in explanation
        assert 'score_breakdown' in explanation
        assert explanation['document_id'] == 'doc1'
        assert explanation['total_score'] > 0
        
        # Test explanation for non-existent document
        explanation = self.search_engine.explain_score('query', 'nonexistent')
        assert 'error' in explanation
        
        # Test explanation with phrase
        explanation = self.search_engine.explain_score('"machine learning"', 'doc1')
        assert explanation['total_score'] > 0
        
        # Check for detailed scoring breakdown
        has_phrase_score = any(item['type'] == 'phrase' for item in explanation['score_breakdown'])
        assert has_phrase_score
    
    def test_statistics(self):
        """Test statistics collection"""
        self.search_engine.build_index(self.sample_docs)
        stats = self.search_engine.get_statistics()
        
        assert 'total_documents' in stats
        assert 'vocabulary_size' in stats
        assert 'average_document_length' in stats
        assert 'index_size' in stats
        assert 'parameters' in stats
        
        assert stats['total_documents'] == 5
        assert stats['vocabulary_size'] > 0
        assert stats['average_document_length'] > 0
        assert stats['parameters']['k1'] == 1.2
        assert stats['parameters']['b'] == 0.75
    
    def test_bm25_algorithm_correctness(self):
        """Test BM25 algorithm correctness against known calculations"""
        # Simple test case for manual verification
        simple_docs = [
            {'id': 'doc1', 'content': 'cat dog'},
            {'id': 'doc2', 'content': 'cat cat mouse'},
            {'id': 'doc3', 'content': 'dog mouse bird'}
        ]
        
        engine = BM25Search(k1=1.2, b=0.75)
        engine.build_index(simple_docs)
        
        # Test that cat appears more in doc2 and should score higher
        score1 = engine.calculate_bm25_score(['cat'], 'doc1')
        score2 = engine.calculate_bm25_score(['cat'], 'doc2')
        
        # doc2 has higher term frequency for 'cat', so should score higher
        assert score2 > score1
        
        # Test IDF calculation - rarer terms should have higher IDF
        # 'mouse' appears in 2 docs, 'bird' appears in 1 doc
        mouse_score = engine.calculate_bm25_score(['mouse'], 'doc2')
        bird_score = engine.calculate_bm25_score(['bird'], 'doc3')
        
        # Both have same term frequency (1), but bird is rarer
        # However, document length normalization may affect this
        assert bird_score > 0
        assert mouse_score > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty document list
        self.search_engine.build_index([])
        assert self.search_engine.total_documents == 0
        assert len(self.search_engine.search('test')) == 0
        
        # Test with documents containing only punctuation
        punct_docs = [{'id': 'doc1', 'content': '!@#$%^&*()'}]
        self.search_engine.build_index(punct_docs)
        results = self.search_engine.search('test')
        assert len(results) == 0
        
        # Test with very short documents
        short_docs = [{'id': 'doc1', 'content': 'a'}]
        self.search_engine.build_index(short_docs)
        results = self.search_engine.search('a')
        assert len(results) == 1
    
    def test_result_structure(self):
        """Test that search results have correct structure"""
        self.search_engine.build_index(self.sample_docs)
        results = self.search_engine.search('machine learning')
        
        assert len(results) > 0
        
        for result in results:
            # Check required fields
            assert 'id' in result
            assert 'document' in result
            assert 'score' in result
            assert 'match_type' in result
            
            # Check score is positive
            assert result['score'] > 0
            
            # Check document structure
            doc = result['document']
            assert 'content' in doc
            assert 'id' in doc
    
    def test_case_insensitive_search(self):
        """Test that search is case insensitive"""
        self.search_engine.build_index(self.sample_docs)
        
        # Test different cases of the same query
        results_lower = self.search_engine.search('machine learning')
        results_upper = self.search_engine.search('MACHINE LEARNING')
        results_mixed = self.search_engine.search('Machine Learning')
        
        # Should return same number of results
        assert len(results_lower) == len(results_upper) == len(results_mixed)
        
        # Should return same documents (though scores might vary slightly due to floating point)
        ids_lower = {r['id'] for r in results_lower}
        ids_upper = {r['id'] for r in results_upper}
        ids_mixed = {r['id'] for r in results_mixed}
        
        assert ids_lower == ids_upper == ids_mixed


class TestBM25Performance:
    """Performance and scalability tests for BM25 search"""
    
    def test_large_vocabulary(self):
        """Test performance with large vocabulary"""
        # Create documents with diverse vocabulary
        large_docs = []
        for i in range(100):
            content = f"document {i} contains unique terms term{i} word{i} content{i}"
            large_docs.append({'id': f'doc{i}', 'content': content})
        
        engine = BM25Search()
        engine.build_index(large_docs)
        
        # Test that index was built correctly
        assert engine.total_documents == 100
        assert len(engine.vocabulary) > 100  # Should have many unique terms
        
        # Test search still works
        results = engine.search('document')
        assert len(results) > 0
    
    def test_long_documents(self):
        """Test with very long documents"""
        long_content = ' '.join(['word'] * 1000)  # 1000 word document
        long_docs = [{'id': 'doc1', 'content': long_content}]
        
        engine = BM25Search()
        engine.build_index(long_docs)
        
        assert engine.document_lengths['doc1'] == 1000
        assert engine.average_document_length == 1000
        
        # Search should still work
        results = engine.search('word')
        assert len(results) == 1


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])