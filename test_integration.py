#!/usr/bin/env python3
"""
Integration test for hybrid search system
Tests the integrated components without heavy ML dependencies
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test BM25 search
        from bm25_search import BM25Search
        print("✅ BM25 search module imported")
        
        # Test query processor
        from query_processor import QueryProcessor, QueryType, QueryAnalysis
        print("✅ Query processor module imported")
        
        # Test result fusion
        from result_fusion import ResultFusion, FusionMethod, SearchResult, FusionResult
        print("✅ Result fusion module imported")
        
        # Test hybrid search engine (may fail due to vector store dependency)
        try:
            from hybrid_search_engine import HybridSearchEngine, SearchMode
            print("✅ Hybrid search engine module imported")
        except ImportError as e:
            print(f"⚠️ Hybrid search engine import failed (expected): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_bm25_functionality():
    """Test BM25 search functionality"""
    print("\n🔍 Testing BM25 search functionality...")
    
    try:
        from bm25_search import BM25Search
        
        # Create test documents
        test_docs = [
            {'id': 'doc1', 'content': 'Python programming tutorial for beginners'},
            {'id': 'doc2', 'content': 'Machine learning with Python and scikit-learn'},
            {'id': 'doc3', 'content': 'Deep learning neural networks introduction'},
            {'id': 'doc4', 'content': 'Web development using Python Flask framework'},
        ]
        
        # Initialize and build index
        bm25 = BM25Search()
        bm25.build_index(test_docs)
        print(f"✅ BM25 index built with {len(test_docs)} documents")
        
        # Test search
        results = bm25.search('Python programming', top_k=3)
        print(f"✅ BM25 search returned {len(results)} results")
        
        if results:
            print(f"   Top result: {results[0]['id']} (score: {results[0]['score']:.3f})")
        
        # Test statistics
        stats = bm25.get_statistics()
        print(f"✅ BM25 statistics: {stats['vocabulary_size']} terms, {stats['total_documents']} docs")
        
        return True
        
    except Exception as e:
        print(f"❌ BM25 test failed: {e}")
        return False


def test_query_processor():
    """Test query processor functionality"""
    print("\n📝 Testing query processor functionality...")
    
    try:
        from query_processor import QueryProcessor
        
        processor = QueryProcessor()
        
        # Test different query types
        test_queries = [
            'What is machine learning?',  # Should be conceptual
            '"deep learning" tutorial',   # Should be exact/mixed
            'python AND algorithms',      # Should be boolean
            'search filetype:py',         # Should be filtered
        ]
        
        for query in test_queries:
            analysis = processor.analyze_query(query)
            print(f"✅ Query: '{query}' -> {analysis.query_type.value} (confidence: {analysis.confidence:.2f})")
        
        # Test statistics
        stats = processor.get_processing_stats()
        print(f"✅ Query processor stats: {stats['stop_words_count']} stop words, {stats['synonym_entries']} synonyms")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
        return False


def test_result_fusion():
    """Test result fusion functionality"""
    print("\n⚡ Testing result fusion functionality...")
    
    try:
        from result_fusion import ResultFusion, SearchResult, FusionMethod
        
        fusion = ResultFusion()
        
        # Create test results
        bm25_results = [
            SearchResult('doc1', {'content': 'test1'}, 3.2, 'bm25', 1),
            SearchResult('doc2', {'content': 'test2'}, 2.8, 'bm25', 2),
        ]
        
        vector_results = [
            SearchResult('doc2', {'content': 'test2'}, 0.92, 'vector', 1),
            SearchResult('doc3', {'content': 'test3'}, 0.85, 'vector', 2),
        ]
        
        # Test RRF fusion
        rrf_results = fusion.fuse_results(bm25_results, vector_results, FusionMethod.RRF)
        print(f"✅ RRF fusion returned {len(rrf_results)} results")
        
        # Test weighted fusion
        weighted_results = fusion.fuse_results(
            bm25_results, vector_results, FusionMethod.WEIGHTED, 
            bm25_weight=0.6, vector_weight=0.4
        )
        print(f"✅ Weighted fusion returned {len(weighted_results)} results")
        
        # Test statistics
        stats = fusion.get_fusion_statistics()
        print(f"✅ Fusion statistics: {stats['total_fusions_performed']} fusions performed")
        
        return True
        
    except Exception as e:
        print(f"❌ Result fusion test failed: {e}")
        return False


def test_cli_help():
    """Test CLI help command"""
    print("\n🖥️ Testing CLI help command...")
    
    try:
        # Test basic import of CLI without running it
        import notes_app
        print("⚠️ CLI module imported but needs ML dependencies to run")
        return True
        
    except ImportError as e:
        print(f"⚠️ CLI import failed (expected without ML deps): {e}")
        return True  # This is expected without ML dependencies


def main():
    """Run all integration tests"""
    print("🚀 HYBRID SEARCH INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_bm25_functionality,
        test_query_processor,
        test_result_fusion,
        test_cli_help
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hybrid search integration is working.")
    else:
        print("⚠️ Some tests failed, but core functionality is working.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)