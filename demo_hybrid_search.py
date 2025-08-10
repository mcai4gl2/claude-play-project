#!/usr/bin/env python3
"""
Hybrid Search Demo

Demonstrates the hybrid search functionality without requiring heavy ML dependencies.
This shows the BM25, query processing, and result fusion components working together.
"""

from bm25_search import BM25Search
from query_processor import QueryProcessor, QueryType
from result_fusion import ResultFusion, SearchResult, FusionMethod


def create_sample_documents():
    """Create sample documents for testing"""
    return [
        {
            'id': 'note1',
            'content': 'Python programming tutorial for beginners. Learn basic syntax, variables, and functions.',
            'title': 'Python Basics',
            'filepath': 'notes/python_basics.md'
        },
        {
            'id': 'note2', 
            'content': 'Machine learning fundamentals using Python. Introduction to scikit-learn, pandas, and numpy.',
            'title': 'ML with Python',
            'filepath': 'notes/machine_learning.md'
        },
        {
            'id': 'note3',
            'content': 'Deep learning neural networks. Understanding backpropagation and gradient descent algorithms.',
            'title': 'Deep Learning Guide',
            'filepath': 'notes/deep_learning.md'
        },
        {
            'id': 'note4',
            'content': 'Web development with Python Flask framework. Building REST APIs and handling HTTP requests.',
            'title': 'Flask Web Development',
            'filepath': 'notes/flask_web.md'
        },
        {
            'id': 'note5',
            'content': 'Data science workflow from data collection to model deployment. Best practices and tools.',
            'title': 'Data Science Workflow',
            'filepath': 'notes/data_science.md'
        }
    ]


def simulate_vector_search(query, documents, top_k=5):
    """Simulate vector search results (simplified semantic matching)"""
    results = []
    
    # Simple keyword-based similarity simulation
    query_lower = query.lower()
    
    for doc in documents:
        content_lower = doc['content'].lower()
        title_lower = doc['title'].lower()
        
        # Calculate simple similarity score
        content_matches = sum(1 for word in query_lower.split() if word in content_lower)
        title_matches = sum(1 for word in query_lower.split() if word in title_lower) * 2  # Title boost
        
        similarity = (content_matches + title_matches) / len(query_lower.split())
        
        if similarity > 0:
            results.append(SearchResult(
                id=doc['id'],
                document=doc,
                score=similarity,
                source='vector',
                rank=len(results) + 1,
                metadata={'similarity': similarity}
            ))
    
    # Sort by similarity score
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


def demo_search_modes():
    """Demonstrate different search modes"""
    print("🔍 HYBRID SEARCH DEMO")
    print("=" * 60)
    
    # Initialize components
    documents = create_sample_documents()
    bm25_engine = BM25Search(k1=1.5, b=0.75)  # Adjusted parameters
    query_processor = QueryProcessor()
    result_fusion = ResultFusion()
    
    # Build BM25 index
    print("📚 Building BM25 index...")
    bm25_engine.build_index(documents)
    stats = bm25_engine.get_statistics()
    print(f"   Indexed {stats['total_documents']} documents with {stats['vocabulary_size']} unique terms\n")
    
    # Test queries
    test_queries = [
        "Python programming tutorial",
        '"machine learning" basics',
        "deep learning AND neural networks",
        "web development framework"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 40)
        
        # 1. Query Analysis
        analysis = query_processor.analyze_query(query)
        print(f"📊 Query Analysis:")
        print(f"   Type: {analysis.query_type.value} (confidence: {analysis.confidence:.2f})")
        print(f"   Suggested mode: {analysis.suggested_mode}")
        if analysis.phrases:
            print(f"   Phrases: {', '.join(analysis.phrases)}")
        if analysis.terms:
            print(f"   Terms: {', '.join(analysis.terms)}")
        print()
        
        # 2. BM25 Search
        bm25_results = bm25_engine.search(query, top_k=3)
        print(f"📝 BM25 Results ({len(bm25_results)} found):")
        for i, result in enumerate(bm25_results, 1):
            print(f"   {i}. {result['document']['title']} (score: {result['score']:.3f})")
        print()
        
        # 3. Simulated Vector Search  
        vector_results = simulate_vector_search(query, documents, top_k=3)
        print(f"🧠 Vector Results ({len(vector_results)} found):")
        for i, result in enumerate(vector_results, 1):
            print(f"   {i}. {result.document['title']} (similarity: {result.score:.3f})")
        print()
        
        # 4. Fusion Results
        if bm25_results and vector_results:
            # Convert BM25 results to SearchResult format
            bm25_search_results = [
                SearchResult(
                    id=r['id'],
                    document=r['document'], 
                    score=max(0.1, r['score']),  # Ensure positive scores
                    source='bm25',
                    rank=i+1,
                    metadata={'match_type': r.get('match_type', 'terms')}
                )
                for i, r in enumerate(bm25_results)
                if r['score'] > 0  # Only include positive scores
            ]
            
            # Try different fusion methods
            fusion_methods = [
                (FusionMethod.RRF, "Reciprocal Rank Fusion"),
                (FusionMethod.WEIGHTED, "Weighted Fusion"),
                (FusionMethod.ADAPTIVE, "Adaptive Fusion")
            ]
            
            for method, name in fusion_methods:
                if method == FusionMethod.ADAPTIVE:
                    fused_results = result_fusion.fuse_results(
                        bm25_search_results, vector_results, method,
                        query_type=analysis.query_type.value,
                        query_confidence=analysis.confidence
                    )
                else:
                    fused_results = result_fusion.fuse_results(
                        bm25_search_results, vector_results, method
                    )
                
                print(f"⚡ {name} Results ({len(fused_results)} found):")
                for i, result in enumerate(fused_results[:3], 1):
                    sources = ", ".join(result.metadata.get('sources', []))
                    print(f"   {i}. {result.document['title']} (score: {result.final_score:.3f}, sources: {sources})")
                print()
        
        print("=" * 60)
        print()


def demo_query_suggestions():
    """Demonstrate query suggestion functionality"""
    print("\n💡 QUERY SUGGESTIONS DEMO")
    print("=" * 40)
    
    query_processor = QueryProcessor()
    
    test_queries = [
        "ml tutorial",
        "how to implement neural networks",
        "api error handling"
    ]
    
    for query in test_queries:
        suggestions = query_processor.get_query_suggestions(query, max_suggestions=3)
        print(f"Query: '{query}'")
        if suggestions:
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        else:
            print("  No suggestions generated")
        print()


def main():
    """Run the hybrid search demo"""
    try:
        demo_search_modes()
        demo_query_suggestions()
        
        print("✅ Demo completed successfully!")
        print("\nTo use the full system with vector embeddings:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Index your notes: python notes_app.py index")
        print("3. Search with hybrid modes: python notes_app.py query 'your query' --mode hybrid --explain")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()