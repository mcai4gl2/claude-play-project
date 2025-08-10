"""
Hybrid Search Engine Orchestrator

This module orchestrates the hybrid search system, combining BM25 traditional search
with vector semantic search using various fusion algorithms.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

from bm25_search import BM25Search
from query_processor import QueryProcessor, QueryAnalysis, QueryType
from result_fusion import ResultFusion, FusionMethod, SearchResult, FusionResult


class SearchMode(Enum):
    """Search mode options"""
    AUTO = "auto"           # Intelligent routing based on query type
    TRADITIONAL = "traditional"  # BM25-only search
    SEMANTIC = "semantic"   # Vector-only search (existing behavior)
    HYBRID = "hybrid"       # Always combine both approaches
    CUSTOM = "custom"       # User-defined weighting


class HybridSearchEngine:
    """
    Main hybrid search orchestrator that combines BM25 and vector search.
    
    This class handles query processing, search routing, result fusion,
    and provides explanations for search results.
    """
    
    def __init__(self, vector_search_engine, notes_dir: Optional[str] = None):
        """
        Initialize the hybrid search engine.
        
        Args:
            vector_search_engine: Existing vector search engine instance
            notes_dir: Directory containing notes for BM25 indexing
        """
        self.vector_engine = vector_search_engine
        self.bm25_engine = BM25Search()
        self.query_processor = QueryProcessor()
        self.result_fusion = ResultFusion()
        
        self.notes_dir = notes_dir
        self.bm25_indexed = False
        
        # Configuration
        self.default_fusion_method = FusionMethod.ADAPTIVE
        self.default_weights = {'bm25': 0.5, 'vector': 0.5}
        
        # Statistics
        self.search_stats = {
            'total_searches': 0,
            'mode_usage': {},
            'average_results': {'bm25': 0, 'vector': 0, 'hybrid': 0}
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def build_bm25_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            True if indexing succeeded
        """
        try:
            self.bm25_engine.build_index(documents)
            self.bm25_indexed = True
            self.logger.info(f"BM25 index built with {len(documents)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
            return False
    
    def search(self, query: str, 
               mode: SearchMode = SearchMode.AUTO,
               top_k: int = 5,
               similarity_threshold: Optional[float] = None,
               bm25_weight: float = 0.5,
               vector_weight: float = 0.5,
               fusion_method: Optional[FusionMethod] = None) -> List[Dict[str, Any]]:
        """
        Main search method with hybrid capabilities.
        
        Args:
            query: Search query string
            mode: Search mode to use
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold for vector search
            bm25_weight: Weight for BM25 results (for custom mode)
            vector_weight: Weight for vector results (for custom mode)
            fusion_method: Specific fusion method to use
            
        Returns:
            List of search results with scores and metadata
        """
        if not query or not query.strip():
            return []
        
        # Update statistics
        self.search_stats['total_searches'] += 1
        mode_name = mode.value
        self.search_stats['mode_usage'][mode_name] = (
            self.search_stats['mode_usage'].get(mode_name, 0) + 1
        )
        
        # Process query
        query_analysis = self.query_processor.analyze_query(query)
        
        # Route search based on mode
        if mode == SearchMode.AUTO:
            # Use suggested mode from query analysis
            suggested_mode = query_analysis.suggested_mode
            if suggested_mode == 'traditional':
                return self._traditional_search(query, top_k, query_analysis)
            elif suggested_mode == 'semantic':
                return self._semantic_search(query, top_k, similarity_threshold, query_analysis)
            else:  # hybrid or auto
                return self._hybrid_search(query, top_k, similarity_threshold, 
                                         query_analysis, fusion_method)
        
        elif mode == SearchMode.TRADITIONAL:
            return self._traditional_search(query, top_k, query_analysis)
        
        elif mode == SearchMode.SEMANTIC:
            return self._semantic_search(query, top_k, similarity_threshold, query_analysis)
        
        elif mode == SearchMode.HYBRID:
            return self._hybrid_search(query, top_k, similarity_threshold, 
                                     query_analysis, fusion_method)
        
        elif mode == SearchMode.CUSTOM:
            return self._custom_search(query, top_k, similarity_threshold,
                                     bm25_weight, vector_weight, query_analysis, fusion_method)
        
        else:
            # Default to auto mode
            return self.search(query, SearchMode.AUTO, top_k, similarity_threshold)
    
    def _traditional_search(self, query: str, top_k: int, 
                          query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Perform BM25-only search"""
        if not self.bm25_indexed:
            self.logger.warning("BM25 index not available, falling back to vector search")
            return self._semantic_search(query, top_k, None, query_analysis)
        
        try:
            bm25_results = self.bm25_engine.search(query, top_k=top_k)
            
            # Convert to standard format
            results = []
            for result in bm25_results:
                results.append({
                    'id': result['id'],
                    'document': result['document'],
                    'score': result['score'],
                    'relevance_score': result['score'],  # For compatibility
                    'search_mode': 'traditional',
                    'match_type': result.get('match_type', 'terms'),
                    'matched_terms': result.get('matched_terms', []),
                    'matched_phrases': result.get('matched_phrases', []),
                    'query_analysis': query_analysis.metadata
                })
            
            self._update_result_stats('bm25', len(results))
            return results
            
        except Exception as e:
            self.logger.error(f"Traditional search failed: {e}")
            return self._semantic_search(query, top_k, None, query_analysis)
    
    def _semantic_search(self, query: str, top_k: int, 
                        similarity_threshold: Optional[float],
                        query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Perform vector-only search using existing vector engine"""
        try:
            # Use existing vector search method
            if hasattr(self.vector_engine, 'search'):
                # FAISS-based vector store
                results = self.vector_engine.search(
                    query, top_k=top_k, similarity_threshold=similarity_threshold
                )
            else:
                # Fallback to query engine search
                results = []
            
            # Add hybrid search metadata
            for result in results:
                result['search_mode'] = 'semantic'
                result['query_analysis'] = query_analysis.metadata
            
            self._update_result_stats('vector', len(results))
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int,
                      similarity_threshold: Optional[float],
                      query_analysis: QueryAnalysis,
                      fusion_method: Optional[FusionMethod] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining both engines"""
        
        # Get results from both engines
        bm25_results = []
        vector_results = []
        
        # BM25 search
        if self.bm25_indexed:
            try:
                bm25_raw = self.bm25_engine.search(query, top_k=top_k * 2)  # Get more for fusion
                bm25_results = [
                    SearchResult(
                        id=r['id'],
                        document=r['document'],
                        score=max(0.0, r['score']),  # Ensure non-negative scores
                        source='bm25',
                        rank=idx + 1,
                        metadata={'match_type': r.get('match_type', 'terms')}
                    )
                    for idx, r in enumerate(bm25_raw)
                    if r['score'] > 0  # Filter out negative scores
                ]
            except Exception as e:
                self.logger.warning(f"BM25 search failed in hybrid mode: {e}")
        
        # Vector search
        try:
            vector_raw = self._semantic_search(query, top_k * 2, similarity_threshold, query_analysis)
            vector_results = [
                SearchResult(
                    id=r.get('id', r.get('document', {}).get('id', str(idx))),
                    document=r.get('document', r),
                    score=r.get('relevance_score', r.get('score', 0.0)),
                    source='vector',
                    rank=idx + 1,
                    metadata={'similarity': r.get('relevance_score', r.get('score', 0.0))}
                )
                for idx, r in enumerate(vector_raw)
            ]
        except Exception as e:
            self.logger.warning(f"Vector search failed in hybrid mode: {e}")
        
        # Fuse results
        if not bm25_results and not vector_results:
            return []
        
        # Determine fusion method
        if fusion_method is None:
            fusion_method = self.default_fusion_method
        
        # Apply fusion
        if fusion_method == FusionMethod.ADAPTIVE:
            fused_results = self.result_fusion.fuse_results(
                bm25_results, vector_results, fusion_method,
                query_type=query_analysis.query_type.value,
                query_confidence=query_analysis.confidence
            )
        else:
            fused_results = self.result_fusion.fuse_results(
                bm25_results, vector_results, fusion_method
            )
        
        # Convert to standard format
        results = []
        for result in fused_results[:top_k]:
            results.append({
                'id': result.id,
                'document': result.document,
                'score': result.final_score,
                'relevance_score': result.final_score,
                'search_mode': 'hybrid',
                'fusion_method': result.fusion_method,
                'source_ranks': result.source_ranks,
                'source_scores': result.source_scores,
                'score_breakdown': result.score_breakdown,
                'query_analysis': query_analysis.metadata
            })
        
        self._update_result_stats('hybrid', len(results))
        return results
    
    def _custom_search(self, query: str, top_k: int,
                      similarity_threshold: Optional[float],
                      bm25_weight: float, vector_weight: float,
                      query_analysis: QueryAnalysis,
                      fusion_method: Optional[FusionMethod] = None) -> List[Dict[str, Any]]:
        """Perform custom weighted search"""
        
        # Use hybrid search with custom weights
        if fusion_method is None:
            fusion_method = FusionMethod.WEIGHTED
        
        # Get results from both engines (similar to hybrid search)
        bm25_results = []
        vector_results = []
        
        # BM25 search
        if self.bm25_indexed:
            try:
                bm25_raw = self.bm25_engine.search(query, top_k=top_k * 2)
                bm25_results = [
                    SearchResult(
                        id=r['id'], document=r['document'], score=max(0.0, r['score']),
                        source='bm25', rank=idx + 1,
                        metadata={'match_type': r.get('match_type', 'terms')}
                    )
                    for idx, r in enumerate(bm25_raw) if r['score'] > 0
                ]
            except Exception as e:
                self.logger.warning(f"BM25 search failed in custom mode: {e}")
        
        # Vector search
        try:
            vector_raw = self._semantic_search(query, top_k * 2, similarity_threshold, query_analysis)
            vector_results = [
                SearchResult(
                    id=r.get('id', r.get('document', {}).get('id', str(idx))),
                    document=r.get('document', r), score=r.get('relevance_score', r.get('score', 0.0)),
                    source='vector', rank=idx + 1,
                    metadata={'similarity': r.get('relevance_score', r.get('score', 0.0))}
                )
                for idx, r in enumerate(vector_raw)
            ]
        except Exception as e:
            self.logger.warning(f"Vector search failed in custom mode: {e}")
        
        # Apply weighted fusion
        if fusion_method == FusionMethod.WEIGHTED:
            fused_results = self.result_fusion.fuse_results(
                bm25_results, vector_results, fusion_method,
                bm25_weight=bm25_weight, vector_weight=vector_weight
            )
        else:
            fused_results = self.result_fusion.fuse_results(
                bm25_results, vector_results, fusion_method
            )
        
        # Convert to standard format
        results = []
        for result in fused_results[:top_k]:
            results.append({
                'id': result.id,
                'document': result.document,
                'score': result.final_score,
                'relevance_score': result.final_score,
                'search_mode': 'custom',
                'fusion_method': result.fusion_method,
                'bm25_weight': bm25_weight,
                'vector_weight': vector_weight,
                'source_ranks': result.source_ranks,
                'source_scores': result.source_scores,
                'query_analysis': query_analysis.metadata
            })
        
        return results
    
    def _update_result_stats(self, engine_type: str, count: int):
        """Update search result statistics"""
        current_avg = self.search_stats['average_results'][engine_type]
        total_searches = self.search_stats['total_searches']
        
        # Update running average
        new_avg = (current_avg * (total_searches - 1) + count) / total_searches
        self.search_stats['average_results'][engine_type] = new_avg
    
    def explain_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide detailed explanation of search results.
        
        Args:
            query: Original search query
            results: Search results to explain
            
        Returns:
            Detailed explanation dictionary
        """
        if not results:
            return {'explanation': 'No results to explain', 'query': query}
        
        query_analysis = self.query_processor.analyze_query(query)
        
        explanation = {
            'query': query,
            'query_analysis': {
                'type': query_analysis.query_type.value,
                'confidence': query_analysis.confidence,
                'suggested_mode': query_analysis.suggested_mode,
                'phrases': query_analysis.phrases,
                'terms': query_analysis.terms,
                'filters': query_analysis.filters,
                'metadata': query_analysis.metadata
            },
            'search_mode': results[0].get('search_mode', 'unknown'),
            'total_results': len(results),
            'result_explanations': []
        }
        
        # Explain individual results
        for idx, result in enumerate(results[:3]):  # Explain top 3 results
            result_explanation = {
                'rank': idx + 1,
                'document_id': result.get('id'),
                'final_score': result.get('score', result.get('relevance_score', 0.0)),
                'search_mode': result.get('search_mode')
            }
            
            # Add mode-specific explanations
            if result.get('search_mode') == 'hybrid':
                result_explanation.update({
                    'fusion_method': result.get('fusion_method'),
                    'source_ranks': result.get('source_ranks', {}),
                    'source_scores': result.get('source_scores', {}),
                    'score_breakdown': result.get('score_breakdown', {})
                })
                
            elif result.get('search_mode') == 'traditional':
                result_explanation.update({
                    'match_type': result.get('match_type'),
                    'matched_terms': result.get('matched_terms', []),
                    'matched_phrases': result.get('matched_phrases', [])
                })
                
            elif result.get('search_mode') == 'semantic':
                result_explanation['similarity_score'] = result.get('relevance_score', result.get('score'))
            
            explanation['result_explanations'].append(result_explanation)
        
        return explanation
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        return {
            'search_statistics': self.search_stats.copy(),
            'fusion_statistics': self.result_fusion.get_fusion_statistics(),
            'query_processor_stats': self.query_processor.get_processing_stats(),
            'bm25_stats': self.bm25_engine.get_statistics() if self.bm25_indexed else {},
            'configuration': {
                'default_fusion_method': self.default_fusion_method.value,
                'default_weights': self.default_weights.copy(),
                'bm25_indexed': self.bm25_indexed
            }
        }
    
    def configure_defaults(self, fusion_method: Optional[FusionMethod] = None,
                          bm25_weight: Optional[float] = None,
                          vector_weight: Optional[float] = None):
        """Configure default search parameters"""
        if fusion_method:
            self.default_fusion_method = fusion_method
        
        if bm25_weight is not None:
            self.default_weights['bm25'] = bm25_weight
            
        if vector_weight is not None:
            self.default_weights['vector'] = vector_weight