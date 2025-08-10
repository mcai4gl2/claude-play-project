"""
Result Fusion Algorithms for Hybrid Search

This module provides algorithms for combining results from different search engines
(BM25 traditional search and vector semantic search) into unified, ranked results.
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class FusionMethod(Enum):
    """Enumeration of different result fusion methods"""
    RRF = "reciprocal_rank_fusion"      # Reciprocal Rank Fusion
    WEIGHTED = "weighted_score"         # Weighted score combination
    ADAPTIVE = "adaptive"               # Query-type adaptive weighting
    BORDA = "borda_count"              # Borda count ranking
    COMBSUM = "combsum"                # Sum of normalized scores
    COMBANMZ = "combanmz"              # Sum of normalized scores with normalization


@dataclass
class FusionResult:
    """Result from fusion process"""
    id: str
    document: Dict[str, Any]
    final_score: float
    fusion_method: str
    score_breakdown: Dict[str, float]
    source_ranks: Dict[str, int]
    source_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Standardized search result format for fusion input"""
    id: str
    document: Dict[str, Any]
    score: float
    source: str  # 'bm25' or 'vector'
    rank: int
    metadata: Dict[str, Any] = None


class ResultFusion:
    """
    Result fusion engine that combines search results from multiple sources.
    
    Implements various fusion algorithms to merge and rank results from
    BM25 traditional search and vector semantic search engines.
    """
    
    def __init__(self):
        """Initialize the result fusion engine"""
        self.fusion_stats = {
            'total_fusions': 0,
            'method_usage': {},
            'average_result_counts': {
                'bm25': 0,
                'vector': 0,
                'combined': 0
            }
        }
    
    def normalize_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Normalize search result scores to [0, 1] range.
        
        Args:
            results: List of search results to normalize
            
        Returns:
            List of results with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case where all scores are the same
        if max_score == min_score:
            normalized_results = []
            for result in results:
                normalized_result = SearchResult(
                    id=result.id,
                    document=result.document,
                    score=1.0,  # All get same normalized score
                    source=result.source,
                    rank=result.rank,
                    metadata=result.metadata
                )
                normalized_results.append(normalized_result)
            return normalized_results
        
        # Min-max normalization
        normalized_results = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized_result = SearchResult(
                id=result.id,
                document=result.document,
                score=normalized_score,
                source=result.source,
                rank=result.rank,
                metadata=result.metadata
            )
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    def reciprocal_rank_fusion(self, result_lists: List[List[SearchResult]], 
                              k: float = 60.0) -> List[FusionResult]:
        """
        Implement Reciprocal Rank Fusion (RRF) algorithm.
        
        RRF combines rankings by computing RRF score = sum(1/(k + rank)) for each document.
        
        Args:
            result_lists: List of ranked result lists from different search engines
            k: RRF parameter (typically 60)
            
        Returns:
            List of fused results ranked by RRF score
        """
        if not result_lists:
            return []
        
        # Collect all unique documents
        all_docs = {}
        
        for source_idx, results in enumerate(result_lists):
            source_name = results[0].source if results else f"source_{source_idx}"
            
            for rank, result in enumerate(results, 1):  # Rank starts at 1
                doc_id = result.id
                
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'document': result.document,
                        'rrf_score': 0.0,
                        'source_ranks': {},
                        'source_scores': {},
                        'sources': set()
                    }
                
                # Calculate RRF contribution
                rrf_contribution = 1.0 / (k + rank)
                all_docs[doc_id]['rrf_score'] += rrf_contribution
                all_docs[doc_id]['source_ranks'][source_name] = rank
                all_docs[doc_id]['source_scores'][source_name] = result.score
                all_docs[doc_id]['sources'].add(source_name)
        
        # Create fusion results
        fusion_results = []
        for doc_id, doc_data in all_docs.items():
            fusion_result = FusionResult(
                id=doc_id,
                document=doc_data['document'],
                final_score=doc_data['rrf_score'],
                fusion_method='reciprocal_rank_fusion',
                score_breakdown={'rrf_score': doc_data['rrf_score']},
                source_ranks=doc_data['source_ranks'],
                source_scores=doc_data['source_scores'],
                metadata={
                    'source_count': len(doc_data['sources']),
                    'sources': list(doc_data['sources']),
                    'k_parameter': k
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by RRF score (descending)
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results
    
    def weighted_score_fusion(self, bm25_results: List[SearchResult], 
                             vector_results: List[SearchResult],
                             bm25_weight: float = 0.5, 
                             vector_weight: float = 0.5) -> List[FusionResult]:
        """
        Combine results using weighted score fusion.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of fused results ranked by weighted combined score
        """
        # Normalize weights
        total_weight = bm25_weight + vector_weight
        if total_weight > 0:
            bm25_weight = bm25_weight / total_weight
            vector_weight = vector_weight / total_weight
        else:
            bm25_weight = vector_weight = 0.5
        
        # Normalize result scores
        normalized_bm25 = self.normalize_results(bm25_results)
        normalized_vector = self.normalize_results(vector_results)
        
        # Collect all unique documents
        all_docs = {}
        
        # Process BM25 results
        for result in normalized_bm25:
            doc_id = result.id
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'document': result.document,
                    'bm25_score': 0.0,
                    'vector_score': 0.0,
                    'bm25_rank': None,
                    'vector_rank': None,
                    'sources': set()
                }
            
            all_docs[doc_id]['bm25_score'] = result.score
            all_docs[doc_id]['bm25_rank'] = result.rank
            all_docs[doc_id]['sources'].add('bm25')
        
        # Process vector results
        for result in normalized_vector:
            doc_id = result.id
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'document': result.document,
                    'bm25_score': 0.0,
                    'vector_score': 0.0,
                    'bm25_rank': None,
                    'vector_rank': None,
                    'sources': set()
                }
            
            all_docs[doc_id]['vector_score'] = result.score
            all_docs[doc_id]['vector_rank'] = result.rank
            all_docs[doc_id]['sources'].add('vector')
        
        # Calculate weighted scores
        fusion_results = []
        for doc_id, doc_data in all_docs.items():
            weighted_score = (bm25_weight * doc_data['bm25_score'] + 
                            vector_weight * doc_data['vector_score'])
            
            source_ranks = {}
            source_scores = {}
            
            if doc_data['bm25_rank'] is not None:
                source_ranks['bm25'] = doc_data['bm25_rank']
                source_scores['bm25'] = doc_data['bm25_score']
            
            if doc_data['vector_rank'] is not None:
                source_ranks['vector'] = doc_data['vector_rank']
                source_scores['vector'] = doc_data['vector_score']
            
            fusion_result = FusionResult(
                id=doc_id,
                document=doc_data['document'],
                final_score=weighted_score,
                fusion_method='weighted_score',
                score_breakdown={
                    'weighted_score': weighted_score,
                    'bm25_contribution': bm25_weight * doc_data['bm25_score'],
                    'vector_contribution': vector_weight * doc_data['vector_score']
                },
                source_ranks=source_ranks,
                source_scores=source_scores,
                metadata={
                    'bm25_weight': bm25_weight,
                    'vector_weight': vector_weight,
                    'source_count': len(doc_data['sources']),
                    'sources': list(doc_data['sources'])
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by weighted score (descending)
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results
    
    def adaptive_fusion(self, bm25_results: List[SearchResult],
                       vector_results: List[SearchResult],
                       query_type: str,
                       query_confidence: float = 0.5) -> List[FusionResult]:
        """
        Adaptive fusion that adjusts weights based on query type and confidence.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            query_type: Type of query ('exact', 'conceptual', 'mixed', etc.)
            query_confidence: Confidence in query type detection (0-1)
            
        Returns:
            List of adaptively fused results
        """
        # Determine adaptive weights based on query type
        if query_type in ['exact', 'boolean', 'filtered']:
            # Favor BM25 for exact/structured queries
            base_bm25_weight = 0.7
            base_vector_weight = 0.3
        elif query_type == 'conceptual':
            # Favor vector search for conceptual queries
            base_bm25_weight = 0.3
            base_vector_weight = 0.7
        else:  # mixed or unknown
            # Balanced weighting
            base_bm25_weight = 0.5
            base_vector_weight = 0.5
        
        # Adjust weights based on confidence
        # High confidence: use suggested weights more strongly
        # Low confidence: move toward balanced (0.5, 0.5)
        confidence_factor = max(0.1, min(1.0, query_confidence))
        
        bm25_weight = base_bm25_weight * confidence_factor + 0.5 * (1 - confidence_factor)
        vector_weight = base_vector_weight * confidence_factor + 0.5 * (1 - confidence_factor)
        
        # Use weighted score fusion with adaptive weights
        fusion_results = self.weighted_score_fusion(
            bm25_results, vector_results, bm25_weight, vector_weight
        )
        
        # Update metadata to reflect adaptive nature
        for result in fusion_results:
            result.fusion_method = 'adaptive'
            result.metadata.update({
                'query_type': query_type,
                'query_confidence': query_confidence,
                'confidence_factor': confidence_factor,
                'base_bm25_weight': base_bm25_weight,
                'base_vector_weight': base_vector_weight,
                'adaptive_reasoning': f"Adaptive weights for {query_type} query"
            })
        
        return fusion_results
    
    def borda_count_fusion(self, result_lists: List[List[SearchResult]]) -> List[FusionResult]:
        """
        Implement Borda count fusion method.
        
        Each document gets points based on its position: highest rank gets most points.
        
        Args:
            result_lists: List of ranked result lists from different search engines
            
        Returns:
            List of fused results ranked by Borda count
        """
        if not result_lists:
            return []
        
        # Collect all unique documents with Borda points
        all_docs = {}
        
        for source_idx, results in enumerate(result_lists):
            source_name = results[0].source if results else f"source_{source_idx}"
            max_points = len(results)
            
            for rank, result in enumerate(results):
                doc_id = result.id
                # Borda points: max_points - rank (0-indexed rank)
                borda_points = max_points - rank
                
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'document': result.document,
                        'borda_score': 0,
                        'source_ranks': {},
                        'source_scores': {},
                        'sources': set()
                    }
                
                all_docs[doc_id]['borda_score'] += borda_points
                all_docs[doc_id]['source_ranks'][source_name] = rank + 1  # 1-indexed for reporting
                all_docs[doc_id]['source_scores'][source_name] = result.score
                all_docs[doc_id]['sources'].add(source_name)
        
        # Create fusion results
        fusion_results = []
        for doc_id, doc_data in all_docs.items():
            fusion_result = FusionResult(
                id=doc_id,
                document=doc_data['document'],
                final_score=float(doc_data['borda_score']),
                fusion_method='borda_count',
                score_breakdown={'borda_score': doc_data['borda_score']},
                source_ranks=doc_data['source_ranks'],
                source_scores=doc_data['source_scores'],
                metadata={
                    'source_count': len(doc_data['sources']),
                    'sources': list(doc_data['sources'])
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by Borda score (descending)
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results
    
    def combsum_fusion(self, bm25_results: List[SearchResult],
                      vector_results: List[SearchResult]) -> List[FusionResult]:
        """
        CombSUM: Sum of normalized scores from all engines.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            
        Returns:
            List of fused results ranked by sum of normalized scores
        """
        # This is essentially weighted fusion with equal weights
        return self.weighted_score_fusion(bm25_results, vector_results, 1.0, 1.0)
    
    def fuse_results(self, bm25_results: List[SearchResult],
                    vector_results: List[SearchResult],
                    method: FusionMethod = FusionMethod.RRF,
                    **kwargs) -> List[FusionResult]:
        """
        Main entry point for result fusion.
        
        Args:
            bm25_results: Results from BM25 search engine
            vector_results: Results from vector search engine
            method: Fusion method to use
            **kwargs: Additional parameters for specific fusion methods
            
        Returns:
            List of fused and ranked results
        """
        if not bm25_results and not vector_results:
            return []
        
        # Update statistics
        self.fusion_stats['total_fusions'] += 1
        method_name = method.value
        self.fusion_stats['method_usage'][method_name] = (
            self.fusion_stats['method_usage'].get(method_name, 0) + 1
        )
        self.fusion_stats['average_result_counts']['bm25'] = (
            (self.fusion_stats['average_result_counts']['bm25'] * (self.fusion_stats['total_fusions'] - 1) + 
             len(bm25_results)) / self.fusion_stats['total_fusions']
        )
        self.fusion_stats['average_result_counts']['vector'] = (
            (self.fusion_stats['average_result_counts']['vector'] * (self.fusion_stats['total_fusions'] - 1) + 
             len(vector_results)) / self.fusion_stats['total_fusions']
        )
        
        # Handle single-source cases
        if not bm25_results:
            # Only vector results available
            fusion_results = []
            for result in vector_results:
                fusion_result = FusionResult(
                    id=result.id,
                    document=result.document,
                    final_score=result.score,
                    fusion_method=f"{method_name}_vector_only",
                    score_breakdown={'vector_score': result.score},
                    source_ranks={'vector': result.rank},
                    source_scores={'vector': result.score},
                    metadata={'source_count': 1, 'sources': ['vector']}
                )
                fusion_results.append(fusion_result)
            return fusion_results
        
        if not vector_results:
            # Only BM25 results available
            fusion_results = []
            for result in bm25_results:
                fusion_result = FusionResult(
                    id=result.id,
                    document=result.document,
                    final_score=result.score,
                    fusion_method=f"{method_name}_bm25_only",
                    score_breakdown={'bm25_score': result.score},
                    source_ranks={'bm25': result.rank},
                    source_scores={'bm25': result.score},
                    metadata={'source_count': 1, 'sources': ['bm25']}
                )
                fusion_results.append(fusion_result)
            return fusion_results
        
        # Apply selected fusion method
        if method == FusionMethod.RRF:
            k = kwargs.get('k', 60.0)
            results = self.reciprocal_rank_fusion([bm25_results, vector_results], k)
            
        elif method == FusionMethod.WEIGHTED:
            bm25_weight = kwargs.get('bm25_weight', 0.5)
            vector_weight = kwargs.get('vector_weight', 0.5)
            results = self.weighted_score_fusion(bm25_results, vector_results, 
                                               bm25_weight, vector_weight)
            
        elif method == FusionMethod.ADAPTIVE:
            query_type = kwargs.get('query_type', 'mixed')
            query_confidence = kwargs.get('query_confidence', 0.5)
            results = self.adaptive_fusion(bm25_results, vector_results, 
                                         query_type, query_confidence)
            
        elif method == FusionMethod.BORDA:
            results = self.borda_count_fusion([bm25_results, vector_results])
            
        elif method == FusionMethod.COMBSUM:
            results = self.combsum_fusion(bm25_results, vector_results)
            
        else:
            # Default to RRF
            results = self.reciprocal_rank_fusion([bm25_results, vector_results])
        
        # Update final statistics
        self.fusion_stats['average_result_counts']['combined'] = (
            (self.fusion_stats['average_result_counts']['combined'] * (self.fusion_stats['total_fusions'] - 1) + 
             len(results)) / self.fusion_stats['total_fusions']
        )
        
        return results
    
    def deduplicate_results(self, results: List[FusionResult], 
                           similarity_threshold: float = 0.95) -> List[FusionResult]:
        """
        Remove duplicate or highly similar results from fusion output.
        
        Args:
            results: List of fusion results
            similarity_threshold: Threshold for considering results as duplicates
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return results
        
        deduplicated = []
        seen_docs = set()
        
        for result in results:
            doc_id = result.id
            
            # Simple deduplication by document ID
            if doc_id not in seen_docs:
                deduplicated.append(result)
                seen_docs.add(doc_id)
        
        return deduplicated
    
    def explain_fusion(self, result: FusionResult) -> Dict[str, Any]:
        """
        Provide detailed explanation of how a result was scored during fusion.
        
        Args:
            result: Fusion result to explain
            
        Returns:
            Dictionary with detailed fusion explanation
        """
        explanation = {
            'document_id': result.id,
            'final_score': result.final_score,
            'fusion_method': result.fusion_method,
            'score_breakdown': result.score_breakdown.copy(),
            'source_information': {
                'ranks': result.source_ranks.copy(),
                'original_scores': result.source_scores.copy(),
                'sources_used': result.metadata.get('sources', []),
                'source_count': result.metadata.get('source_count', 0)
            },
            'fusion_metadata': result.metadata.copy()
        }
        
        # Add method-specific explanations
        if result.fusion_method == 'reciprocal_rank_fusion':
            k = result.metadata.get('k_parameter', 60)
            explanation['method_details'] = {
                'description': 'Reciprocal Rank Fusion combines rankings using 1/(k+rank)',
                'k_parameter': k,
                'rrf_calculation': 'Sum of 1/({} + rank) across all sources'.format(k)
            }
        
        elif result.fusion_method == 'weighted_score':
            bm25_weight = result.metadata.get('bm25_weight', 'unknown')
            vector_weight = result.metadata.get('vector_weight', 'unknown')
            explanation['method_details'] = {
                'description': 'Weighted combination of normalized scores',
                'bm25_weight': bm25_weight,
                'vector_weight': vector_weight,
                'calculation': f'({bm25_weight} * bm25_score) + ({vector_weight} * vector_score)'
            }
        
        elif result.fusion_method == 'adaptive':
            explanation['method_details'] = {
                'description': 'Query-adaptive weighting based on query type and confidence',
                'query_type': result.metadata.get('query_type', 'unknown'),
                'confidence': result.metadata.get('query_confidence', 'unknown'),
                'reasoning': result.metadata.get('adaptive_reasoning', 'unknown')
            }
        
        return explanation
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fusion operations.
        
        Returns:
            Dictionary with fusion statistics
        """
        return {
            'total_fusions_performed': self.fusion_stats['total_fusions'],
            'method_usage_counts': self.fusion_stats['method_usage'].copy(),
            'average_input_sizes': self.fusion_stats['average_result_counts'].copy(),
            'supported_methods': [method.value for method in FusionMethod],
            'version': '1.0'
        }