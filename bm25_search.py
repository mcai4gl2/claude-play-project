"""
BM25 Search Engine Implementation

Provides traditional keyword-based search using the BM25 ranking algorithm.
Includes inverted index construction, exact phrase matching, and boolean queries.
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any
import string


class BM25Search:
    """
    BM25 (Best Matching 25) search engine implementation.
    
    BM25 is a probabilistic ranking function used for information retrieval
    that ranks documents based on query term frequency, document frequency,
    and document length normalization.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, k3: float = 1000):
        """
        Initialize BM25 search engine with tuning parameters.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Document length normalization parameter (0-1, typically 0.75)
            k3: Query term frequency saturation parameter (typically large)
        """
        self.k1 = k1
        self.b = b
        self.k3 = k3
        
        # Index structures
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vocabulary: Set[str] = set()
        
        # Statistics
        self.total_documents = 0
        self.average_document_length = 0.0
        
        # Preprocessing
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their', 'if'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, lowercasing, and removing punctuation.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize on whitespace
        tokens = text.split()
        
        # Remove stop words (optional - can be disabled for technical documents)
        # tokens = [token for token in tokens if token not in self.stop_words]
        
        return [token for token in tokens if token.strip()]
    
    def extract_phrases(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract quoted phrases and individual terms from query.
        
        Args:
            query: Search query string
            
        Returns:
            Tuple of (phrases, individual_terms)
        """
        phrases = []
        remaining_query = query
        
        # Extract quoted phrases
        phrase_pattern = r'"([^"]+)"'
        for match in re.finditer(phrase_pattern, query):
            phrases.append(match.group(1).lower())
            remaining_query = remaining_query.replace(match.group(0), ' ')
        
        # Extract individual terms from remaining query
        individual_terms = self.preprocess_text(remaining_query)
        
        return phrases, individual_terms
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build inverted index from document collection.
        
        Args:
            documents: List of document dictionaries with 'id', 'content', etc.
        """
        self.inverted_index.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()
        self.documents.clear()
        self.vocabulary.clear()
        
        total_length = 0
        
        for doc in documents:
            doc_id = doc.get('id', str(hash(doc['content'])))
            content = doc['content']
            
            # Store full document
            self.documents[doc_id] = doc
            
            # Preprocess content
            tokens = self.preprocess_text(content)
            doc_length = len(tokens)
            self.document_lengths[doc_id] = doc_length
            total_length += doc_length
            
            # Count term frequencies in this document
            term_counts = Counter(tokens)
            
            # Update inverted index
            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.vocabulary.add(term)
        
        # Calculate document frequencies
        for term, doc_dict in self.inverted_index.items():
            self.document_frequencies[term] = len(doc_dict)
        
        # Calculate statistics
        self.total_documents = len(documents)
        self.average_document_length = total_length / self.total_documents if self.total_documents > 0 else 0
    
    def calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            doc_id: Document identifier
            
        Returns:
            BM25 score for the document
        """
        if doc_id not in self.document_lengths:
            return 0.0
        
        score = 0.0
        doc_length = self.document_lengths[doc_id]
        query_term_counts = Counter(query_terms)
        
        for term, query_freq in query_term_counts.items():
            if term not in self.inverted_index:
                continue
                
            # Term frequency in document
            term_freq = self.inverted_index[term].get(doc_id, 0)
            if term_freq == 0:
                continue
            
            # Document frequency
            doc_freq = self.document_frequencies[term]
            
            # IDF calculation
            idf = math.log((self.total_documents - doc_freq + 0.5) / (doc_freq + 0.5))
            
            # Term frequency normalization
            normalized_tf = (term_freq * (self.k1 + 1)) / (
                term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            )
            
            # Query term frequency normalization
            normalized_qtf = (query_freq * (self.k3 + 1)) / (query_freq + self.k3)
            
            # Add to total score
            score += idf * normalized_tf * normalized_qtf
        
        return score
    
    def search_phrases(self, phrases: List[str], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Search for exact phrases in documents.
        
        Args:
            phrases: List of phrases to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if not phrases:
            return []
        
        phrase_results = []
        
        for phrase in phrases:
            phrase_tokens = self.preprocess_text(phrase)
            if not phrase_tokens:
                continue
            
            # Find documents containing all phrase tokens
            candidate_docs = None
            for token in phrase_tokens:
                if token in self.inverted_index:
                    token_docs = set(self.inverted_index[token].keys())
                    if candidate_docs is None:
                        candidate_docs = token_docs
                    else:
                        candidate_docs = candidate_docs.intersection(token_docs)
                else:
                    candidate_docs = set()
                    break
            
            if not candidate_docs:
                continue
            
            # Check for exact phrase matches
            for doc_id in candidate_docs:
                doc_content = self.documents[doc_id]['content'].lower()
                if phrase in doc_content:
                    # Calculate phrase match score (higher for exact phrases)
                    phrase_score = 10.0 * len(phrase_tokens)  # Boost phrase matches
                    phrase_results.append({
                        'id': doc_id,
                        'document': self.documents[doc_id],
                        'score': phrase_score,
                        'match_type': 'phrase',
                        'matched_phrase': phrase
                    })
        
        # Sort by score and return top results
        phrase_results.sort(key=lambda x: x['score'], reverse=True)
        return phrase_results[:top_k]
    
    def search_terms(self, terms: List[str], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Search for individual terms using BM25 scoring.
        
        Args:
            terms: List of terms to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of matching documents with BM25 scores
        """
        if not terms:
            return []
        
        # Find all candidate documents
        candidate_docs = set()
        for term in terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        
        if not candidate_docs:
            return []
        
        # Calculate BM25 scores for all candidates
        scored_results = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(terms, doc_id)
            if score > 0:
                scored_results.append({
                    'id': doc_id,
                    'document': self.documents[doc_id],
                    'score': score,
                    'match_type': 'terms',
                    'matched_terms': [term for term in terms if term in self.inverted_index and doc_id in self.inverted_index[term]]
                })
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Main search method that handles both phrases and terms.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            return []
        
        # Extract phrases and terms
        phrases, terms = self.extract_phrases(query)
        
        # Search for phrases and terms separately
        phrase_results = self.search_phrases(phrases, top_k * 2)
        term_results = self.search_terms(terms, top_k * 2)
        
        # Combine results, avoiding duplicates
        combined_results = {}
        
        # Add phrase results (higher priority)
        for result in phrase_results:
            doc_id = result['id']
            if doc_id not in combined_results:
                combined_results[doc_id] = result
            else:
                # Combine scores if document already found
                combined_results[doc_id]['score'] += result['score']
                if 'matched_phrases' not in combined_results[doc_id]:
                    combined_results[doc_id]['matched_phrases'] = []
                combined_results[doc_id]['matched_phrases'].append(result['matched_phrase'])
        
        # Add term results
        for result in term_results:
            doc_id = result['id']
            if doc_id not in combined_results:
                combined_results[doc_id] = result
            else:
                # Combine scores if document already found
                combined_results[doc_id]['score'] += result['score']
                if 'matched_terms' not in combined_results[doc_id]:
                    combined_results[doc_id]['matched_terms'] = combined_results[doc_id].get('matched_terms', [])
                combined_results[doc_id]['matched_terms'].extend(result['matched_terms'])
        
        # Sort final results by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]
    
    def explain_score(self, query: str, doc_id: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of how a document was scored for a query.
        
        Args:
            query: Original search query
            doc_id: Document identifier to explain
            
        Returns:
            Dictionary with detailed scoring explanation
        """
        if doc_id not in self.documents:
            return {'error': 'Document not found'}
        
        phrases, terms = self.extract_phrases(query)
        explanation = {
            'document_id': doc_id,
            'query': query,
            'phrases': phrases,
            'terms': terms,
            'total_score': 0.0,
            'score_breakdown': [],
            'document_stats': {
                'length': self.document_lengths.get(doc_id, 0),
                'avg_length': self.average_document_length
            }
        }
        
        # Explain phrase matches
        for phrase in phrases:
            if phrase in self.documents[doc_id]['content'].lower():
                phrase_score = 10.0 * len(self.preprocess_text(phrase))
                explanation['total_score'] += phrase_score
                explanation['score_breakdown'].append({
                    'type': 'phrase',
                    'text': phrase,
                    'score': phrase_score,
                    'reason': 'Exact phrase match'
                })
        
        # Explain term matches
        if terms:
            bm25_score = self.calculate_bm25_score(terms, doc_id)
            explanation['total_score'] += bm25_score
            
            # Detailed term scoring
            query_term_counts = Counter(terms)
            doc_length = self.document_lengths[doc_id]
            
            for term, query_freq in query_term_counts.items():
                if term in self.inverted_index and doc_id in self.inverted_index[term]:
                    term_freq = self.inverted_index[term][doc_id]
                    doc_freq = self.document_frequencies[term]
                    
                    idf = math.log((self.total_documents - doc_freq + 0.5) / (doc_freq + 0.5))
                    normalized_tf = (term_freq * (self.k1 + 1)) / (
                        term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
                    )
                    normalized_qtf = (query_freq * (self.k3 + 1)) / (query_freq + self.k3)
                    term_score = idf * normalized_tf * normalized_qtf
                    
                    explanation['score_breakdown'].append({
                        'type': 'term',
                        'text': term,
                        'score': term_score,
                        'term_freq': term_freq,
                        'doc_freq': doc_freq,
                        'idf': idf,
                        'normalized_tf': normalized_tf,
                        'normalized_qtf': normalized_qtf
                    })
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': self.average_document_length,
            'index_size': sum(len(doc_dict) for doc_dict in self.inverted_index.values()),
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'k3': self.k3
            }
        }