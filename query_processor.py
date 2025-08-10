"""
Query Processor for Hybrid Search System

Analyzes and preprocesses search queries to determine the best search strategy
and optimize query terms for both BM25 and vector search components.
"""

import re
import string
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass


class QueryType(Enum):
    """Enumeration of different query types"""
    EXACT = "exact"           # Quoted phrases, technical terms
    CONCEPTUAL = "conceptual" # Natural language questions
    MIXED = "mixed"          # Combination of exact and conceptual
    FILTERED = "filtered"    # Includes metadata filters
    BOOLEAN = "boolean"      # Contains boolean operators


@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    original_query: str
    query_type: QueryType
    phrases: List[str]
    terms: List[str]
    expanded_terms: List[str]
    filters: Dict[str, Any]
    confidence: float
    suggested_mode: str
    metadata: Dict[str, Any]


class QueryProcessor:
    """
    Processes and analyzes search queries to optimize search strategy.
    
    Handles query type detection, term extraction, query expansion,
    and provides recommendations for search mode selection.
    """
    
    def __init__(self):
        """Initialize query processor with default configurations"""
        
        # Stop words (minimal set to preserve technical terms)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'they', 'have', 'had', 'what',
            'said', 'each', 'which', 'do', 'how', 'their', 'if', 'or', 'this'
        }
        
        # Question words that indicate conceptual queries
        self.question_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 
            'could', 'would', 'should', 'does', 'did', 'will', 'is', 'are'
        }
        
        # Technical indicators that suggest exact search
        self.technical_indicators = {
            'error', 'exception', 'function', 'method', 'class', 'variable',
            'config', 'setting', 'parameter', 'argument', 'return', 'import',
            'export', 'api', 'endpoint', 'database', 'query', 'syntax'
        }
        
        # Boolean operators (as whole words to avoid false positives)
        self.boolean_operators = {'and', 'or', 'not', '&&', '||', '!'}
        self.boolean_operator_patterns = [
            r'\band\b', r'\bor\b', r'\bnot\b', r'&&', r'\|\|', r'!'
        ]
        
        # Synonym dictionary for query expansion
        self.synonyms = {
            'ml': ['machine learning', 'ml'],
            'ai': ['artificial intelligence', 'ai'],
            'dl': ['deep learning', 'dl'],
            'nlp': ['natural language processing', 'nlp'],
            'nn': ['neural network', 'neural networks', 'nn'],
            'api': ['application programming interface', 'api'],
            'ui': ['user interface', 'ui'],
            'ux': ['user experience', 'ux'],
            'db': ['database', 'db'],
            'os': ['operating system', 'os'],
            'ide': ['integrated development environment', 'ide']
        }
        
        # File type filters
        self.file_type_patterns = {
            'filetype:md': '.md',
            'filetype:txt': '.txt', 
            'filetype:rst': '.rst',
            'filetype:py': '.py',
            'filetype:js': '.js',
            'filetype:ts': '.ts',
            'filetype:java': '.java',
            'filetype:cpp': '.cpp',
            'filetype:c': '.c',
            'ext:md': '.md',
            'ext:txt': '.txt',
            'ext:rst': '.rst',
            'ext:py': '.py',
            'ext:js': '.js',
            'ext:ts': '.ts'
        }
    
    def preprocess_text(self, text: str, preserve_quotes: bool = False) -> str:
        """
        Preprocess text with optional quote preservation.
        
        Args:
            text: Input text to preprocess
            preserve_quotes: Whether to preserve quoted phrases
            
        Returns:
            Preprocessed text
        """
        if preserve_quotes:
            return text.strip()
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_phrases(self, query: str) -> Tuple[List[str], str]:
        """
        Extract quoted phrases from query and return remaining text.
        
        Args:
            query: Original query string
            
        Returns:
            Tuple of (phrases, remaining_query)
        """
        phrases = []
        remaining_query = query
        
        # Extract quoted phrases (both single and double quotes)
        quote_patterns = [r'"([^"]+)"', r"'([^']+)'"]
        
        for pattern in quote_patterns:
            for match in re.finditer(pattern, query):
                phrase = match.group(1).strip().lower()
                if phrase:  # Only add non-empty phrases
                    phrases.append(phrase)
                    remaining_query = remaining_query.replace(match.group(0), ' ')
        
        return phrases, remaining_query.strip()
    
    def extract_filters(self, query: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract metadata filters from query.
        
        Args:
            query: Query string that may contain filters
            
        Returns:
            Tuple of (filters_dict, remaining_query)
        """
        filters = {}
        remaining_query = query
        
        # File type filters
        for pattern, extension in self.file_type_patterns.items():
            if pattern in query.lower():
                filters['file_extension'] = extension
                remaining_query = remaining_query.replace(pattern, ' ')
        
        # Date filters (basic patterns)
        date_patterns = [
            (r'after:(\d{4}-\d{2}-\d{2})', 'date_after'),
            (r'before:(\d{4}-\d{2}-\d{2})', 'date_before'),
            (r'year:(\d{4})', 'year')
        ]
        
        for pattern, filter_name in date_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                filters[filter_name] = match.group(1)
                remaining_query = remaining_query.replace(match.group(0), ' ')
        
        # Title filter
        title_pattern = r'title:([^\s]+)'
        title_matches = re.finditer(title_pattern, query.lower())
        for match in title_matches:
            filters['title_contains'] = match.group(1)
            remaining_query = remaining_query.replace(match.group(0), ' ')
        
        return filters, remaining_query.strip()
    
    def tokenize(self, text: str, remove_stop_words: bool = False) -> List[str]:
        """
        Tokenize text into individual terms.
        
        Args:
            text: Text to tokenize
            remove_stop_words: Whether to remove stop words
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Preserve technical patterns (dots, underscores) temporarily
        preserved_patterns = []
        pattern_replacements = {}
        
        # Find technical patterns to preserve
        technical_patterns = [
            r'\b\w+\.\w+(?:\.\w+)*\b',  # api.endpoint.method
            r'\b\w+_\w+(?:_\w+)*\b',    # function_name
            r'\b[A-Z_]+\b'              # ERROR_CODES (but lowercased)
        ]
        
        for i, pattern in enumerate(technical_patterns):
            for match in re.finditer(pattern, text):
                placeholder = f"__TECH_{i}_{len(preserved_patterns)}__"
                preserved_patterns.append(match.group(0))
                pattern_replacements[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
        
        # Remove remaining punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split on whitespace
        tokens = text.split()
        
        # Restore technical patterns
        for i, token in enumerate(tokens):
            if token in pattern_replacements:
                tokens[i] = pattern_replacements[token]
        
        # Remove stop words if requested
        if remove_stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def detect_query_type(self, query: str, phrases: List[str], terms: List[str], 
                         filters: Dict[str, Any]) -> Tuple[QueryType, float]:
        """
        Detect the type of query and confidence level.
        
        Args:
            query: Original query string
            phrases: Extracted phrases
            terms: Extracted terms
            filters: Extracted filters
            
        Returns:
            Tuple of (query_type, confidence)
        """
        confidence = 0.5  # Base confidence
        
        # Check for filters first
        if filters:
            return QueryType.FILTERED, 0.9
        
        # Check for boolean operators (using patterns to avoid false positives)
        query_lower = query.lower()
        has_boolean = any(re.search(pattern, query_lower) for pattern in self.boolean_operator_patterns)
        if has_boolean:
            return QueryType.BOOLEAN, 0.8
        
        # Check for exact indicators
        exact_indicators = 0
        conceptual_indicators = 0
        
        # Quoted phrases strongly indicate exact search
        if phrases:
            exact_indicators += len(phrases) * 2
        
        # Question words indicate conceptual search
        first_word = terms[0] if terms else ""
        if first_word in self.question_words:
            conceptual_indicators += 2
        
        # Check for question patterns
        question_patterns = [r'\?$', r'^how\s+to', r'^what\s+is', r'^why\s+']
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                conceptual_indicators += 1
        
        # Technical terms suggest exact search (both dictionary and patterns)
        technical_term_count = sum(1 for term in terms if term in self.technical_indicators)
        # Also check for technical patterns
        for term in terms:
            if ('.' in term and not term.startswith('.') and not term.endswith('.')) or \
               ('_' in term and term.replace('_', '').isalpha()) or \
               (term.isupper() and len(term) > 2):
                technical_term_count += 1
        exact_indicators += technical_term_count
        
        # Short queries with technical terms are likely exact
        if len(terms) <= 3 and technical_term_count > 0:
            exact_indicators += 1
        
        # Long queries are more likely conceptual
        if len(terms) > 6:
            conceptual_indicators += 1
        
        # Natural language patterns
        if any(phrase in query_lower for phrase in ['how do i', 'what does', 'can you explain']):
            conceptual_indicators += 2
        
        # Determine type based on indicators
        total_indicators = exact_indicators + conceptual_indicators
        
        if total_indicators == 0:
            return QueryType.CONCEPTUAL, 0.5  # Default to conceptual
        
        exact_ratio = exact_indicators / total_indicators
        confidence = min(0.9, 0.5 + (abs(exact_ratio - 0.5) * 0.8))
        
        if exact_indicators > conceptual_indicators:
            if conceptual_indicators > 0:
                return QueryType.MIXED, confidence
            else:
                return QueryType.EXACT, confidence
        elif conceptual_indicators > exact_indicators:
            if exact_indicators > 0:
                return QueryType.MIXED, confidence
            else:
                return QueryType.CONCEPTUAL, confidence
        else:
            return QueryType.MIXED, confidence
    
    def expand_query(self, terms: List[str]) -> List[str]:
        """
        Expand query terms with synonyms and related terms.
        
        Args:
            terms: Original query terms
            
        Returns:
            Expanded list of terms including synonyms
        """
        expanded = terms.copy()
        
        for term in terms:
            if term in self.synonyms:
                # Add synonyms but avoid duplicates
                for synonym in self.synonyms[term]:
                    if synonym not in expanded:
                        expanded.append(synonym)
        
        return expanded
    
    def suggest_search_mode(self, query_type: QueryType, confidence: float, 
                          has_phrases: bool, has_filters: bool) -> str:
        """
        Suggest the best search mode based on query analysis.
        
        Args:
            query_type: Detected query type
            confidence: Confidence in type detection
            has_phrases: Whether query contains phrases
            has_filters: Whether query contains filters
            
        Returns:
            Suggested search mode
        """
        if has_filters:
            return 'traditional'  # Filters work best with traditional search
        
        if query_type == QueryType.EXACT and confidence > 0.7:
            return 'traditional'
        elif query_type == QueryType.CONCEPTUAL and confidence > 0.7:
            return 'semantic'
        elif query_type == QueryType.MIXED or confidence < 0.6:
            return 'hybrid'
        elif query_type == QueryType.BOOLEAN:
            return 'traditional'
        else:
            return 'auto'  # Let the hybrid engine decide
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform complete query analysis.
        
        Args:
            query: Original search query
            
        Returns:
            QueryAnalysis object with all analysis results
        """
        if not query or not query.strip():
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.CONCEPTUAL,
                phrases=[],
                terms=[],
                expanded_terms=[],
                filters={},
                confidence=0.0,
                suggested_mode='semantic',
                metadata={
                    'empty_query': True,
                    'phrase_count': 0,
                    'term_count': 0,
                    'expanded_term_count': 0,
                    'filter_count': 0,
                    'query_length': 0,
                    'has_question_words': False,
                    'has_technical_terms': False,
                    'has_boolean_operators': False
                }
            )
        
        # Extract components
        phrases, remaining_after_phrases = self.extract_phrases(query)
        filters, remaining_after_filters = self.extract_filters(remaining_after_phrases)
        terms = self.tokenize(remaining_after_filters, remove_stop_words=True)
        expanded_terms = self.expand_query(terms)
        
        # Detect query type
        query_type, confidence = self.detect_query_type(
            query, phrases, terms, filters
        )
        
        # Suggest search mode
        suggested_mode = self.suggest_search_mode(
            query_type, confidence, len(phrases) > 0, len(filters) > 0
        )
        
        # Collect metadata
        metadata = {
            'phrase_count': len(phrases),
            'term_count': len(terms),
            'expanded_term_count': len(expanded_terms),
            'filter_count': len(filters),
            'query_length': len(query),
            'has_question_words': any(term in self.question_words for term in terms),
            'has_technical_terms': any(term in self.technical_indicators for term in terms),
            'has_boolean_operators': any(re.search(pattern, query.lower()) for pattern in self.boolean_operator_patterns)
        }
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            phrases=phrases,
            terms=terms,
            expanded_terms=expanded_terms,
            filters=filters,
            confidence=confidence,
            suggested_mode=suggested_mode,
            metadata=metadata
        )
    
    def get_query_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """
        Generate query suggestions based on analysis.
        
        Args:
            query: Original query
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested alternative queries
        """
        analysis = self.analyze_query(query)
        suggestions = []
        
        # If query has no phrases but has technical terms, suggest adding quotes
        if not analysis.phrases and analysis.metadata['has_technical_terms']:
            technical_terms = [term for term in analysis.terms if term in self.technical_indicators]
            for term in technical_terms[:2]:  # Suggest quoting first 2 technical terms
                suggestions.append(f'"{term}"')
        
        # If query is very long, suggest shorter alternatives
        if analysis.metadata['term_count'] > 8:
            # Suggest using just the first few terms
            shorter = ' '.join(analysis.terms[:4])
            suggestions.append(shorter)
        
        # If query has synonyms, suggest expanded versions
        expanded_different = [term for term in analysis.expanded_terms if term not in analysis.terms]
        if expanded_different:
            for term in expanded_different[:3]:
                suggestions.append(term)
        
        # If no question words but seems conceptual, suggest question format
        if (analysis.query_type == QueryType.CONCEPTUAL and 
            not analysis.metadata['has_question_words'] and 
            analysis.metadata['term_count'] > 2):
            suggestions.append(f"What is {' '.join(analysis.terms[:3])}?")
            suggestions.append(f"How does {' '.join(analysis.terms[:3])} work?")
        
        return suggestions[:max_suggestions]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get query processor statistics and configuration.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'stop_words_count': len(self.stop_words),
            'question_words_count': len(self.question_words),
            'technical_indicators_count': len(self.technical_indicators),
            'synonym_entries': len(self.synonyms),
            'supported_file_types': list(self.file_type_patterns.values()),
            'boolean_operators': list(self.boolean_operators)
        }