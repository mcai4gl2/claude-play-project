from typing import List, Dict, Optional, Tuple
from vector_store import VectorStore
from note_parser import NoteParser
import os


class QueryEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def search_notes(self, query: str, top_k: int = 5, 
                    similarity_threshold: Optional[float] = None) -> List[Dict[str, any]]:
        if not query or not query.strip():
            return []
        
        results = self.vector_store.search(query, top_k)
        
        if similarity_threshold is not None:
            results = [r for r in results if r['distance'] <= similarity_threshold]
        
        # Add relevance score (1 - normalized distance)
        for result in results:
            result['relevance_score'] = max(0, 1 - result['distance'])
        
        return results
    
    def index_notes(self, notes_directory: str, force_reindex: bool = False) -> Dict[str, any]:
        parser = NoteParser(notes_directory)
        notes = parser.parse_notes()
        
        if not notes:
            return {
                'status': 'no_notes_found',
                'message': f'No supported notes found in {notes_directory}',
                'indexed_count': 0
            }
        
        # Check if we need to reindex
        existing_count = self.vector_store.get_collection_info()['count']
        
        if force_reindex or existing_count == 0:
            if existing_count > 0:
                cleared = self.vector_store.clear_collection()
                print(f"Cleared {cleared} existing documents")
            
            indexed_count = self.vector_store.add_notes(notes)
            
            return {
                'status': 'success',
                'message': f'Successfully indexed {indexed_count} notes',
                'indexed_count': indexed_count,
                'notes_directory': notes_directory
            }
        else:
            # Check for new or updated files
            new_notes = []
            for note in notes:
                if not self.vector_store.document_exists(note['filepath']):
                    new_notes.append(note)
            
            if new_notes:
                indexed_count = self.vector_store.add_notes(new_notes)
                return {
                    'status': 'incremental_update',
                    'message': f'Added {indexed_count} new notes',
                    'indexed_count': indexed_count,
                    'total_count': existing_count + indexed_count
                }
            else:
                return {
                    'status': 'up_to_date',
                    'message': 'Index is up to date',
                    'indexed_count': 0,
                    'total_count': existing_count
                }
    
    def get_similar_notes(self, note_content: str, top_k: int = 3,
                         exclude_filepath: Optional[str] = None) -> List[Dict[str, any]]:
        results = self.search_notes(note_content, top_k + 1)
        
        # Exclude the source note itself if specified
        if exclude_filepath:
            results = [r for r in results 
                      if r['metadata']['filepath'] != exclude_filepath]
        
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, any]:
        info = self.vector_store.get_collection_info()
        
        return {
            'total_notes': info['count'],
            'collection_name': info['name'],
            'persist_directory': info['persist_directory']
        }
    
    def search_by_title(self, title_query: str, top_k: int = 5) -> List[Dict[str, any]]:
        # This is a simple implementation - in a real system you might want 
        # to use a separate title index for better performance
        results = self.vector_store.search(title_query, top_k * 2)
        
        # Filter and score based on title similarity
        title_matches = []
        title_query_lower = title_query.lower()
        
        for result in results:
            title = result['metadata']['title'].lower()
            
            # Simple title scoring: exact match > contains > semantic similarity
            if title == title_query_lower:
                result['title_score'] = 1.0
                title_matches.append(result)
            elif title_query_lower in title or title in title_query_lower:
                result['title_score'] = 0.8
                title_matches.append(result)
            elif result['relevance_score'] > 0.7:
                result['title_score'] = result['relevance_score']
                title_matches.append(result)
        
        # Sort by title score, then by relevance
        title_matches.sort(key=lambda x: (x['title_score'], x['relevance_score']), 
                          reverse=True)
        
        return title_matches[:top_k]