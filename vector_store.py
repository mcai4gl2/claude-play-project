import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import uuid


class VectorStore:
    def __init__(self, persist_directory: str = "./vector_db", collection_name: str = "notes"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create persist directory
        self.persist_directory.mkdir(exist_ok=True)
        
        # File paths for persistence
        self.index_file = self.persist_directory / f"{collection_name}_index.faiss"
        self.metadata_file = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        # Initialize or load index
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        if self.index_file.exists() and self.metadata_file.exists():
            # Load existing index
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.documents = []
            self.metadatas = []
            self.ids = []
    
    def _save_index(self):
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_notes(self, notes: List[Dict[str, str]]) -> int:
        if not notes:
            return 0
            
        documents = []
        metadatas = []
        ids = []
        
        for note in notes:
            doc_id = str(uuid.uuid4())
            documents.append(note['content'])
            metadatas.append({
                'filename': note['filename'],
                'filepath': note['filepath'],
                'title': note['title']
            })
            ids.append(doc_id)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Save to disk
        self._save_index()
        
        return len(notes)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        if not query.strip() or self.index.ntotal == 0:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, top_k)
        
        search_results = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                # Convert inner product back to distance (1 - similarity)
                similarity = distances[0][i]
                distance = max(0, 1 - similarity)
                
                result = {
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'distance': distance,
                    'id': self.ids[idx]
                }
                search_results.append(result)
        
        return search_results
    
    def clear_collection(self) -> int:
        count = self.index.ntotal if self.index else 0
        
        # Reset index and data
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # Save empty state
        self._save_index()
        
        return count
    
    def get_collection_info(self) -> Dict[str, any]:
        count = self.index.ntotal if self.index else 0
        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def document_exists(self, filepath: str) -> bool:
        for metadata in self.metadatas:
            if metadata.get('filepath') == filepath:
                return True
        return False