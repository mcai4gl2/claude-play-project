import pytest
import tempfile
import shutil
from pathlib import Path
from vector_store import VectorStore


class TestVectorStore:
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vector_store = VectorStore(
            persist_directory=str(self.temp_dir / "test_chroma"),
            collection_name="test_notes"
        )
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        assert self.vector_store.collection_name == "test_notes"
        expected_path = str(self.temp_dir / "test_chroma")
        actual_path = str(self.vector_store.persist_directory)
        assert expected_path in actual_path or actual_path.endswith("test_chroma")
        assert self.vector_store.embedding_model is not None
    
    def test_add_empty_notes(self):
        count = self.vector_store.add_notes([])
        assert count == 0
    
    def test_add_single_note(self):
        notes = [{
            'content': 'This is a test note about machine learning',
            'filename': 'ml_note.md',
            'filepath': '/path/to/ml_note.md',
            'title': 'Machine Learning Basics'
        }]
        
        count = self.vector_store.add_notes(notes)
        assert count == 1
        
        info = self.vector_store.get_collection_info()
        assert info['count'] == 1
    
    def test_add_multiple_notes(self):
        notes = [
            {
                'content': 'Python is a programming language',
                'filename': 'python.md',
                'filepath': '/path/to/python.md',
                'title': 'Python Programming'
            },
            {
                'content': 'Machine learning involves algorithms and data',
                'filename': 'ml.txt',
                'filepath': '/path/to/ml.txt',
                'title': 'ML Overview'
            }
        ]
        
        count = self.vector_store.add_notes(notes)
        assert count == 2
        
        info = self.vector_store.get_collection_info()
        assert info['count'] == 2
    
    def test_search_empty_query(self):
        notes = [{
            'content': 'Test content',
            'filename': 'test.md',
            'filepath': '/test.md',
            'title': 'Test'
        }]
        self.vector_store.add_notes(notes)
        
        results = self.vector_store.search("")
        assert results == []
        
        results = self.vector_store.search("   ")
        assert results == []
    
    def test_search_with_results(self):
        notes = [
            {
                'content': 'Python is a versatile programming language used for web development',
                'filename': 'python.md',
                'filepath': '/python.md',
                'title': 'Python Guide'
            },
            {
                'content': 'Machine learning algorithms can classify and predict data patterns',
                'filename': 'ml.md',
                'filepath': '/ml.md',
                'title': 'ML Algorithms'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        results = self.vector_store.search("programming language", top_k=1)
        
        assert len(results) == 1
        assert 'content' in results[0]
        assert 'metadata' in results[0]
        assert 'distance' in results[0]
        assert results[0]['metadata']['title'] == 'Python Guide'
    
    def test_search_top_k_parameter(self):
        notes = [
            {'content': f'Note {i} about Python programming', 'filename': f'note{i}.md', 
             'filepath': f'/note{i}.md', 'title': f'Note {i}'}
            for i in range(5)
        ]
        
        self.vector_store.add_notes(notes)
        
        results = self.vector_store.search("Python", top_k=3)
        assert len(results) == 3
        
        results = self.vector_store.search("Python", top_k=10)
        assert len(results) == 5  # Should not exceed available documents
    
    def test_clear_collection(self):
        notes = [{
            'content': 'Test content',
            'filename': 'test.md',
            'filepath': '/test.md',
            'title': 'Test'
        }]
        
        self.vector_store.add_notes(notes)
        assert self.vector_store.get_collection_info()['count'] == 1
        
        cleared_count = self.vector_store.clear_collection()
        assert cleared_count == 1
        assert self.vector_store.get_collection_info()['count'] == 0
    
    def test_clear_empty_collection(self):
        cleared_count = self.vector_store.clear_collection()
        assert cleared_count == 0
    
    def test_get_collection_info(self):
        info = self.vector_store.get_collection_info()
        
        assert 'name' in info
        assert 'count' in info
        assert 'persist_directory' in info
        assert info['name'] == "test_notes"
        assert info['count'] == 0
    
    def test_document_exists(self):
        notes = [{
            'content': 'Test content',
            'filename': 'test.md',
            'filepath': '/path/to/test.md',
            'title': 'Test'
        }]
        
        assert not self.vector_store.document_exists('/path/to/test.md')
        
        self.vector_store.add_notes(notes)
        assert self.vector_store.document_exists('/path/to/test.md')
        assert not self.vector_store.document_exists('/path/to/nonexistent.md')
    
    def test_search_similarity_ordering(self):
        notes = [
            {
                'content': 'Python programming language tutorial',
                'filename': 'python_tutorial.md',
                'filepath': '/python_tutorial.md',
                'title': 'Python Tutorial'
            },
            {
                'content': 'Java programming concepts and examples',
                'filename': 'java_concepts.md',
                'filepath': '/java_concepts.md',
                'title': 'Java Concepts'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        results = self.vector_store.search("Python programming", top_k=2)
        
        assert len(results) == 2
        # First result should be more similar (lower distance)
        assert results[0]['distance'] < results[1]['distance']
        assert 'Python' in results[0]['content']