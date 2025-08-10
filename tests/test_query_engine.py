import pytest
import tempfile
import shutil
from pathlib import Path
from query_engine import QueryEngine
from vector_store import VectorStore
from note_parser import NoteParser


class TestQueryEngine:
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.notes_dir = self.temp_dir / "notes"
        self.notes_dir.mkdir()
        
        self.vector_store = VectorStore(
            persist_directory=str(self.temp_dir / "test_chroma"),
            collection_name="test_engine"
        )
        self.query_engine = QueryEngine(self.vector_store)
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        assert self.query_engine.vector_store == self.vector_store
    
    def test_search_notes_empty_query(self):
        results = self.query_engine.search_notes("")
        assert results == []
        
        results = self.query_engine.search_notes("   ")
        assert results == []
        
        results = self.query_engine.search_notes(None)
        assert results == []
    
    def test_search_notes_with_results(self):
        # Create test notes
        notes = [
            {
                'content': 'Python is great for machine learning and data science',
                'filename': 'python_ml.md',
                'filepath': '/python_ml.md',
                'title': 'Python ML'
            },
            {
                'content': 'JavaScript is used for web development',
                'filename': 'js_web.md',
                'filepath': '/js_web.md',
                'title': 'JavaScript Web'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        results = self.query_engine.search_notes("machine learning", top_k=1)
        
        assert len(results) == 1
        assert 'relevance_score' in results[0]
        assert results[0]['relevance_score'] > 0
        assert 'Python' in results[0]['content']
    
    def test_search_notes_with_similarity_threshold(self):
        notes = [
            {
                'content': 'Python programming tutorial',
                'filename': 'python.md',
                'filepath': '/python.md',
                'title': 'Python'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        # Search with high threshold - should get results
        results = self.query_engine.search_notes("Python tutorial", 
                                                similarity_threshold=0.5)
        assert len(results) > 0
        
        # Search with very low threshold - should get no results
        results = self.query_engine.search_notes("completely different topic", 
                                                similarity_threshold=0.1)
        assert len(results) == 0
    
    def test_index_notes_empty_directory(self):
        result = self.query_engine.index_notes(str(self.notes_dir))
        
        assert result['status'] == 'no_notes_found'
        assert result['indexed_count'] == 0
    
    def test_index_notes_success(self):
        # Create test files
        (self.notes_dir / "note1.md").write_text("# Note 1\nContent 1", encoding='utf-8')
        (self.notes_dir / "note2.txt").write_text("Note 2 Title\nContent 2", encoding='utf-8')
        
        result = self.query_engine.index_notes(str(self.notes_dir))
        
        assert result['status'] == 'success'
        assert result['indexed_count'] == 2
        assert str(self.notes_dir) in result['notes_directory']
    
    def test_index_notes_force_reindex(self):
        # Create and index initial notes
        (self.notes_dir / "note1.md").write_text("# Note 1\nContent", encoding='utf-8')
        
        # First index
        result1 = self.query_engine.index_notes(str(self.notes_dir))
        assert result1['status'] == 'success'
        assert result1['indexed_count'] == 1
        
        # Index again without force - should be up to date
        result2 = self.query_engine.index_notes(str(self.notes_dir))
        assert result2['status'] == 'up_to_date'
        assert result2['indexed_count'] == 0
        
        # Force reindex
        result3 = self.query_engine.index_notes(str(self.notes_dir), force_reindex=True)
        assert result3['status'] == 'success'
        assert result3['indexed_count'] == 1
    
    def test_index_notes_incremental_update(self):
        # Create and index initial note
        (self.notes_dir / "note1.md").write_text("# Note 1\nContent", encoding='utf-8')
        
        result1 = self.query_engine.index_notes(str(self.notes_dir))
        assert result1['indexed_count'] == 1
        
        # Add a new note
        (self.notes_dir / "note2.md").write_text("# Note 2\nNew content", encoding='utf-8')
        
        result2 = self.query_engine.index_notes(str(self.notes_dir))
        assert result2['status'] == 'incremental_update'
        assert result2['indexed_count'] == 1
        assert result2['total_count'] == 2
    
    def test_get_similar_notes(self):
        notes = [
            {
                'content': 'Python programming concepts and syntax',
                'filename': 'python.md',
                'filepath': '/python.md',
                'title': 'Python Guide'
            },
            {
                'content': 'Python web development with Flask framework',
                'filename': 'flask.md',
                'filepath': '/flask.md',
                'title': 'Flask Tutorial'
            },
            {
                'content': 'Java object-oriented programming principles',
                'filename': 'java.md',
                'filepath': '/java.md',
                'title': 'Java OOP'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        # Search for similar notes to Python content
        similar = self.query_engine.get_similar_notes(
            "Python programming language", top_k=2
        )
        
        assert len(similar) == 2
        # Should return Python-related notes first
        assert any('Python' in note['content'] for note in similar)
    
    def test_get_similar_notes_with_exclusion(self):
        notes = [
            {
                'content': 'Python basics tutorial',
                'filename': 'python1.md',
                'filepath': '/python1.md',
                'title': 'Python Basics'
            },
            {
                'content': 'Advanced Python programming',
                'filename': 'python2.md',
                'filepath': '/python2.md',
                'title': 'Advanced Python'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        similar = self.query_engine.get_similar_notes(
            "Python programming",
            top_k=2,
            exclude_filepath='/python1.md'
        )
        
        # Should exclude the specified file
        assert len(similar) == 1
        assert similar[0]['metadata']['filepath'] == '/python2.md'
    
    def test_get_stats(self):
        stats = self.query_engine.get_stats()
        
        assert 'total_notes' in stats
        assert 'collection_name' in stats
        assert 'persist_directory' in stats
        assert stats['total_notes'] == 0
        assert stats['collection_name'] == "test_engine"
    
    def test_search_by_title(self):
        notes = [
            {
                'content': 'Content about machine learning algorithms',
                'filename': 'ml_guide.md',
                'filepath': '/ml_guide.md',
                'title': 'Machine Learning Guide'
            },
            {
                'content': 'Python programming tutorial content',
                'filename': 'python_tutorial.md',
                'filepath': '/python_tutorial.md',
                'title': 'Python Tutorial'
            },
            {
                'content': 'Advanced machine learning techniques',
                'filename': 'advanced_ml.md',
                'filepath': '/advanced_ml.md',
                'title': 'Advanced ML Techniques'
            }
        ]
        
        self.vector_store.add_notes(notes)
        
        # Search by exact title match
        results = self.query_engine.search_by_title("Machine Learning Guide")
        assert len(results) > 0
        assert results[0]['metadata']['title'] == 'Machine Learning Guide'
        assert 'title_score' in results[0]
        
        # Search by partial title match
        results = self.query_engine.search_by_title("machine learning")
        assert len(results) >= 1
        titles = [r['metadata']['title'] for r in results]
        assert any('Machine Learning' in title for title in titles)
    
    def test_search_by_title_empty_query(self):
        results = self.query_engine.search_by_title("")
        assert results == []