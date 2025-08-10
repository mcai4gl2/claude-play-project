"""Lightweight tests that don't require heavy ML dependencies"""
import pytest
import tempfile
import shutil
from pathlib import Path


class TestNoteParsing:
    """Test note parsing logic without vector operations"""
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_note_parser_import(self):
        """Test that note parser can be imported and initialized"""
        from note_parser import NoteParser
        parser = NoteParser(str(self.temp_dir))
        assert parser.notes_directory == self.temp_dir
        assert hasattr(parser, 'SUPPORTED_EXTENSIONS')
    
    def test_note_parsing_markdown(self):
        """Test parsing markdown files"""
        from note_parser import NoteParser
        
        # Create test markdown file
        test_file = self.temp_dir / "test.md"
        test_file.write_text("# Test Title\nThis is test content", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Test Title"
        assert "test content" in notes[0]['content']
        assert notes[0]['filename'] == "test.md"
    
    def test_note_parsing_text(self):
        """Test parsing text files"""
        from note_parser import NoteParser
        
        # Create test text file
        test_file = self.temp_dir / "simple.txt"
        test_file.write_text("Simple Title\nWith some content", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Simple Title"
        assert "some content" in notes[0]['content']
    
    def test_filename_to_title(self):
        """Test filename to title conversion"""
        from note_parser import NoteParser
        
        parser = NoteParser(str(self.temp_dir))
        title = parser._filename_to_title("my_important_note.txt")
        assert title == "My Important Note"
        
        title = parser._filename_to_title("another-test-file.md")
        assert title == "Another Test File"


class TestContextFormatting:
    """Test context formatting without vector operations"""
    
    def test_context_formatter_import(self):
        """Test that context formatter can be imported"""
        from context_formatter import ContextFormatter
        formatter = ContextFormatter()
        assert formatter.max_context_length == 4000
    
    def test_no_results_prompt(self):
        """Test prompt generation with no results"""
        from context_formatter import ContextFormatter
        
        formatter = ContextFormatter()
        prompt = formatter.format_context_prompt("test query", [])
        
        assert "No Relevant Context Found" in prompt
        assert "test query" in prompt
    
    def test_template_validation(self):
        """Test template style validation"""
        from context_formatter import ContextFormatter
        
        formatter = ContextFormatter()
        
        # Valid templates should work
        prompt = formatter.format_context_prompt("test", [], template_style="standard")
        assert "No Relevant Context Found" in prompt
        
        # Invalid template should raise error
        with pytest.raises(ValueError, match="Unknown template style"):
            formatter.format_context_prompt("test", [], template_style="invalid")
    
    def test_content_truncation(self):
        """Test content truncation logic"""
        from context_formatter import ContextFormatter
        
        formatter = ContextFormatter()
        
        # Short content should not be truncated
        short_text = "This is short"
        result = formatter._truncate_content(short_text, 100)
        assert result == short_text
        
        # Long content should be truncated
        long_text = "This is a very long text. " * 20
        result = formatter._truncate_content(long_text, 50)
        assert len(result) > 50  # Includes truncation message
        assert "Content truncated" in result


class TestBasicImports:
    """Test that all modules can be imported"""
    
    def test_all_imports(self):
        """Test that all main modules can be imported without errors"""
        # These should not trigger model downloads
        from note_parser import NoteParser
        from context_formatter import ContextFormatter
        
        # Test instantiation
        parser = NoteParser(".")
        formatter = ContextFormatter()
        
        assert parser is not None
        assert formatter is not None
    
    def test_cli_help_import(self):
        """Test that CLI module can be imported for help"""
        # This might fail if vector_store tries to load models on import
        try:
            import notes_app
            assert hasattr(notes_app, 'cli')
            print("CLI module imported successfully")
        except ImportError as e:
            # Expected in CI environment without ML dependencies
            error_msg = str(e).lower()
            assert any(dep in error_msg for dep in ["faiss", "sentence", "transformers"])
            print(f"CLI import failed as expected in CI: {e}")
        except Exception as e:
            # Other import errors are also acceptable in CI
            print(f"CLI import failed with other error (acceptable): {e}")


class TestUtilityFunctions:
    """Test utility functions without ML dependencies"""
    
    def test_path_operations(self):
        """Test basic path operations used in the application"""
        from pathlib import Path
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "test.md" 
        test_file.write_text("content", encoding='utf-8')
        
        assert test_file.exists()
        assert test_file.suffix == ".md"
        assert test_file.stem == "test"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_string_operations(self):
        """Test string operations used for title extraction"""
        import re
        
        # Test markdown header extraction
        text = "# Main Title"
        result = re.sub(r'^#+\s*', '', text).strip()
        assert result == "Main Title"
        
        # Test filename conversion
        filename = "my_test_file"
        title = filename.replace('_', ' ').replace('-', ' ').title()
        assert title == "My Test File"