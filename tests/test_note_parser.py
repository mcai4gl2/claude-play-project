import pytest
import tempfile
import shutil
from pathlib import Path
from note_parser import NoteParser


class TestNoteParser:
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_parse_notes_empty_directory(self):
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        assert notes == []
    
    def test_parse_notes_nonexistent_directory(self):
        parser = NoteParser("/nonexistent/directory")
        with pytest.raises(FileNotFoundError):
            parser.parse_notes()
    
    def test_parse_single_markdown_file(self):
        note_file = self.temp_dir / "test.md"
        note_file.write_text("# Test Title\nThis is test content", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Test Title"
        assert notes[0]['content'] == "# Test Title\nThis is test content"
        assert notes[0]['filename'] == "test.md"
        assert "test.md" in notes[0]['filepath']
    
    def test_parse_single_txt_file(self):
        note_file = self.temp_dir / "simple.txt"
        note_file.write_text("Simple Note\nWith some content", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Simple Note"
        assert notes[0]['content'] == "Simple Note\nWith some content"
    
    def test_parse_rst_file_with_title(self):
        note_file = self.temp_dir / "doc.rst"
        content = "RST Document\n============\nContent here"
        note_file.write_text(content, encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "RST Document"
    
    def test_parse_multiple_files(self):
        (self.temp_dir / "note1.md").write_text("# First Note\nContent 1", encoding='utf-8')
        (self.temp_dir / "note2.txt").write_text("Second Note\nContent 2", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 2
        titles = [note['title'] for note in notes]
        assert "First Note" in titles
        assert "Second Note" in titles
    
    def test_parse_recursive_subdirectories(self):
        subdir = self.temp_dir / "subdir"
        subdir.mkdir()
        
        (self.temp_dir / "root.md").write_text("# Root Note\nRoot content", encoding='utf-8')
        (subdir / "sub.txt").write_text("Sub Note\nSub content", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 2
        titles = [note['title'] for note in notes]
        assert "Root Note" in titles
        assert "Sub Note" in titles
    
    def test_ignore_unsupported_extensions(self):
        (self.temp_dir / "note.md").write_text("# Valid Note\nContent", encoding='utf-8')
        (self.temp_dir / "ignored.pdf").write_text("Should be ignored", encoding='utf-8')
        (self.temp_dir / "also_ignored.docx").write_text("Also ignored", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Valid Note"
    
    def test_skip_empty_files(self):
        (self.temp_dir / "empty.md").write_text("", encoding='utf-8')
        (self.temp_dir / "whitespace.txt").write_text("   \n  \n", encoding='utf-8')
        (self.temp_dir / "valid.md").write_text("# Valid\nContent", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "Valid"
    
    def test_title_extraction_from_filename(self):
        (self.temp_dir / "my_important_note.txt").write_text("Some content without clear title", encoding='utf-8')
        
        parser = NoteParser(str(self.temp_dir))
        notes = parser.parse_notes()
        
        assert len(notes) == 1
        assert notes[0]['title'] == "My Important Note"
    
    def test_supported_extensions(self):
        assert '.txt' in NoteParser.SUPPORTED_EXTENSIONS
        assert '.md' in NoteParser.SUPPORTED_EXTENSIONS
        assert '.rst' in NoteParser.SUPPORTED_EXTENSIONS
        assert len(NoteParser.SUPPORTED_EXTENSIONS) == 3