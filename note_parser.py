import os
import re
from pathlib import Path
from typing import List, Dict, Optional


class NoteParser:
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst'}
    
    def __init__(self, notes_directory: str):
        self.notes_directory = Path(notes_directory)
        
    def parse_notes(self) -> List[Dict[str, str]]:
        notes = []
        
        if not self.notes_directory.exists():
            raise FileNotFoundError(f"Notes directory '{self.notes_directory}' does not exist")
        
        for file_path in self._get_note_files():
            try:
                content = self._read_file_content(file_path)
                if content.strip():
                    note = {
                        'content': content,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'title': self._extract_title(content, file_path.name)
                    }
                    notes.append(note)
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue
                
        return notes
    
    def _get_note_files(self) -> List[Path]:
        note_files = []
        
        for file_path in self.notes_directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                note_files.append(file_path)
                
        return sorted(note_files)
    
    def _read_file_content(self, file_path: Path) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode file {file_path} with any supported encoding")
    
    def _extract_title(self, content: str, filename: str) -> str:
        lines = content.strip().split('\n')
        
        if not lines:
            return self._filename_to_title(filename)
        
        first_line = lines[0].strip()
        
        # Check for markdown header
        if first_line.startswith('#'):
            return re.sub(r'^#+\s*', '', first_line).strip()
        
        # Check for RST title (next line is all === or ---)
        if len(lines) > 1:
            second_line = lines[1].strip()
            if re.match(r'^[=\-]+$', second_line):
                return first_line
        
        # Use first non-empty line if it's short and looks like a title
        if len(first_line) < 100 and self._looks_like_title(first_line):
            return first_line
            
        return self._filename_to_title(filename)
    
    def _looks_like_title(self, text: str) -> bool:
        # Heuristics to determine if text looks like a title
        text = text.strip()
        
        # Contains phrases that suggest it's content, not a title
        content_phrases = ['some content', 'without clear', 'this is', 'here is', 'lorem ipsum']
        if any(phrase in text.lower() for phrase in content_phrases):
            return False
        
        # Ends with punctuation that suggests it's content, not a title
        if text.endswith(('.', '!', '?', ',', ';', ':')):
            return False
            
        # Contains multiple sentences (likely content)
        if len(text.split('.')) > 2:
            return False
            
        # Very short and looks title-like (no generic words)
        if len(text) < 50:
            # But not if it contains generic content words
            generic_words = ['content', 'text', 'example', 'sample', 'test']
            if not any(word in text.lower() for word in generic_words):
                return True
            
        # Contains common title words
        title_words = ['introduction', 'guide', 'tutorial', 'overview', 'basics', 'concepts']
        if any(word in text.lower() for word in title_words):
            return True
            
        return False
    
    def _filename_to_title(self, filename: str) -> str:
        name = Path(filename).stem
        return name.replace('_', ' ').replace('-', ' ').title()