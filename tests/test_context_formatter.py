import pytest
from context_formatter import ContextFormatter


class TestContextFormatter:
    
    def setup_method(self):
        self.formatter = ContextFormatter(max_context_length=1000)
    
    def test_initialization(self):
        assert self.formatter.max_context_length == 1000
        
        custom_formatter = ContextFormatter(max_context_length=500)
        assert custom_formatter.max_context_length == 500
    
    def test_format_context_prompt_empty_results(self):
        query = "What is machine learning?"
        results = []
        
        prompt = self.formatter.format_context_prompt(query, results)
        
        assert "No Relevant Context Found" in prompt
        assert query in prompt
        assert "general answer" in prompt
    
    def test_format_standard_template(self):
        query = "Explain Python programming"
        results = [
            {
                'content': 'Python is a high-level programming language',
                'metadata': {
                    'title': 'Python Basics',
                    'filename': 'python.md',
                    'filepath': '/python.md'
                },
                'relevance_score': 0.85
            }
        ]
        
        prompt = self.formatter.format_context_prompt(query, results, template_style="standard")
        
        assert "# Context Information" in prompt
        assert "Note 1: Python Basics" in prompt
        assert "python.md" in prompt
        assert "0.85" in prompt
        assert query in prompt
        assert "comprehensive answer" in prompt
    
    def test_format_detailed_template(self):
        query = "How does machine learning work?"
        results = [
            {
                'content': 'Machine learning uses algorithms to find patterns in data',
                'metadata': {
                    'title': 'ML Introduction',
                    'filename': 'ml_intro.md',
                    'filepath': '/docs/ml_intro.md'
                },
                'relevance_score': 0.92,
                'distance': 0.08
            }
        ]
        
        prompt = self.formatter.format_context_prompt(query, results, template_style="detailed")
        
        assert "# Comprehensive Context Analysis" in prompt
        assert "Document 1: ML Introduction" in prompt
        assert "ml_intro.md" in prompt
        assert "/docs/ml_intro.md" in prompt
        assert "0.920" in prompt
        assert "0.080" in prompt
        assert "## Instructions" in prompt
    
    def test_format_minimal_template(self):
        query = "What is Python?"
        results = [
            {
                'content': 'Python is a programming language',
                'metadata': {
                    'title': 'Python',
                    'filename': 'python.txt'
                },
                'relevance_score': 0.9
            }
        ]
        
        prompt = self.formatter.format_context_prompt(query, results, template_style="minimal")
        
        assert "Context:" in prompt
        assert "[python.txt]" in prompt
        assert "Question: What is Python?" in prompt
        assert len(prompt.split('\n')) < 10  # Should be concise
    
    def test_invalid_template_style(self):
        query = "Test query"
        results = []
        
        with pytest.raises(ValueError, match="Unknown template style"):
            self.formatter.format_context_prompt(query, results, template_style="invalid")
    
    def test_include_sources_false(self):
        query = "Test query"
        results = [
            {
                'content': 'Test content',
                'metadata': {
                    'title': 'Test',
                    'filename': 'test.md'
                },
                'relevance_score': 0.8
            }
        ]
        
        prompt = self.formatter.format_context_prompt(query, results, include_sources=False)
        
        assert "test.md" not in prompt
        assert "0.8" not in prompt
        assert "Test content" in prompt
    
    def test_truncate_content_short_text(self):
        content = "This is a short text."
        truncated = self.formatter._truncate_content(content, 100)
        
        assert truncated == content
    
    def test_truncate_content_long_text_with_sentence(self):
        content = "This is the first sentence. This is the second sentence. This is a very long third sentence that goes on and on."
        truncated = self.formatter._truncate_content(content, 60)
        
        assert "first sentence." in truncated
        assert "Content truncated" in truncated
        assert len(truncated) < len(content)
    
    def test_truncate_content_long_text_no_sentence(self):
        content = "This is a very long text without proper sentence endings that just keeps going and going"
        truncated = self.formatter._truncate_content(content, 30)
        
        assert truncated.endswith("...\n\n[Content truncated...]")
        assert len(truncated) > 30  # Should be longer due to truncation message
    
    def test_max_context_length_truncation(self):
        long_formatter = ContextFormatter(max_context_length=200)
        query = "Test"
        results = [
            {
                'content': 'Very long content ' * 50,  # This will be very long
                'metadata': {
                    'title': 'Long Note',
                    'filename': 'long.md'
                },
                'relevance_score': 0.9
            }
        ]
        
        prompt = long_formatter.format_context_prompt(query, results)
        
        assert len(prompt) <= 250  # Should be around max_context_length + some buffer
        assert "Context truncated" in prompt
    
    def test_format_search_results_summary_empty(self):
        summary = self.formatter.format_search_results_summary([])
        assert summary == "No results found."
    
    def test_format_search_results_summary_with_results(self):
        results = [
            {
                'metadata': {
                    'title': 'First Note',
                    'filename': 'first.md'
                },
                'relevance_score': 0.95
            },
            {
                'metadata': {
                    'title': 'Second Note',
                    'filename': 'second.txt'
                },
                'relevance_score': 0.78
            }
        ]
        
        summary = self.formatter.format_search_results_summary(results)
        
        assert "Found 2 relevant notes:" in summary
        assert "1. **First Note** (first.md) - Relevance: 0.95" in summary
        assert "2. **Second Note** (second.txt) - Relevance: 0.78" in summary
    
    def test_create_copy_ready_prompt(self):
        original_prompt = "This is a test prompt\nwith multiple lines"
        copy_ready = self.formatter.create_copy_ready_prompt(original_prompt)
        
        assert "COPY THE CONTENT BELOW TO YOUR LLM:" in copy_ready
        assert "=" * 60 in copy_ready
        assert "END OF PROMPT" in copy_ready
        assert original_prompt in copy_ready
        
        lines = copy_ready.split('\n')
        assert len(lines) > 4  # Should have header, content, and footer
    
    def test_multiple_results_ordering(self):
        query = "Programming languages"
        results = [
            {
                'content': 'Python content',
                'metadata': {
                    'title': 'Python Guide',
                    'filename': 'python.md'
                },
                'relevance_score': 0.9
            },
            {
                'content': 'Java content',
                'metadata': {
                    'title': 'Java Tutorial',
                    'filename': 'java.md'
                },
                'relevance_score': 0.8
            }
        ]
        
        prompt = self.formatter.format_context_prompt(query, results)
        
        # Should maintain order of results
        python_index = prompt.find("Note 1: Python Guide")
        java_index = prompt.find("Note 2: Java Tutorial")
        
        assert python_index < java_index
        assert python_index > -1
        assert java_index > -1