from typing import List, Dict, Optional
from datetime import datetime


class ContextFormatter:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def format_context_prompt(self, query: str, search_results: List[Dict], 
                            include_sources: bool = True,
                            template_style: str = "standard") -> str:
        # Validate template style first
        valid_styles = ["standard", "detailed", "minimal"]
        if template_style not in valid_styles:
            raise ValueError(f"Unknown template style: {template_style}")
        
        if not search_results:
            return self._format_no_results_prompt(query)
        
        if template_style == "standard":
            return self._format_standard_template(query, search_results, include_sources)
        elif template_style == "detailed":
            return self._format_detailed_template(query, search_results, include_sources)
        elif template_style == "minimal":
            return self._format_minimal_template(query, search_results, include_sources)
    
    def _format_standard_template(self, query: str, search_results: List[Dict], 
                                include_sources: bool) -> str:
        context_parts = []
        context_parts.append("# Context Information")
        context_parts.append("")
        context_parts.append("Based on the following relevant notes, please answer the question:")
        context_parts.append("")
        
        # Add each relevant note
        for i, result in enumerate(search_results, 1):
            title = result['metadata']['title']
            content = result['content']
            filename = result['metadata']['filename']
            relevance = result.get('relevance_score', 0)
            
            context_parts.append(f"## Note {i}: {title}")
            if include_sources:
                context_parts.append(f"*Source: {filename} (Relevance: {relevance:.2f})*")
            context_parts.append("")
            
            # Truncate content if too long
            truncated_content = self._truncate_content(content, 800)
            context_parts.append(truncated_content)
            context_parts.append("")
        
        # Add the question
        context_parts.append("---")
        context_parts.append("")
        context_parts.append(f"**Question:** {query}")
        context_parts.append("")
        context_parts.append("Please provide a comprehensive answer based on the context above.")
        
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "\n\n[Context truncated...]"
        
        return full_context
    
    def _format_detailed_template(self, query: str, search_results: List[Dict], 
                                include_sources: bool) -> str:
        context_parts = []
        context_parts.append("# Comprehensive Context Analysis")
        context_parts.append("")
        context_parts.append(f"**Query:** {query}")
        context_parts.append(f"**Retrieved:** {len(search_results)} relevant documents")
        context_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append("")
        
        for i, result in enumerate(search_results, 1):
            title = result['metadata']['title']
            content = result['content']
            filename = result['metadata']['filename']
            filepath = result['metadata']['filepath']
            relevance = result.get('relevance_score', 0)
            distance = result.get('distance', 0)
            
            context_parts.append(f"## Document {i}: {title}")
            if include_sources:
                context_parts.append(f"- **File:** {filename}")
                context_parts.append(f"- **Path:** {filepath}")
                context_parts.append(f"- **Relevance Score:** {relevance:.3f}")
                context_parts.append(f"- **Distance:** {distance:.3f}")
            context_parts.append("")
            
            context_parts.append("### Content:")
            truncated_content = self._truncate_content(content, 1000)
            context_parts.append(truncated_content)
            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")
        
        context_parts.append("## Instructions")
        context_parts.append("Using the documents above, provide a detailed analysis that:")
        context_parts.append("1. Directly answers the query")
        context_parts.append("2. Synthesizes information across sources")
        context_parts.append("3. Highlights key insights and connections")
        context_parts.append("4. References specific documents when appropriate")
        
        return "\n".join(context_parts)
    
    def _format_minimal_template(self, query: str, search_results: List[Dict], 
                               include_sources: bool) -> str:
        context_parts = []
        
        # Combine all content with minimal formatting
        all_content = []
        for result in search_results:
            content = self._truncate_content(result['content'], 500)
            if include_sources:
                source = result['metadata']['filename']
                all_content.append(f"[{source}] {content}")
            else:
                all_content.append(content)
        
        context_parts.append("Context:")
        context_parts.extend(all_content)
        context_parts.append("")
        context_parts.append(f"Question: {query}")
        
        return "\n".join(context_parts)
    
    def _format_no_results_prompt(self, query: str) -> str:
        return f"""# No Relevant Context Found

**Query:** {query}

No relevant notes were found in the knowledge base for this query. 

Please provide a general answer based on your training knowledge, and suggest what type of information could be added to the notes to better answer this question in the future."""
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundary
        truncated = content[:max_length]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > max_length * 0.7:  # If we can keep at least 70% and end at sentence
            return truncated[:last_sentence_end + 1] + "\n\n[Content truncated...]"
        else:
            # Just truncate and add ellipsis
            return truncated.rstrip() + "...\n\n[Content truncated...]"
    
    def format_search_results_summary(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "No results found."
        
        summary_parts = []
        summary_parts.append(f"Found {len(search_results)} relevant notes:")
        summary_parts.append("")
        
        for i, result in enumerate(search_results, 1):
            title = result['metadata']['title']
            filename = result['metadata']['filename']
            relevance = result.get('relevance_score', 0)
            
            summary_parts.append(f"{i}. **{title}** ({filename}) - Relevance: {relevance:.2f}")
        
        return "\n".join(summary_parts)
    
    def create_copy_ready_prompt(self, formatted_context: str) -> str:
        lines = ["=" * 60, "COPY THE CONTENT BELOW TO YOUR LLM:", "=" * 60, ""]
        lines.extend(formatted_context.split('\n'))
        lines.extend(["", "=" * 60, "END OF PROMPT", "=" * 60])
        
        return "\n".join(lines)