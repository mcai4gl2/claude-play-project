# Python Notes Vector Database Application Plan

## Core Components:
1. **Note Parser** - Scan subfolder for text files (.txt, .md, etc.) and extract content
2. **Vector Database** - Use ChromaDB (lightweight, local) for embedding storage
3. **Embedding Model** - Use sentence-transformers for text vectorization 
4. **Query Engine** - Search functionality returning top-k most similar notes
5. **Context Generator** - Format results into a prompt template for LLM use

## Implementation Steps:
1. **Setup & Dependencies** - Create requirements.txt with chromadb, sentence-transformers, pytest
2. **Note Parser Module** (`note_parser.py`) - Recursive file scanner with content extraction
3. **Vector Store Module** (`vector_store.py`) - ChromaDB integration with embeddings
4. **Query Module** (`query_engine.py`) - Similarity search with configurable parameters
5. **Context Formatter** (`context_formatter.py`) - LLM prompt template generator
6. **Main CLI Script** (`notes_app.py`) - Commands for indexing and querying
7. **Unit Tests** - Comprehensive test suite for all modules using pytest
8. **Test Data Generation** - Sample notes in various formats for demonstration
9. **Documentation** - README.md with installation, usage, and examples
10. **Plan Documentation** - Save this plan as PLAN.md for reference
11. **Demo** - Run the complete workflow to show functionality

## File Structure:
```
notes_app/
├── notes_app.py          # Main CLI application
├── note_parser.py        # File parsing logic
├── vector_store.py       # ChromaDB integration
├── query_engine.py       # Search functionality
├── context_formatter.py  # Prompt generation
├── requirements.txt      # Dependencies
├── README.md            # Usage documentation
├── PLAN.md              # This implementation plan
├── tests/               # Unit test directory
│   ├── test_note_parser.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   └── test_context_formatter.py
└── sample_notes/        # Test data
    ├── note1.md
    ├── note2.txt
    └── note3.md
```

## Key Features:
- Support multiple file formats (.txt, .md, .rst)
- Persistent vector storage with ChromaDB
- Configurable similarity search (top-k results)
- Rich context formatting with source references
- CLI interface: `python notes_app.py index` and `python notes_app.py query "question"`
- Comprehensive test coverage with pytest
- Sample data and complete demo workflow