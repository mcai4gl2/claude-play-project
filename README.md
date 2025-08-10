# Notes Vector Database Application

A Python application that parses notes from a subfolder, stores them in a vector database, and provides semantic search functionality to generate context prompts for LLM models.

## 🚀 Features

- **Multi-format Support**: Parse `.md`, `.txt`, and `.rst` files
- **Semantic Search**: Use FAISS and sentence transformers for similarity search
- **Context Generation**: Create LLM-ready prompts with retrieved context
- **CLI Interface**: Easy-to-use command-line interface
- **Persistent Storage**: FAISS provides persistent vector storage
- **Flexible Templates**: Multiple output formats (standard, detailed, minimal)

## 📁 Project Structure

```
notes_app/
├── notes_app.py          # Main CLI application
├── note_parser.py        # File parsing logic
├── vector_store.py       # FAISS integration
├── query_engine.py       # Search functionality
├── context_formatter.py  # Prompt generation
├── requirements.txt      # Dependencies
├── README.md            # This file
├── PLAN.md              # Implementation plan
├── tests/               # Unit test directory
│   ├── test_note_parser.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   └── test_context_formatter.py
└── sample_notes/        # Sample test data
    ├── python_basics.md
    ├── machine_learning_intro.txt
    ├── web_development.md
    ├── data_science_workflow.rst
    └── deep_learning_concepts.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone or download the project files**

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The installation may take several minutes as it downloads large ML models.

### Dependencies

- `faiss-cpu==1.7.4`: Vector database for embeddings
- `sentence-transformers==2.2.2`: Text embedding models
- `pytest==7.4.2`: Testing framework
- `numpy==1.24.3`: Numerical computing
- `click==8.1.7`: CLI interface framework

## 📖 Usage

### Basic Commands

The application provides several commands through the CLI interface:

```bash
python notes_app.py --help
```

### 1. Index Your Notes

First, index your notes to create the vector database:

```bash
# Index notes from default directory (./sample_notes)
python notes_app.py index

# Index from custom directory
python notes_app.py -d /path/to/your/notes index

# Force reindex (overwrites existing index)
python notes_app.py index --force
```

**Expected Output:**
```
📂 Indexing notes from: ./sample_notes
✅ Successfully indexed 5 notes
```

### 2. Query and Generate Context

Search for relevant notes and generate context prompts:

```bash
# Basic query
python notes_app.py query "What is machine learning?"

# Limit results
python notes_app.py query "Python programming" -k 3

# Use different template styles
python notes_app.py query "web development" -t detailed
python notes_app.py query "data science" -t minimal

# Generate copy-ready format
python notes_app.py query "deep learning" --copy-ready

# Exclude source information
python notes_app.py query "algorithms" --no-sources
```

**Expected Output:**
```
🔍 Searching for: What is machine learning?
📋 Found 2 relevant notes:

1. **Machine Learning Introduction** (machine_learning_intro.txt) - Relevance: 0.92
2. **Deep Learning Concepts** (deep_learning_concepts.md) - Relevance: 0.78

📝 Generated context prompt:
────────────────────────────────────────────────────────────
# Context Information

Based on the following relevant notes, please answer the question:

## Note 1: Machine Learning Introduction
*Source: machine_learning_intro.txt (Relevance: 0.92)*

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled data to train models
   - Examples: Classification, Regression
...

---

**Question:** What is machine learning?

Please provide a comprehensive answer based on the context above.
```

### 3. Search by Title

Search notes specifically by title:

```bash
python notes_app.py search-title "Python"
```

### 4. View Statistics

Check index information:

```bash
python notes_app.py stats
```

**Expected Output:**
```
📊 Index Statistics:
   Total notes: 5
   Collection: notes
   Database: ./vector_db
```

### 5. Clear Index

Remove all indexed notes:

```bash
python notes_app.py clear
```

## 🎨 Template Styles

The application supports three template styles for context generation:

### Standard (Default)
- Clean, organized format
- Includes source information and relevance scores
- Best for general use

### Detailed
- Comprehensive format with metadata
- Shows file paths, timestamps, and detailed metrics
- Best for debugging or detailed analysis

### Minimal
- Concise format with just content and question
- Minimal formatting
- Best for token-limited scenarios

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_note_parser.py -v

# Run with coverage (if installed)
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

The project includes extensive unit tests for all modules:

- **note_parser.py**: File parsing, title extraction, encoding handling
- **vector_store.py**: ChromaDB operations, embeddings, search
- **query_engine.py**: Indexing, querying, similarity search
- **context_formatter.py**: Template generation, formatting options

## 📂 Sample Notes

The project includes sample notes covering:

- **Python Programming**: Basic syntax, data structures, libraries
- **Machine Learning**: Types, algorithms, applications
- **Web Development**: Frontend, backend, databases, deployment
- **Data Science**: Workflow, tools, skills required
- **Deep Learning**: Neural networks, architectures, frameworks

## 🔧 Configuration Options

### Command-Line Options

- `--notes-dir, -d`: Specify notes directory (default: `./sample_notes`)
- `--db-dir`: Specify vector database directory (default: `./chroma_db`)
- `--top-k, -k`: Number of results to return (default: 5)
- `--template, -t`: Template style (`standard`, `detailed`, `minimal`)
- `--threshold`: Similarity threshold for filtering results
- `--copy-ready`: Format output for easy copying to LLM
- `--no-sources`: Exclude source information from output

### Supported File Formats

- `.md`: Markdown files
- `.txt`: Plain text files
- `.rst`: reStructuredText files

## 🔍 How It Works

1. **Parsing**: The `note_parser.py` scans directories for supported files and extracts content and metadata
2. **Embedding**: The `vector_store.py` uses sentence-transformers to create embeddings and stores them in FAISS
3. **Querying**: The `query_engine.py` performs semantic search to find relevant notes
4. **Formatting**: The `context_formatter.py` generates LLM-ready prompts with the retrieved context

## 🚨 Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure you've activated the virtual environment and installed dependencies
2. **FAISS errors**: Delete the `vector_db` directory and re-run indexing
3. **Encoding issues**: The parser supports UTF-8, Latin-1, and CP1252 encodings
4. **Empty results**: Check similarity threshold or try broader query terms

### Performance Tips

- **First-time setup**: Initial model download takes time
- **Large note collections**: Consider using similarity threshold to filter results
- **Memory usage**: FAISS stores embeddings in memory during operation

## 🔗 Integration

### Use with Web LLMs

1. Run a query: `python notes_app.py query "your question" --copy-ready`
2. Copy the generated prompt
3. Paste into ChatGPT, Claude, or other LLM interfaces

### API Integration

The core modules can be imported and used programmatically:

```python
from query_engine import QueryEngine
from vector_store import VectorStore

# Initialize
vector_store = VectorStore("./my_db")
engine = QueryEngine(vector_store)

# Index notes
result = engine.index_notes("./my_notes")

# Query
results = engine.search_notes("machine learning", top_k=3)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is released under the MIT License.

## 🙏 Acknowledgments

- [FAISS](https://faiss.ai/) for the vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Click](https://click.palletsprojects.com/) for the CLI interface