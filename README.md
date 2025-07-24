# llm_rag_project

A Flask-based Retrieval-Augmented Generation (RAG) system for Project Management knowledge using LangChain and Llama models.

## Features

- 🧠 **AI-Powered Responses** using LlamaCpp with fallback model support
- 📚 **Comprehensive Knowledge Base** with 14,813+ document chunks from 24 project management books
- 🌐 **Web Interface** for interactive querying
- 🔍 **Source Citations** showing which documents were used for answers
- 📊 **Knowledge Auditing** and logging
- ⚡ **Fast Performance** with optimized chunking and retrieval

## Quick Start

1. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:

   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open `http://localhost:8081` in your browser

## Project Structure

```text
llm_rag_project/
├── app.py                 # Main Flask application
├── rag-interface.html     # Web interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore patterns
├── data/                 # Document storage and vector store
│   ├── pmp_combined.txt  # Main knowledge base
│   └── vector_store/     # FAISS vector database
├── models/               # AI model files (not in git)
├── logs/                 # Application logs
├── scripts/              # Utility scripts
│   ├── rebuild_vector_db.py    # Rebuild vector database
│   ├── setup_checker.py        # Environment checker
│   ├── clean_sources.py        # Data cleaning
│   └── ...
├── docs/                 # Documentation
│   ├── Project Management RAG System - Complete Setup Summary.pdf
│   └── ...
└── archive/              # Old files and backups
```

## API Endpoints

- `GET /` - Web interface
- `GET /api/info` - System information (JSON)
- `GET /api/status` - Initialization status
- `POST /api/query` - Query the RAG system

### Query API Example

```bash
curl -X POST http://localhost:8081/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key phases of project management?"}'
```

## Model Support

The system supports model fallback:

1. **Primary**: Llama 3.2 3B Instruct (`llama-3.2-3b-instruct-q4_k_m.gguf`)
2. **Fallback**: TinyLlama 1.1B (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`)

If the primary model fails to load, the system automatically falls back to the smaller model.

## Development

### Scripts

- `scripts/rebuild_vector_db.py` - Rebuild the vector database from documents
- `scripts/setup_checker.py` - Check system requirements and setup
- `scripts/clean_sources.py` - Clean and preprocess document sources

### Logs

Application logs are stored in `logs/` with detailed audit trails of queries and responses.

## Requirements

- Python 3.8+
- Flask
- LangChain
- llama-cpp-python
- FAISS
- sentence-transformers

See `requirements.txt` for complete list.

## License

This project is for educational and research purposes.
