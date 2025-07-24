# Scripts Directory

This directory contains utility scripts for managing the RAG system.

## Script Descriptions

### Core Scripts

- **`rebuild_vector_db.py`** - Rebuilds the FAISS vector database from document sources
- **`setup_checker.py`** - Checks system requirements and validates setup

### Data Processing Scripts

- **`clean_sources.py`** - Cleans and preprocesses document sources
- **`fix_citation_sources.py`** - Fixes citation formatting in documents
- **`accurate_rebuild.py`** - Enhanced version of vector database rebuild
- **`manual_sources.py`** - Manual document source management

### Maintenance Scripts

- **`fix_langchain_versions.py`** - Handles LangChain version compatibility
- **`process_bookmarked_file.py`** - Processes bookmarked content
- **`quick_fix.py`** - Quick fixes for common issues

## Usage

Run scripts from the project root directory:

```bash
python scripts/script_name.py
```

Most scripts are designed to be run when the main application is not running to avoid conflicts with file access.
