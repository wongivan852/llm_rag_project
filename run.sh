#!/bin/bash
# Run script for Project Management RAG System

echo "🚀 Starting Project Management RAG System..."
echo "📂 Working directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
    source .venv/bin/activate
fi

# Check if models exist
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ] && [ ! -f "models/llama-3.2-3b-instruct-q4_k_m.gguf" ]; then
    echo "⚠️  No models found in models/ directory"
    echo "📥 Please download a compatible GGUF model file to the models/ directory"
    exit 1
fi

# Check if data exists
if [ ! -f "data/pmp_combined.txt" ]; then
    echo "⚠️  Data file not found: data/pmp_combined.txt"
    exit 1
fi

echo "🌐 Starting server at http://localhost:8081"
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"

python app.py
