#!/bin/bash

echo "Starting Project Management RAG System..."
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Kill any existing Python processes that might be using port 8081
echo "Checking for existing processes on port 8081..."
lsof -i:8081 | grep LISTEN | awk '{print $2}' | xargs -r kill -9

# Start the web application
echo "Starting web application on http://localhost:8081"
echo "Note: The system will take some time to initialize. Please be patient."
echo "Check the debug log on the web page for status updates."
python3 app.py