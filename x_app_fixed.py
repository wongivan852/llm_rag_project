from flask import Flask, jsonify, render_template_string
import threading
import time
import os
import json
from datetime import datetime

app = Flask(__name__)

# Global state
is_initialized = False
rag_system = None

# Extremely simple HTML template focusing just on the status check
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Project Management RAG</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #status-container { margin-bottom: 20px; }
        #main-container { display: none; }
        #log { height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-top: 20px; font-size: 12px; }
        .query-box { width: 100%; padding: 10px; margin-bottom: 10px; }
        .submit-btn { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Project Management RAG System</h1>
    
    <div id="status-container">
        <h2>Status: <span id="status-text">Checking...</span></h2>
        <p id="message"></p>
        <button onclick="location.reload()">Refresh Page</button>
    </div>
    
    <div id="main-container">
        <h2>System is Ready!</h2>
        <div>
            <textarea class="query-box" id="query" rows="4" placeholder="Enter your project management question here..."></textarea>
            <button class="submit-btn" onclick="submitQuery()">Submit Query</button>
        </div>
        <div id="response" style="margin-top: 20px;"></div>
    </div>
    
    <div id="log"></div>
    
    <script>
        function log(msg) {
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.textContent = new Date().toLocaleTimeString() + ": " + msg;
            logDiv.prepend(entry);
        }
        
        function checkStatus() {
            log("Checking status...");
            
            // Add timestamp to prevent caching
            fetch('/api/status?t=' + Date.now())
            .then(response => {
                log("Got response: " + response.status);
                return response.json();
            })
            .then(data => {
                log("Status data: " + JSON.stringify(data));
                
                document.getElementById('status-text').textContent = data.status;
                document.getElementById('message').textContent = data.message;
                
                if (data.status === 'ready') {
                    log("System is ready!");
                    document.getElementById('status-container').style.display = 'none';
                    document.getElementById('main-container').style.display = 'block';
                } else {
                    setTimeout(checkStatus, 2000);
                }
            })
            .catch(error => {
                log("Error: " + error);
                setTimeout(checkStatus, 5000);
            });
        }
        
        function submitQuery() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            log("Submitting query: " + query);
            document.getElementById('response').innerHTML = "Processing...";
            
            fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            })
            .then(response => response.json())
            .then(data => {
                log("Got response");
                document.getElementById('response').innerHTML = data.response.replace(/\n/g, '<br>');
            })
            .catch(error => {
                log("Error: " + error);
                document.getElementById('response').innerHTML = "Error processing query.";
            });
        }
        
        window.onload = function() {
            log("Page loaded");
            setTimeout(checkStatus, 1000);
        };
    </script>
</body>
</html>
'''

# Simulate initialization
def initialize():
    global is_initialized
    print("Starting initialization...")
    time.sleep(5)  # Simulate work
    print("Initialization complete!")
    is_initialized = True

@app.route('/')
def home():
    print("Home route accessed")
    # Start initialization if not done
    if not is_initialized:
        print("Starting initialization thread")
        threading.Thread(target=initialize).start()
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    status_data = {
        "status": "ready" if is_initialized else "initializing",
        "message": "System is ready!" if is_initialized else "System is initializing..."
    }
    print(f"Status API called, returning: {status_data}")
    
    # Add no-cache headers
    response = jsonify(status_data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/api/query', methods=['POST'])
def query():
    from flask import request
    query_text = request.json.get('query', '')
    return jsonify({
        "response": f"Response to your query: {query_text}\n\nThis is just a simulated response."
    })

if __name__ == '__main__':
    # Use a non-development server
    from waitress import serve
    print("Starting server on http://127.0.0.1:8082")
    serve(app, host="127.0.0.1", port=8082)