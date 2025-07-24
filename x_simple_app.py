from flask import Flask, request, jsonify, render_template_string
import os
import time
import threading

# Configuration
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VECTOR_STORE_PATH = "data/vector_store"

app = Flask(__name__)

# Simple status flag
is_initialized = False

# HTML template - just a minimal version for testing
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Simple RAG Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #log { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-top: 20px; }
        .log-entry { margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Simple RAG Test</h1>
    
    <div id="status-container">
        <h2>System Status: <span id="status-text">Checking...</span></h2>
        <p id="status-message"></p>
    </div>
    
    <div id="main-container" style="display: none;">
        <h2>System is Ready</h2>
        <p>You can now query the system.</p>
    </div>
    
    <h3>Debug Log:</h3>
    <div id="log"></div>
    
    <script>
        function addLog(msg) {
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = new Date().toLocaleTimeString() + ": " + msg;
            logDiv.prepend(entry);
        }
        
        // Check system status
        function checkStatus() {
            addLog("Checking initialization status...");
            
            fetch('/api/status')
            .then(response => {
                addLog("Response received: " + response.status);
                return response.json();
            })
            .then(data => {
                addLog("Status data: " + JSON.stringify(data));
                
                document.getElementById('status-text').textContent = data.status;
                document.getElementById('status-message').textContent = data.message;
                
                if (data.status === 'Complete') {
                    addLog("Initialization complete, showing main interface");
                    document.getElementById('status-container').style.display = 'none';
                    document.getElementById('main-container').style.display = 'block';
                } else {
                    addLog("Status not Complete, checking again in 2 seconds");
                    setTimeout(checkStatus, 2000);
                }
            })
            .catch(error => {
                addLog("Error checking status: " + error);
                setTimeout(checkStatus, 5000);
            });
        }
        
        // Start checking status when page loads
        window.onload = function() {
            addLog("Page loaded, starting status check...");
            setTimeout(checkStatus, 1000);
        };
    </script>
</body>
</html>
'''

def initialize_worker():
    """Simulate initialization"""
    global is_initialized
    print("Starting initialization...")
    time.sleep(5)  # Simulate 5 seconds of work
    is_initialized = True
    print("Initialization complete!")

@app.route('/')
def home():
    print("Home route accessed")
    # Start initialization in a background thread if not already done
    if not is_initialized:
        print("Starting initialization thread")
        threading.Thread(target=initialize_worker).start()
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    """Simple status endpoint"""
    if is_initialized:
        status_data = {
            "status": "Complete",
            "progress": 100,
            "message": "System ready for queries!"
        }
    else:
        status_data = {
            "status": "In progress",
            "progress": 50,
            "message": "System is initializing..."
        }
    
    print(f"Status API called, returning: {status_data}")
    return jsonify(status_data)

if __name__ == '__main__':
    app.run(debug=False, port=8081)