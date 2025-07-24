from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# Simple HTML with just the status check functionality
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Status Test</title>
</head>
<body>
    <h1>Status Test</h1>
    <div id="status">Checking...</div>
    <div id="log"></div>
    
    <script>
        function addLog(msg) {
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.textContent = new Date().toLocaleTimeString() + ": " + msg;
            logDiv.prepend(entry);
        }
        
        function checkStatus() {
            addLog("Sending status request...");
            fetch('/api/status')
            .then(response => {
                addLog("Response received: " + response.status);
                return response.json();
            })
            .then(data => {
                addLog("Data received: " + JSON.stringify(data));
                document.getElementById('status').textContent = 
                    "Status: " + data.status + " - " + data.progress + "% - " + data.message;
                
                setTimeout(checkStatus, 2000);
            })
            .catch(error => {
                addLog("Error: " + error);
                setTimeout(checkStatus, 5000);
            });
        }
        
        window.onload = function() {
            addLog("Page loaded");
            setTimeout(checkStatus, 1000);
        };
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    # Always return "Complete" for testing
    return jsonify({
        "status": "Complete",
        "progress": 100,
        "message": "Test status is working"
    })

if __name__ == '__main__':
    app.run(debug=False, port=8088)