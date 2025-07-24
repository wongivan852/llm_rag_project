from flask import Flask, request, jsonify, send_file
import os
import json
import time
import threading
from datetime import datetime
from flask_cors import CORS
import re

# Configuration
DATA_PATH = "data/pmp_combined.txt"
MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
FALLBACK_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VECTOR_STORE_PATH = "data/vector_store"
LOGS_PATH = "logs"

# Ensure directories exist
os.makedirs(LOGS_PATH, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Global variables
is_initialized = False
rag_system = None
initialization_error = None

# Try to import LlamaCpp
try:
    from langchain_community.llms import LlamaCpp
    LLAMACPP_AVAILABLE = True
    print("✅ LlamaCpp import successful")
except ImportError as e:
    LLAMACPP_AVAILABLE = False
    print(f"❌ LlamaCpp import failed: {e}")

class KnowledgeAuditor:
    """Tracks and audits knowledge utilization"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOGS_PATH, f"audit_{self.session_id}.json")
        self.audit_records = []
        
    def log_query(self, query):
        record = {
            "timestamp": time.time(),
            "event_type": "query",
            "query": query
        }
        self.audit_records.append(record)
        self._save_logs()
        return record
        
    def log_retrieval(self, query, docs):
        record = {
            "timestamp": time.time(),
            "event_type": "retrieval",
            "query": query,
            "chunks_retrieved": len(docs)
        }
        self.audit_records.append(record)
        self._save_logs()
        return docs
    
    def log_generation(self, query, response, runtime):
        record = {
            "timestamp": time.time(),
            "event_type": "generation",
            "query": query,
            "response": response,
            "runtime_seconds": runtime
        }
        self.audit_records.append(record)
        self._save_logs()
    
    def generate_audit_report(self):
        queries = [r for r in self.audit_records if r["event_type"] == "query"]
        retrievals = [r for r in self.audit_records if r["event_type"] == "retrieval"]
        generations = [r for r in self.audit_records if r["event_type"] == "generation"]
        
        return {
            "session_id": self.session_id,
            "total_queries": len(queries),
            "total_retrievals": len(retrievals),
            "total_generations": len(generations),
            "detailed_logs": self.log_file
        }
    
    def _save_logs(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.audit_records, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save logs: {e}")

class ProjectManagementRAG:
    """Simple but effective RAG system for project management"""
    
    def __init__(self):
        self.auditor = KnowledgeAuditor()
        self.llm = None
        self.documents = []
        self.doc_metadata = []
        
    def extract_book_info(self, text):
        """Extract book title and section information from text"""
        # Look for book markers
        book_match = re.search(r'\[BOOK:\s*([^\]]+)\]', text)
        book_title = book_match.group(1) if book_match else "Unknown Source"
        
        # Look for chapter/section headers (lines that are all caps or title case)
        lines = text.split('\n')
        section_title = None
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 5 and len(line) < 100:
                # Check if it's a section header (all caps, title case, or numbered)
                if (line.isupper() or 
                    line.istitle() or 
                    re.match(r'^\d+\.', line) or
                    re.match(r'^Chapter \d+', line, re.IGNORECASE)):
                    section_title = line
                    break
        
        return book_title, section_title
    
    def initialize(self):
        global initialization_error
        
        try:
            print("🚀 Initializing Project Management RAG...")
            start_time = time.time()
            
            # Check if LlamaCpp is available
            if not LLAMACPP_AVAILABLE:
                error_msg = "LlamaCpp not available. Please install: pip install llama-cpp-python"
                print(f"ERROR: {error_msg}")
                initialization_error = error_msg
                return False
            
            # Check files
            if not os.path.exists(MODEL_PATH):
                error_msg = f"Model file not found at {MODEL_PATH}"
                print(f"ERROR: {error_msg}")
                initialization_error = error_msg
                return False
            
            if not os.path.exists(DATA_PATH):
                error_msg = f"Data file not found at {DATA_PATH}"
                print(f"ERROR: {error_msg}")
                initialization_error = error_msg
                return False
            
            # Load documents with enhanced metadata extraction
            print("📚 Loading project management documents...")
            try:
                with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Enhanced chunking - split on book boundaries and major sections
                # First split by book markers
                book_sections = re.split(r'(\[BOOK:[^\]]+\])', content)
                
                current_book = "Unknown Source"
                chunk_id = 0
                
                for section in book_sections:
                    if section.startswith('[BOOK:'):
                        current_book = re.search(r'\[BOOK:\s*([^\]]+)\]', section).group(1)
                        continue
                    
                    if len(section.strip()) < 100:  # Skip very short sections
                        continue
                    
                    # Further split long sections into manageable chunks
                    sub_chunks = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*[A-Z])', section)
                    
                    for chunk in sub_chunks:
                        chunk = chunk.strip()
                        if len(chunk) > 100:  # Only keep substantial chunks
                            # Extract additional metadata
                            book_title, section_title = self.extract_book_info(f"[BOOK: {current_book}]\n{chunk}")
                            
                            # Get a meaningful preview
                            preview = chunk[:200].replace('\n', ' ').strip()
                            if len(chunk) > 200:
                                preview += "..."
                            
                            self.documents.append(chunk)
                            self.doc_metadata.append({
                                "chunk_id": chunk_id,
                                "book_title": book_title,
                                "section_title": section_title,
                                "length": len(chunk),
                                "preview": preview,
                                "source": "pmp_combined.txt"
                            })
                            chunk_id += 1
                
                print(f"✅ Loaded {len(self.documents)} document chunks from {len(set([m['book_title'] for m in self.doc_metadata]))} books")
                
            except Exception as e:
                error_msg = f"Failed to load documents: {e}"
                print(f"ERROR: {error_msg}")
                initialization_error = error_msg
                return False
            
            # Initialize LLM with robust settings and fallback
            print("🧠 Initializing Llama model...")
            model_to_use = MODEL_PATH
            model_name = "Llama 3.2 3B"
            
            # Try primary model first
            try:
                self.llm = LlamaCpp(
                    model_path=MODEL_PATH,
                    temperature=0.1,      # Lower temperature for more focused answers
                    max_tokens=1024,      # Reasonable token limit
                    n_ctx=4096,          # Context window
                    verbose=False,        # Reduce noise
                    stop=["Human:", "Assistant:", "\n\nQuestion:", "\n\nAnswer:"]
                )
                print(f"✅ {model_name} initialized successfully")
                
            except Exception as e:
                print(f"⚠️ Primary model failed: {e}")
                print("🔄 Trying fallback model...")
                
                # Try fallback model
                try:
                    self.llm = LlamaCpp(
                        model_path=FALLBACK_MODEL_PATH,
                        temperature=0.1,
                        max_tokens=512,       # Smaller model, smaller context
                        n_ctx=2048,
                        verbose=False,
                        stop=["Human:", "Assistant:", "\n\nQuestion:", "\n\nAnswer:"]
                    )
                    model_name = "TinyLlama 1.1B"
                    print(f"✅ {model_name} (fallback) initialized successfully")
                    
                except Exception as fallback_error:
                    error_msg = f"Failed to initialize both primary and fallback models. Primary: {e}, Fallback: {fallback_error}"
                    print(f"ERROR: {error_msg}")
                    initialization_error = error_msg
                    return False
            
            total_time = time.time() - start_time
            print(f"🎉 RAG system initialized successfully in {total_time:.2f} seconds!")
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Initialization error: {e}"
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            initialization_error = error_msg
            return False
    
    def smart_search(self, query, top_k=4):
        """Intelligent document search using multiple strategies"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored_docs = []
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            doc_words = set(re.findall(r'\b\w+\b', doc_lower))
            
            score = 0
            
            # Exact phrase matching (highest weight)
            if query_lower in doc_lower:
                score += 100
            
            # Keyword matching
            common_words = query_words.intersection(doc_words)
            score += len(common_words) * 10
            
            # Partial keyword matching
            for query_word in query_words:
                for doc_word in doc_words:
                    if query_word in doc_word or doc_word in query_word:
                        score += 5
            
            # Project management specific terms boost
            pm_terms = ['project', 'management', 'risk', 'scope', 'schedule', 
                       'cost', 'quality', 'stakeholder', 'communication', 'procurement']
            for term in pm_terms:
                if term in query_lower and term in doc_lower:
                    score += 15
            
            # Book title relevance boost
            book_title = self.doc_metadata[i]['book_title'].lower()
            for query_word in query_words:
                if query_word in book_title:
                    score += 20
            
            if score > 0:
                scored_docs.append((score, i, doc, self.doc_metadata[i]))
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [(doc, metadata) for score, idx, doc, metadata in scored_docs[:top_k]]
    
    def query(self, query_text):
        """Process a query through the RAG system"""
        print(f"🔍 Processing query: {query_text}")
        
        if not self.llm:
            return {
                "error": "System not initialized",
                "response": "The system failed to initialize. Please check the logs for details."
            }
        
        try:
            # Log query
            self.auditor.log_query(query_text)
            
            # Find relevant documents
            relevant_docs = self.smart_search(query_text, top_k=3)
            print(f"📖 Found {len(relevant_docs)} relevant documents")
            
            # Log retrieval
            self.auditor.log_retrieval(query_text, [doc for doc, meta in relevant_docs])
            
            if not relevant_docs:
                return {
                    "query": query_text,
                    "response": "I couldn't find relevant information in the project management knowledge base to answer your question. Please try rephrasing your question or asking about topics like project phases, risk management, stakeholder management, etc.",
                    "runtime_seconds": 0,
                    "model": "Llama 3.2 3B",
                    "sources_used": []
                }
            
            # Create context from relevant docs
            context_parts = []
            for i, (doc, metadata) in enumerate(relevant_docs):
                context_parts.append(f"Document {i+1} (from {metadata['book_title']}):\n{doc[:800]}...")
            
            context = "\n\n".join(context_parts)
            
            # Create focused prompt
            prompt = f"""You are a project management expert. Answer the question based on the provided context from project management materials.

Context:
{context}

Question: {query_text}

Instructions:
- Answer based only on the provided context
- Be specific and practical
- Use bullet points for lists when appropriate
- If the context doesn't contain enough information, say so

Answer:"""
            
            # Generate response
            print("🤖 Generating response...")
            start_time = time.time()
            
            try:
                response = self.llm(prompt)
                runtime = time.time() - start_time
                
                # Clean and format response
                response = response.strip()
                
                # Remove any prompt artifacts
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                
                print(f"✅ Response generated in {runtime:.2f} seconds")
                
                # Log generation
                self.auditor.log_generation(query_text, response, runtime)
                
                # Prepare enhanced sources info
                sources_info = []
                for doc, metadata in relevant_docs:
                    # Create a more meaningful content excerpt
                    lines = doc.split('\n')
                    meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]
                    
                    # Take first few meaningful lines for display
                    display_content = '\n'.join(meaningful_lines[:3])
                    if len(display_content) > 400:
                        display_content = display_content[:400] + "..."
                    
                    sources_info.append({
                        "book_title": metadata['book_title'],
                        "section_title": metadata.get('section_title', 'General Content'),
                        "content_preview": display_content,
                        "content_length": metadata['length'],
                        "chunk_id": metadata['chunk_id']
                    })
                
                return {
                    "query": query_text,
                    "response": response,
                    "runtime_seconds": runtime,
                    "model": "Llama 3.2 3B Instruct",
                    "sources_used": sources_info
                }
                
            except Exception as e:
                print(f"❌ LLM generation failed: {e}")
                return {
                    "error": f"Response generation failed: {e}",
                    "response": "I encountered an error while generating a response. This might be due to model loading issues or resource constraints."
                }
                
        except Exception as e:
            import traceback
            print(f"❌ Query processing error: {e}")
            print(traceback.format_exc())
            return {
                "error": str(e),
                "response": "An unexpected error occurred while processing your query."
            }
    
    def get_audit_report(self):
        """Get comprehensive audit report"""
        try:
            report = self.auditor.generate_audit_report()
            report["system_info"] = {
                "documents_loaded": len(self.documents),
                "books_loaded": len(set([m['book_title'] for m in self.doc_metadata])),
                "model_available": self.llm is not None,
                "initialization_error": initialization_error
            }
            return report
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate audit report"
            }

def initialize_system():
    """Initialize the RAG system in a separate thread"""
    global is_initialized, rag_system, initialization_error
    
    print("🔄 Starting RAG system initialization...")
    rag_system = ProjectManagementRAG()
    
    if rag_system.initialize():
        is_initialized = True
        print("✅ RAG system initialization completed successfully!")
    else:
        print(f"❌ RAG system initialization failed: {initialization_error}")

# Start initialization thread
threading.Thread(target=initialize_system, daemon=True).start()

# Flask routes
@app.route('/')
def home():
    """Serve the main HTML interface"""
    try:
        return send_file('rag-interface.html')
    except Exception as e:
        return jsonify({
            "error": f"Could not load interface: {e}",
            "status": "Project Management RAG API", 
            "initialized": is_initialized,
            "model": "Llama 3.2 3B Instruct",
            "llamacpp_available": LLAMACPP_AVAILABLE
        })

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        "status": "Project Management RAG API", 
        "initialized": is_initialized,
        "model": "Llama 3.2 3B Instruct",
        "error": initialization_error,
        "llamacpp_available": LLAMACPP_AVAILABLE
    })

@app.route('/api/status')
def check_status():
    global is_initialized, initialization_error
    
    if initialization_error:
        status = "Failed"
        progress = 0
        message = f"Initialization failed: {initialization_error}"
    elif is_initialized:
        status = "Complete"
        progress = 100
        message = "RAG system ready for queries"
    else:
        status = "Initializing"
        progress = 50
        message = "Loading Llama 3.2 3B model and documents..."
    
    return jsonify({
        "status": status,
        "progress": progress,
        "message": message,
        "model": "Llama 3.2 3B Instruct"
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    global is_initialized, rag_system, initialization_error
    
    print("📨 Query API called")
    
    if initialization_error:
        return jsonify({
            "response": f"System initialization failed: {initialization_error}",
            "error": "System not available"
        })
    
    if not is_initialized or rag_system is None:
        return jsonify({
            "response": "The system is still initializing. Please wait a moment and try again.",
            "error": "System not ready"
        })
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        print(f"📝 Processing query: {query}")
        result = rag_system.query(query)
        print("✅ Query processed successfully")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"❌ Query processing error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "response": "Failed to process your query due to an unexpected error."
        })

@app.route('/api/audit', methods=['GET'])
def get_audit():
    global is_initialized, rag_system
    
    if not is_initialized or rag_system is None:
        return jsonify({
            "error": "System not initialized",
            "message": "Please wait for system initialization to complete."
        })
    
    try:
        report = rag_system.get_audit_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to generate audit report"
        })

if __name__ == '__main__':
    print("🚀 Starting Project Management RAG Server")
    print(f"📂 Primary model: {MODEL_PATH}")
    print(f"📂 Fallback model: {FALLBACK_MODEL_PATH}")
    print(f"📂 Data path: {DATA_PATH}")
    print(f"🌐 Server will run at: http://localhost:8081")
    print("-" * 50)
    
    app.run(debug=False, port=8081, host='127.0.0.1', use_reloader=False)