from flask import Flask, request, jsonify, render_template_string
import os
import json
import time
import threading
from datetime import datetime
from typing import List, Any
from flask_cors import CORS  # Add this import

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
DATA_PATH = "data/pmp_combined.txt"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VECTOR_STORE_PATH = "data/vector_store"
LOGS_PATH = "logs"

# Ensure directories exist
os.makedirs(LOGS_PATH, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
is_initialized = False
rag_system = None

class KnowledgeAuditor:
    """Tracks and audits knowledge utilization from the RAG pipeline"""
    
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
        retrieved_chunks = []
        for i, doc in enumerate(docs):
            chunk_info = {
                "chunk_id": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": getattr(doc, "score", None)
            }
            retrieved_chunks.append(chunk_info)
            
        record = {
            "timestamp": time.time(),
            "event_type": "retrieval",
            "query": query,
            "chunks_retrieved": len(docs),
            "retrieved_chunks": retrieved_chunks
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
        
        total_chunks = sum(r["chunks_retrieved"] for r in retrievals) if retrievals else 0
        
        # Count which parts of the knowledge base were utilized
        chunk_utilization = {}
        for retrieval in retrievals:
            for chunk in retrieval["retrieved_chunks"]:
                chunk_content = chunk["content"][:100] + "..."  # Use first 100 chars as identifier
                if chunk_content in chunk_utilization:
                    chunk_utilization[chunk_content] += 1
                else:
                    chunk_utilization[chunk_content] = 1
        
        # Find most frequently used chunks
        most_used_chunks = sorted(
            chunk_utilization.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        report = {
            "session_id": self.session_id,
            "total_queries": len(queries),
            "total_retrievals": len(retrievals),
            "total_generations": len(generations),
            "total_chunks_retrieved": total_chunks,
            "avg_chunks_per_query": total_chunks / len(retrievals) if retrievals else 0,
            "most_used_chunks": most_used_chunks,
            "detailed_logs": self.log_file
        }
        
        return report
    
    def _save_logs(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.audit_records, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save logs: {e}")

class RAGSystem:
    """Implements a RAG system with knowledge auditing"""
    
    def __init__(self):
        self.auditor = KnowledgeAuditor()
        self.vector_store = None
        self.qa_chain = None
        self.retriever = None
    
    def initialize(self):
        try:
            print("Initializing RAG system...")
            start_time = time.time()
            
            # 1. Load existing vector store
            if os.path.exists(VECTOR_STORE_PATH) and os.path.isfile(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
                print("Loading existing vector store...")
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Load the vector store
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
            else:
                raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Please run the full initialization process.")
            
            # 2. Initialize LLM
            print("Initializing LLM...")
            llm_start = time.time()
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            # Initialize LLM with optimized settings for M1
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                temperature=0.2,
                max_tokens=2048,
                n_ctx=4096,
                n_batch=512,  # Increased for better performance
                verbose=True
            )
            print(f"LLM initialized in {time.time() - llm_start:.2f} seconds")
            
            # 3. Set up retriever and QA chain
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve 5 most relevant chunks
            )
            
            # Create prompt template with better formatting instructions
            template = """
            You are an AI assistant specialized in project management. 
            Use ONLY the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information to answer this question."

            Format your response in a clear, structured way:
            - Use bullet points for lists and multiple points
            - Organize related information under clear headings
            - For steps or processes, use numbered lists
            - Ensure key terms or important concepts are emphasized

            Context:
            {context}

            Question: {question}

            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create the chain - using the original retriever
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,  # Use the original retriever
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            total_time = time.time() - start_time
            print(f"RAG system initialized successfully in {total_time:.2f} seconds!")
            
            return True
            
        except Exception as e:
            import traceback
            print(f"Error initializing RAG system: {e}")
            print(traceback.format_exc())
            return False
    
    def query(self, query_text):
        """Process a query through the RAG system with auditing"""
        if not self.qa_chain or not self.retriever:
            return {
                "error": "System not initialized",
                "response": "The system is not initialized. Please refresh the page and try again."
            }
        
        try:
            # Log the query
            self.auditor.log_query(query_text)
            
            # Get relevant documents directly using the stored retriever
            docs = self.retriever.get_relevant_documents(query_text)
            
            # Log retrieval
            self.auditor.log_retrieval(query_text, docs)
            
            # Measure response time
            start_time = time.time()
            
            # Run the query
            result = self.qa_chain({"query": query_text})
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Extract response and source documents
            response = result['result']
            source_docs = result.get('source_documents', [])
            
            # Log the generation
            self.auditor.log_generation(query_text, response, runtime)
            
            # Return the response with metadata
            return {
                "query": query_text,
                "response": response,
                "runtime_seconds": runtime,
                "sources_used": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in source_docs
                ]
            }
        except Exception as e:
            import traceback
            print(f"Error processing query: {e}")
            print(traceback.format_exc())
            return {
                "error": str(e),
                "response": "An error occurred while processing your query. Please try again."
            }
    
    def get_audit_report(self):
        """Get an audit report of knowledge utilization"""
        try:
            return self.auditor.generate_audit_report()
        except Exception as e:
            print(f"Error generating audit report: {e}")
            return {
                "error": str(e),
                "message": "Failed to generate audit report"
            }

def initialize_system():
    """Initialize the system in background"""
    global is_initialized, rag_system
    
    print("Starting initialization thread")
    rag_system = RAGSystem()
    if rag_system.initialize():
        is_initialized = True
        print("Initialization thread completed successfully")
    else:
        print("Initialization thread failed")

@app.route('/api/status')
def check_status():
    return jsonify({
        "status": "Complete" if is_initialized else "In progress",
        "progress": 100 if is_initialized else 50,
        "message": "System ready for queries!" if is_initialized else "System is initializing..."
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    global is_initialized, rag_system
    
    if not is_initialized or rag_system is None:
        return jsonify({
            "response": "The system is still initializing. Please try again later.",
            "error": "System not initialized"
        })
    
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = rag_system.query(query)
    return jsonify(result)

@app.route('/api/audit', methods=['GET'])
def get_audit():
    global is_initialized, rag_system
    
    if not is_initialized or rag_system is None:
        return jsonify({
            "error": "System not initialized",
            "message": "The system is still initializing. Please try again later."
        })
    
    report = rag_system.get_audit_report()
    return jsonify(report)

# Start initialization when the server starts
threading.Thread(target=initialize_system).start()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8081)