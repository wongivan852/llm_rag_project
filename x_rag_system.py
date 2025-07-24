import os
import json
import time
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Paths configuration
DATA_PATH = "data/pmp_combined.txt"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VECTOR_STORE_PATH = "data/vector_store"
LOGS_PATH = "logs"

# Ensure logs directory exists
os.makedirs(LOGS_PATH, exist_ok=True)

class KnowledgeAuditor:
    """Tracks and audits knowledge utilization from the RAG pipeline"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOGS_PATH, f"audit_{self.session_id}.json")
        self.audit_records = []
        
    def log_query(self, query):
        """Log the initial query"""
        record = {
            "timestamp": time.time(),
            "event_type": "query",
            "query": query
        }
        self.audit_records.append(record)
        self._save_logs()
        return record
        
    def log_retrieval(self, query, docs):
        """Log the retrieved documents"""
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
        """Log the generated response"""
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
        """Generate a summary report of knowledge utilization"""
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
        """Save audit records to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.audit_records, f, indent=2)


class RAGSystem:
    """Implements a RAG system with knowledge auditing"""
    
    def __init__(self):
        self.auditor = KnowledgeAuditor()
        self.vector_store = None
        self.qa_chain = None
        
    def initialize(self):
        """Initialize the RAG system and components"""
        print("Initializing RAG system...")
        
        # 1. Load document
        print("Loading documents...")
        loader = TextLoader(DATA_PATH)
        documents = loader.load()
        
        # 2. Split document into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better retrieval
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from the document")
        
        # 3. Create embeddings and vector store
        print("Creating embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if vector store exists, otherwise create it
        if os.path.exists(VECTOR_STORE_PATH):
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings
            )
        else:
            print("Creating new vector store...")
            self.vector_store = FAISS.from_documents(
                chunks, 
                embeddings
            )
            # Save the vector store for future use
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            self.vector_store.save_local(VECTOR_STORE_PATH)
        
        # 4. Initialize LLM
        print("Initializing LLM...")
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.2,
            max_tokens=2048,
            n_ctx=4096,
            verbose=True
        )
        
        # 5. Create a retriever with auditing
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve 5 most relevant chunks
        )
        
        # Create a custom retriever with auditing
        def audited_retriever(query):
            docs = retriever.get_relevant_documents(query)
            return self.auditor.log_retrieval(query, docs)
        
        # 6. Create RAG chain with custom prompt
        template = """
        You are an AI assistant specialized in project management. 
        Use ONLY the following context to answer the question. 
        If you don't know the answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=audited_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RAG system initialized successfully!")
    
    def query(self, query_text):
        """Process a query through the RAG system with auditing"""
        if not self.qa_chain:
            print("RAG system not initialized! Call initialize() first.")
            return None
        
        # Log the query
        self.auditor.log_query(query_text)
        
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
    
    def get_audit_report(self):
        """Get an audit report of knowledge utilization"""
        return self.auditor.generate_audit_report()


# Usage example
if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.initialize()
    
    # Test query
    query = "What are the key aspects of risk management in project management?"
    result = rag_system.query(query)
    
    print("\n" + "="*50)
    print("QUERY:", result["query"])
    print("-"*50)
    print("RESPONSE:", result["response"])
    print("-"*50)
    print(f"Response time: {result['runtime_seconds']:.2f} seconds")
    print("="*50)
    
    # Print source documents
    print("\nSOURCES USED:")
    for i, source in enumerate(result["sources_used"]):
        print(f"\nSource {i+1}:")
        print(f"Content: {source['content'][:150]}...")
        if "source" in source["metadata"]:
            print(f"Source: {source['metadata']['source']}")
    
    # Generate and print audit report
    print("\n" + "="*50)
    print("KNOWLEDGE UTILIZATION AUDIT REPORT:")
    report = rag_system.get_audit_report()
    for key, value in report.items():
        if key != "most_used_chunks" and key != "detailed_logs":
            print(f"{key}: {value}")
    
    print("\nMost frequently used chunks:")
    for i, (chunk, count) in enumerate(report["most_used_chunks"]):
        print(f"{i+1}. Used {count} times: {chunk}")
    
    print(f"\nDetailed logs saved to: {report['detailed_logs']}")