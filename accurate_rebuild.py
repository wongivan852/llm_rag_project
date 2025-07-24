import os
import shutil
import re
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def detect_book_source(content):
    """Detect book source based on actual PMI book collection"""
    content_lower = content.lower()
    
    # Your actual book patterns
    book_patterns = {
        "PMBOK Guide 7th Edition (2021)": [
            "pmbok", "project management body of knowledge", "seventh edition", 
            "guide seventh", "pmbok guide"
        ],
        "Practice Standard for Project Configuration Management (2007)": [
            "configuration management", "practice standard", "2007"
        ],
        "Managing Change in Organizations Practice Guide (2013)": [
            "managing change", "change in organizations", "organizational change"
        ],
        "Navigating Complexity Practice Guide (2014)": [
            "navigating complexity", "complexity practice", "complex projects"
        ],
        "Governance Practice Guide (2016)": [
            "governance", "portfolios programs projects", "governance practice"
        ],
        "Requirements Management Practice Guide (2016)": [
            "requirements management", "requirements practice", "requirements analysis"
        ],
        "PMI Guide to Business Analysis (2017)": [
            "business analysis", "pmi guide business", "business analyst"
        ],
        "Standard for Portfolio Management 4th Edition (2017)": [
            "portfolio management", "standard portfolio", "fourth edition"
        ],
        "Standard for Organizational Project Management (2018)": [
            "organizational project management", "opm"
        ],
        "Benefits Realization Management Practice Guide (2019)": [
            "benefits realization", "benefits management", "value realization"
        ],
        "Practice Standard for Project Estimating 2nd Edition (2019)": [
            "project estimating", "estimating practice", "cost estimation"
        ],
        "Practice Standard for Scheduling 3rd Edition (2019)": [
            "scheduling practice", "project scheduling", "schedule development"
        ],
        "Practice Standard Work Breakdown Structures 3rd Edition (2019)": [
            "work breakdown structures", "wbs", "work breakdown"
        ],
        "Standard for Earned Value Management (2019)": [
            "earned value management", "evm", "earned value"
        ],
        "Standard for Risk Management (2019)": [
            "risk management", "risk standard", "risk assessment"
        ],
        "Choose Your WoW Disciplined Agile 2nd Edition (2022)": [
            "disciplined agile", "choose your wow", "agile approach", "way of working"
        ],
        "Process Groups Practice Guide (2023)": [
            "process groups", "initiating", "planning", "executing", "monitoring", "closing"
        ],
        "AI Essentials for Project Professionals (2024)": [
            "ai essentials", "artificial intelligence", "ai project", "machine learning"
        ],
        "Business Analysis for Practitioners 2nd Edition (2024)": [
            "business analysis practitioners", "business analysis practice"
        ],
        "Risk Management Practice Guide (2024)": [
            "risk management practice", "risk guide 2024"
        ],
        "Standard for Program Management 5th Edition (2024)": [
            "program management", "standard program", "fifth edition"
        ],
        "Leading AI Transformation (2025)": [
            "ai transformation", "leading ai", "artificial intelligence transformation"
        ],
        "Project Management Offices Practice Guide (2025)": [
            "project management offices", "pmo", "pmo practice"
        ]
    }
    
    # Score each book
    book_scores = {}
    for book_name, keywords in book_patterns.items():
        score = 0
        for keyword in keywords:
            if keyword in content_lower:
                score += 1
                # Extra weight for unique identifiers
                if keyword in ["pmbok", "evm", "wbs", "opm", "pmo"]:
                    score += 2
        book_scores[book_name] = score
    
    # Return best match
    if book_scores:
        best_match = max(book_scores.items(), key=lambda x: x[1])
        if best_match[1] > 0:
            return best_match[0]
    
    return "PMBOK Guide 7th Edition (2021)"  # Default

def rebuild_with_accurate_sources():
    print("Rebuilding with accurate PMI book detection...")
    
    # Use original file
    SOURCE_FILE = "data/pmp_combined.txt"
    VECTOR_STORE_PATH = "data/vector_store"
    BACKUP_PATH = f"data/vector_store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found!")
        return False
    
    print(f"Using: {SOURCE_FILE} ({os.path.getsize(SOURCE_FILE):,} bytes)")
    
    # Backup
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.copytree(VECTOR_STORE_PATH, BACKUP_PATH)
        print(f"Backup: {BACKUP_PATH}")
    
    # Load and split
    print("Loading document...")
    loader = TextLoader(SOURCE_FILE, encoding='utf-8')
    documents = loader.load()
    
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Detect sources
    print("Detecting sources...")
    enhanced_chunks = []
    source_counts = {}
    
    for i, chunk in enumerate(chunks):
        source = detect_book_source(chunk.page_content)
        source_counts[source] = source_counts.get(source, 0) + 1
        
        chunk.metadata.update({
            'source': source,
            'book': source,
            'chunk_id': i
        })
        enhanced_chunks.append(chunk)
        
        if i % 200 == 0:
            print(f"  Processed {i}/{len(chunks)}...")
    
    print(f"Enhanced {len(enhanced_chunks)} chunks")
    print("\nDetected sources:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} chunks")
    
    # Create vector store
    print("\nCreating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    
    vector_store = FAISS.from_documents(enhanced_chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    # Test
    print("\nTesting...")
    results = vector_store.similarity_search("project management", k=3)
    for i, result in enumerate(results):
        source = result.metadata.get('source', 'Unknown')
        preview = result.page_content[:80]
        print(f"  Test {i+1}: {source} - {preview}...")
    
    print("\nSUCCESS! Your sources should now show specific PMI book names.")
    print("Restart your server: python3 app.py")
    
    return True

if __name__ == "__main__":
    rebuild_with_accurate_sources()
