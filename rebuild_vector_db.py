import os
import shutil
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def rebuild_vector_database():
    print("Starting vector database rebuild...")
    
    # Configuration
    ENHANCED_FILE = "data/pmp_enhanced.txt"
    VECTOR_STORE_PATH = "data/vector_store"
    BACKUP_PATH = f"data/vector_store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check enhanced file exists
    if not os.path.exists(ENHANCED_FILE):
        print(f"Error: {ENHANCED_FILE} not found!")
        print("Please create the enhanced file first.")
        return False
    
    print(f"Found enhanced file: {os.path.getsize(ENHANCED_FILE)} bytes")
    
    # Backup existing vector store
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            shutil.copytree(VECTOR_STORE_PATH, BACKUP_PATH)
            print(f"Backup created: {BACKUP_PATH}")
        except Exception as e:
            print(f"Warning: Could not backup: {e}")
    
    # Load document
    print("Loading enhanced document...")
    try:
        loader = TextLoader(ENHANCED_FILE, encoding='utf-8')
        documents = loader.load()
        print(f"Loaded {len(documents[0].page_content)} characters")
    except Exception as e:
        print(f"Error loading document: {e}")
        return False
    
    # Split documents
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n[SOURCE:", "\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Enhance chunks with source metadata
    print("Adding source metadata...")
    enhanced_chunks = []
    source_counts = {}
    
    for i, chunk in enumerate(chunks):
        content_lower = chunk.page_content.lower()
        
        # Detect source based on content
        if 'pmbok' in content_lower or 'project management body' in content_lower:
            source = 'PMBOK Guide'
        elif 'agile' in content_lower or 'scrum' in content_lower:
            source = 'Agile Practice Guide'
        elif 'risk management' in content_lower:
            source = 'Risk Management Guide'
        elif 'quality' in content_lower and 'management' in content_lower:
            source = 'Quality Management Guide'
        elif 'stakeholder' in content_lower:
            source = 'Stakeholder Management Guide'
        elif 'schedule' in content_lower or 'time management' in content_lower:
            source = 'Schedule Management Guide'
        elif 'cost' in content_lower or 'budget' in content_lower:
            source = 'Cost Management Guide'
        elif 'procurement' in content_lower:
            source = 'Procurement Management Guide'
        else:
            source = 'Project Management Guide'
        
        # Count sources
        source_counts[source] = source_counts.get(source, 0) + 1
        
        # Clean content (remove source markers)
        clean_content = chunk.page_content
        while "[SOURCE:" in clean_content:
            start = clean_content.find("[SOURCE:")
            end = clean_content.find("]", start)
            if end > start:
                clean_content = clean_content[:start] + clean_content[end+1:]
            else:
                break
        
        # Update chunk
        chunk.page_content = clean_content.strip()
        chunk.metadata.update({
            'source': source,
            'book': source,
            'chunk_id': i,
            'total_chunks': len(chunks)
        })
        enhanced_chunks.append(chunk)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(chunks)} chunks...")
    
    print(f"Enhanced {len(enhanced_chunks)} chunks")
    print("Source distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False
    
    # Create vector store
    print("Building vector store...")
    try:
        # Remove old vector store
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        
        # Create new vector store
        vector_store = FAISS.from_documents(enhanced_chunks, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        
        print(f"Vector store saved to: {VECTOR_STORE_PATH}")
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Restore backup if available
        if os.path.exists(BACKUP_PATH):
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            shutil.copytree(BACKUP_PATH, VECTOR_STORE_PATH)
            print("Restored backup")
        return False
    
    # Test the vector store
    print("Testing vector store...")
    try:
        test_results = vector_store.similarity_search("project management", k=2)
        for i, result in enumerate(test_results):
            source = result.metadata.get('source', 'Unknown')
            preview = result.page_content[:80].replace('\n', ' ')
            print(f"  Test {i+1}: {source} - {preview}...")
    except Exception as e:
        print(f"Warning: Test failed: {e}")
    
    print("\nSUCCESS! Vector database rebuilt with enhanced sources.")
    return True

if __name__ == "__main__":
    success = rebuild_vector_database()
    if success:
        print("\nNext step: Restart your server with 'python3 app.py'")
    else:
        print("\nRebuild failed. Please check the errors above.")
