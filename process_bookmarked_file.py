import os
import re
import shutil
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def process_bookmarked_file():
    print("=== PROCESSING BOOKMARKED PMI FILE ===")
    
    # Read the bookmarked file
    with open('data/pmp_combined.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by book markers
    book_sections = re.split(r'\[BOOK: ([^\]]+)\]', content)
    
    if len(book_sections) < 3:
        print("❌ No book markers found!")
        print("Please add [BOOK: Book Name] markers to your file first.")
        return False
    
    print(f"✅ Found {(len(book_sections)-1)//2} book sections")
    
    # Process each book section
    all_chunks = []
    source_counts = {}
    
    for i in range(1, len(book_sections), 2):
        book_name = book_sections[i].strip()
        book_content = book_sections[i+1].strip()
        
        if len(book_content) < 200:  # Skip very short sections
            print(f"⚠️  Skipping short section: {book_name}")
            continue
        
        print(f"📚 Processing: {book_name}")
        print(f"   Content length: {len(book_content):,} characters")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        
        doc = Document(page_content=book_content)
        chunks = text_splitter.split_documents([doc])
        
        # Add metadata to each chunk
        for j, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source': book_name,
                'book': book_name,
                'chunk_id': len(all_chunks),
                'book_chunk_id': j
            })
            all_chunks.append(chunk)
        
        source_counts[book_name] = len(chunks)
        print(f"   Created {len(chunks)} chunks")
    
    print(f"\n📊 Final source distribution:")
    total_chunks = len(all_chunks)
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_chunks) * 100
        print(f"   {source}: {count} chunks ({percentage:.1f}%)")
    
    # Create vector store
    print(f"\n🔮 Creating vector store with {total_chunks} chunks...")
    
    # Backup existing vector store
    if os.path.exists('data/vector_store'):
        backup_path = f"data/vector_store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move('data/vector_store', backup_path)
        print(f"   Backed up existing store to: {backup_path}")
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local('data/vector_store')
    
    print(f"✅ Vector store created successfully!")
    
    # Test the new vector store
    print(f"\n🧪 Testing vector store...")
    test_queries = [
        "project management process groups",
        "earned value management",
        "work breakdown structures", 
        "risk assessment",
        "business analysis"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = vector_store.similarity_search(query, k=2)
            for idx, result in enumerate(results):
                source = result.metadata.get('source', 'Unknown')
                preview = result.page_content[:80].replace('\n', ' ')
                print(f"     {idx+1}. {source}")
                print(f"        {preview}...")
        except Exception as e:
            print(f"     ❌ Test failed: {e}")
    
    print(f"\n🎉 SUCCESS!")
    print(f"   • {total_chunks} chunks created")
    print(f"   • {len(source_counts)} books processed")
    print(f"   • Clean source attribution ready")
    print(f"\n🚀 Next step: Restart your server with 'python3 app.py'")
    
    return True

if __name__ == "__main__":
    process_bookmarked_file()
