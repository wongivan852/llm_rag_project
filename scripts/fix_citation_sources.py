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

def extract_clean_source_from_citation(content):
    """Extract clean book name from citation-heavy content"""
    content_lower = content.lower()
    
    # Look for specific PMI publications in citations
    if 'practice standard for work breakdown structures' in content_lower:
        if 'third edition' in content_lower:
            return "Practice Standard Work Breakdown Structures 3rd Edition (2019)"
        else:
            return "Practice Standard Work Breakdown Structures"
    
    elif 'pmbok guide' in content_lower or 'project management body of knowledge' in content_lower:
        if 'seventh edition' in content_lower:
            return "PMBOK Guide 7th Edition (2021)"
        elif 'sixth edition' in content_lower:
            return "PMBOK Guide 6th Edition (2017)"
        elif 'fifth edition' in content_lower:
            return "PMBOK Guide 5th Edition (2013)"
        else:
            return "PMBOK Guide"
    
    elif 'earned value management' in content_lower and ('standard' in content_lower or 'evm' in content_lower):
        return "Standard for Earned Value Management (2019)"
    
    elif 'practice standard for scheduling' in content_lower or 'scheduling practice' in content_lower:
        return "Practice Standard for Scheduling 3rd Edition (2019)"
    
    elif 'practice standard for project estimating' in content_lower or 'estimating practice' in content_lower:
        return "Practice Standard for Project Estimating 2nd Edition (2019)"
    
    elif 'risk management' in content_lower and ('standard' in content_lower or 'practice guide' in content_lower):
        if '2024' in content_lower:
            return "Risk Management Practice Guide (2024)"
        else:
            return "Standard for Risk Management (2019)"
    
    elif 'business analysis' in content_lower:
        if 'pmi guide' in content_lower:
            return "PMI Guide to Business Analysis (2017)"
        else:
            return "Business Analysis for Practitioners 2nd Edition (2024)"
    
    elif 'requirements management' in content_lower:
        return "Requirements Management Practice Guide (2016)"
    
    elif 'portfolio management' in content_lower and 'standard' in content_lower:
        return "Standard for Portfolio Management 4th Edition (2017)"
    
    elif 'program management' in content_lower and 'standard' in content_lower:
        return "Standard for Program Management 5th Edition (2024)"
    
    elif 'benefits realization' in content_lower:
        return "Benefits Realization Management Practice Guide (2019)"
    
    elif 'organizational project management' in content_lower:
        return "Standard for Organizational Project Management (2018)"
    
    elif 'governance' in content_lower and ('portfolio' in content_lower or 'program' in content_lower):
        return "Governance Practice Guide (2016)"
    
    elif 'configuration management' in content_lower:
        return "Practice Standard for Project Configuration Management (2007)"
    
    elif 'managing change' in content_lower or 'change management' in content_lower:
        return "Managing Change in Organizations Practice Guide (2013)"
    
    elif 'navigating complexity' in content_lower:
        return "Navigating Complexity Practice Guide (2014)"
    
    elif 'process groups' in content_lower:
        return "Process Groups Practice Guide (2023)"
    
    elif 'disciplined agile' in content_lower or 'choose your wow' in content_lower:
        return "Choose Your WoW Disciplined Agile 2nd Edition (2022)"
    
    elif 'ai essentials' in content_lower or ('artificial intelligence' in content_lower and 'project professionals' in content_lower):
        return "AI Essentials for Project Professionals (2024)"
    
    elif 'ai transformation' in content_lower:
        return "Leading AI Transformation (2025)"
    
    elif 'project management offices' in content_lower or 'pmo' in content_lower:
        return "Project Management Offices Practice Guide (2025)"
    
    # Handle external references (like NASA)
    elif 'nasa' in content_lower and 'wbs' in content_lower:
        return "NASA WBS Handbook (External Reference)"
    
    elif 'iso' in content_lower:
        return "ISO Standards (External Reference)"
    
    elif 'ieee' in content_lower:
        return "IEEE Standards (External Reference)"
    
    # General categorization for unidentified content
    elif any(term in content_lower for term in ['wbs', 'work breakdown']):
        return "Work Breakdown Structures References"
    
    elif 'agile' in content_lower or 'scrum' in content_lower:
        return "Agile Practice References"
    
    elif 'risk' in content_lower:
        return "Risk Management References"
    
    elif 'quality' in content_lower:
        return "Quality Management References"
    
    elif 'stakeholder' in content_lower:
        return "Stakeholder Management References"
    
    else:
        return "PMI Project Management Standards"

def rebuild_with_citation_aware_sources():
    """Rebuild vector store with citation-aware source detection"""
    
    print("=== REBUILDING WITH CITATION-AWARE SOURCE DETECTION ===")
    
    # Find the current data file
    data_files = []
    for file in os.listdir('data'):
        if file.endswith('.txt') and os.path.getsize(f'data/{file}') > 1000000:  # > 1MB
            data_files.append((file, os.path.getsize(f'data/{file}')))
    
    if not data_files:
        print("❌ No suitable data file found!")
        return False
    
    # Use the largest file
    source_file = max(data_files, key=lambda x: x[1])[0]
    source_path = f'data/{source_file}'
    
    print(f"✅ Using file: {source_path}")
    print(f"   Size: {os.path.getsize(source_path):,} bytes")
    
    # Backup existing vector store
    if os.path.exists('data/vector_store'):
        backup_path = f"data/vector_store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move('data/vector_store', backup_path)
        print(f"✅ Backed up vector store to: {backup_path}")
    
    # Load document
    print("\n📖 Loading document...")
    loader = TextLoader(source_path, encoding='utf-8')
    documents = loader.load()
    
    # Split with appropriate chunk size
    print("✂️  Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Slightly larger for better context
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Process chunks with citation-aware source detection
    print("\n🔍 Applying citation-aware source detection...")
    enhanced_chunks = []
    source_counts = {}
    
    for i, chunk in enumerate(chunks):
        # Extract clean source name
        clean_source = extract_clean_source_from_citation(chunk.page_content)
        
        # Count sources
        source_counts[clean_source] = source_counts.get(clean_source, 0) + 1
        
        # Update metadata
        chunk.metadata.update({
            'source': clean_source,
            'book': clean_source,
            'chunk_id': i,
            'total_chunks': len(chunks)
        })
        
        enhanced_chunks.append(chunk)
        
        if i % 200 == 0 and i > 0:
            print(f"   Processed {i}/{len(chunks)} chunks...")
    
    print(f"✅ Enhanced {len(enhanced_chunks)} chunks")
    
    # Show source distribution
    print(f"\n📊 Clean source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(enhanced_chunks)) * 100
        print(f"   {source}: {count} chunks ({percentage:.1f}%)")
    
    # Create vector store
    print(f"\n🔮 Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(enhanced_chunks, embeddings)
    vector_store.save_local('data/vector_store')
    
    print(f"✅ Vector store created!")
    
    # Test the results
    print(f"\n🧪 Testing improved source detection...")
    test_queries = [
        "work breakdown structures",
        "earned value management", 
        "project scheduling",
        "risk management"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        for j, result in enumerate(results):
            source = result.metadata.get('source', 'Unknown')
            preview = result.page_content[:100].replace('\n', ' ')
            print(f"     {j+1}. {source}")
            print(f"        {preview}...")
    
    print(f"\n🎉 SUCCESS!")
    print(f"   Sources should now show clean book names instead of citations")
    print(f"   Restart your server: python3 app.py")
    
    return True

if __name__ == "__main__":
    rebuild_with_citation_aware_sources()
