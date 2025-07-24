import re
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

def extract_clean_book_name(content):
    """Extract clean book name from content"""
    content_lower = content.lower()
    
    # Look for PMBOK patterns
    if re.search(r'pmbok.*guide.*seventh.*edition', content_lower):
        return "PMBOK Guide 7th Edition"
    elif re.search(r'pmbok.*guide.*sixth.*edition', content_lower):
        return "PMBOK Guide 6th Edition"
    elif re.search(r'pmbok.*guide.*fifth.*edition', content_lower):
        return "PMBOK Guide 5th Edition"
    elif 'pmbok' in content_lower or 'project management body of knowledge' in content_lower:
        return "PMBOK Guide"
    
    # Look for specific practice guides and standards
    if 'earned value management' in content_lower or 'evm' in content_lower:
        return "Earned Value Management Standard"
    elif 'work breakdown structures' in content_lower or 'wbs' in content_lower:
        return "Work Breakdown Structures Standard"
    elif 'risk management' in content_lower and 'standard' in content_lower:
        return "Risk Management Standard"
    elif 'scheduling' in content_lower and 'practice' in content_lower:
        return "Scheduling Practice Standard"
    elif 'estimating' in content_lower and 'practice' in content_lower:
        return "Estimating Practice Standard"
    elif 'configuration management' in content_lower:
        return "Configuration Management Standard"
    elif 'business analysis' in content_lower and 'pmi' in content_lower:
        return "PMI Guide to Business Analysis"
    elif 'business analysis' in content_lower:
        return "Business Analysis Practice Guide"
    elif 'requirements management' in content_lower:
        return "Requirements Management Practice Guide"
    elif 'benefits realization' in content_lower:
        return "Benefits Realization Management Guide"
    elif 'portfolio management' in content_lower and 'standard' in content_lower:
        return "Portfolio Management Standard"
    elif 'program management' in content_lower and 'standard' in content_lower:
        return "Program Management Standard"
    elif 'organizational project management' in content_lower:
        return "Organizational Project Management Standard"
    elif 'governance' in content_lower and ('portfolio' in content_lower or 'program' in content_lower):
        return "Governance Practice Guide"
    elif 'managing change' in content_lower or 'change management' in content_lower:
        return "Managing Change Practice Guide"
    elif 'navigating complexity' in content_lower:
        return "Navigating Complexity Practice Guide"
    elif 'process groups' in content_lower:
        return "Process Groups Practice Guide"
    elif 'disciplined agile' in content_lower or 'choose your wow' in content_lower:
        return "Disciplined Agile Practice Guide"
    elif 'ai essentials' in content_lower or ('artificial intelligence' in content_lower and 'project' in content_lower):
        return "AI Essentials for Project Professionals"
    elif 'ai transformation' in content_lower:
        return "Leading AI Transformation"
    elif 'project management offices' in content_lower or 'pmo' in content_lower:
        return "Project Management Offices Guide"
    
    # General categorization
    elif any(term in content_lower for term in ['agile', 'scrum']):
        return "Agile Practice Guides"
    elif 'risk' in content_lower:
        return "Risk Management Guides"
    elif 'quality' in content_lower:
        return "Quality Management Guides"
    elif 'stakeholder' in content_lower:
        return "Stakeholder Management Guides"
    else:
        return "PMI Standards Collection"

# Main rebuild function
print("Rebuilding with clean source names...")

SOURCE_FILE = "data/pmp_combined.txt"
VECTOR_STORE_PATH = "data/vector_store"
BACKUP_PATH = f"data/vector_store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not os.path.exists(SOURCE_FILE):
    print(f"Error: {SOURCE_FILE} not found!")
    exit(1)

print(f"Using: {SOURCE_FILE} ({os.path.getsize(SOURCE_FILE):,} bytes)")

# Backup
if os.path.exists(VECTOR_STORE_PATH):
    shutil.copytree(VECTOR_STORE_PATH, BACKUP_PATH)
    print(f"Backup: {BACKUP_PATH}")

# Load and split
print("Loading and splitting document...")
loader = TextLoader(SOURCE_FILE, encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Extract clean sources
print("Extracting clean source names...")
enhanced_chunks = []
source_counts = {}

for i, chunk in enumerate(chunks):
    clean_source = extract_clean_book_name(chunk.page_content)
    source_counts[clean_source] = source_counts.get(clean_source, 0) + 1
    
    chunk.metadata.update({
        'source': clean_source,
        'book': clean_source,
        'chunk_id': i
    })
    enhanced_chunks.append(chunk)
    
    if i % 200 == 0:
        print(f"  Processed {i}/{len(chunks)}...")

print(f"Enhanced {len(enhanced_chunks)} chunks")
print("\nClean source distribution:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(enhanced_chunks)) * 100
    print(f"  {source}: {count} chunks ({percentage:.1f}%)")

# Create vector store
print("\nBuilding vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(VECTOR_STORE_PATH):
    shutil.rmtree(VECTOR_STORE_PATH)

vector_store = FAISS.from_documents(enhanced_chunks, embeddings)
vector_store.save_local(VECTOR_STORE_PATH)

# Test
print("\nTesting...")
test_results = vector_store.similarity_search("project management", k=3)
for i, result in enumerate(test_results):
    source = result.metadata.get('source', 'Unknown')
    preview = result.page_content[:80]
    print(f"  Test {i+1}: {source}")
    print(f"    {preview}...")

print("\nSUCCESS! Sources should now show clean book names.")
print("Restart your server: python3 app.py")
