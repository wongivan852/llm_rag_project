#!/usr/bin/env python3
"""
LangChain Version Compatibility Fixer
This script fixes version conflicts in LangChain packages
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_langchain_versions():
    """Fix LangChain version compatibility issues"""
    
    print("🔧 Fixing LangChain Version Compatibility Issues")
    print("=" * 50)
    
    # Step 1: Uninstall problematic packages
    print("1. Uninstalling current LangChain packages...")
    uninstall_packages = [
        "langchain",
        "langchain-community", 
        "langchain-core",
        "langchain-huggingface"
    ]
    
    for package in uninstall_packages:
        print(f"   Uninstalling {package}...")
        success, stdout, stderr = run_command(f"pip uninstall {package} -y")
        if success:
            print(f"   ✅ {package} uninstalled")
        else:
            print(f"   ⚠️ {package} was not installed or failed to uninstall")
    
    # Step 2: Install compatible versions
    print("\n2. Installing compatible LangChain versions...")
    
    # Install core packages first with specific compatible versions
    compatible_packages = [
        "langchain-core==0.1.45",
        "langchain-community==0.0.29", 
        "langchain==0.1.16"
    ]
    
    for package in compatible_packages:
        print(f"   Installing {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            print(f"   ✅ {package} installed successfully")
        else:
            print(f"   ❌ Failed to install {package}")
            print(f"   Error: {stderr}")
    
    # Step 3: Install HuggingFace embeddings
    print("\n3. Installing HuggingFace embeddings...")
    success, stdout, stderr = run_command("pip install langchain-huggingface==0.0.3")
    if success:
        print("   ✅ langchain-huggingface installed successfully")
    else:
        print("   ⚠️ langchain-huggingface failed, trying alternative...")
        # Fallback to community embeddings
        success2, stdout2, stderr2 = run_command("pip install sentence-transformers")
        if success2:
            print("   ✅ sentence-transformers installed as fallback")
        else:
            print("   ❌ Failed to install embeddings package")
    
    # Step 4: Test imports
    print("\n4. Testing imports...")
    test_script = '''
try:
    from langchain_community.llms import LlamaCpp
    print("✅ LlamaCpp import successful")
except Exception as e:
    print(f"❌ LlamaCpp import failed: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("✅ FAISS import successful")
except Exception as e:
    print(f"❌ FAISS import failed: {e}")

try:
    from langchain.chains import RetrievalQA
    print("✅ RetrievalQA import successful")
except Exception as e:
    print(f"❌ RetrievalQA import failed: {e}")

try:
    from langchain.prompts import PromptTemplate
    print("✅ PromptTemplate import successful")
except Exception as e:
    print(f"❌ PromptTemplate import failed: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ HuggingFaceEmbeddings import successful")
except Exception as e:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ HuggingFaceEmbeddings (community) import successful")
    except Exception as e2:
        print(f"❌ HuggingFaceEmbeddings import failed: {e2}")
'''
    
    success, stdout, stderr = run_command(f'python3 -c "{test_script}"')
    print(stdout)
    if stderr:
        print(f"Warnings: {stderr}")
    
    print("\n" + "=" * 50)
    if "❌" not in stdout:
        print("🎉 All imports successful! You can now run:")
        print("   python3 app.py")
    else:
        print("⚠️ Some imports failed. Trying alternative approach...")
        print("\nAlternative fix: Install older stable versions")
        
        # Alternative: use older but stable versions
        alt_packages = [
            "pip uninstall langchain langchain-community langchain-core -y",
            "pip install langchain==0.0.354",
            "pip install sentence-transformers",
            "pip install faiss-cpu"
        ]
        
        for cmd in alt_packages:
            print(f"Running: {cmd}")
            run_command(cmd)

def main():
    print("🚀 LangChain Compatibility Fixer")
    print("This will fix version conflicts in your LangChain installation")
    
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() in ['y', 'yes']:
        fix_langchain_versions()
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()