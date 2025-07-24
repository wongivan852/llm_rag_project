#!/usr/bin/env python3
"""
Setup Checker for Project Management RAG System
This script checks if all required files and dependencies are in place
"""

import os
import sys
import subprocess
import pkg_resources

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        "data",
        "models", 
        "logs",
        "data/vector_store"
    ]
    
    print("🔍 Checking directory structure...")
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"❌ Missing directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    if missing_dirs:
        print(f"\n📁 Creating missing directories...")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")
    
    return len(missing_dirs) == 0

def check_files():
    """Check if required files exist"""
    required_files = [
        ("data/pmp_combined.txt", "Training data file"),
        ("models/llama-3.2-3b-instruct-q4_k_m.gguf", "Llama 3.2 3B model file"),
        ("data/vector_store/index.faiss", "FAISS vector store index"),
        ("app.py", "Flask application"),
        ("rag-interface.html", "Web interface")
    ]
    
    print("\n🔍 Checking required files...")
    missing_files = []
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_files.append((file_path, description))
            print(f"❌ Missing file: {file_path} ({description})")
        else:
            file_size = os.path.getsize(file_path)
            print(f"✅ File exists: {file_path} ({description}) - {file_size:,} bytes")
    
    return missing_files

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        "flask",
        "flask-cors", 
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "faiss-cpu",  # or faiss-gpu
        "sentence-transformers",
        "llama-cpp-python"
    ]
    
    print("\n🔍 Checking Python packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ Package installed: {package}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"❌ Missing package: {package}")
    
    return missing_packages

def generate_install_script(missing_packages):
    """Generate installation script for missing packages"""
    if not missing_packages:
        return None
    
    script_content = """#!/bin/bash
# Auto-generated installation script for missing packages

echo "Installing missing Python packages..."

"""
    
    for package in missing_packages:
        script_content += f'pip install {package}\n'
    
    script_content += """
echo "Installation complete!"
echo "You may need to restart your terminal after installation."
"""
    
    with open("install_requirements.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("install_requirements.sh", 0o755)
    
    return "install_requirements.sh"

def check_model_download():
    """Check if model needs to be downloaded"""
    model_path = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        print(f"\n📥 Model Download Instructions:")
        print(f"The Llama 3.2 3B model is not found at: {model_path}")
        print(f"")
        print(f"To download the model:")
        print(f"1. Visit: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        print(f"2. Download: llama-3.2-3b-instruct-q4_k_m.gguf")
        print(f"3. Place it in the models/ directory")
        print(f"")
        print(f"Alternative command (if you have huggingface-hub):")
        print(f"huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF llama-3.2-3b-instruct-q4_k_m.gguf --local-dir models/")
        
        return False
    
    return True

def main():
    """Main setup checker function"""
    print("🚀 Project Management RAG System - Setup Checker")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Check directories
    dirs_ok = check_directories()
    
    # Check files
    missing_files = check_files()
    
    # Check Python packages
    missing_packages = check_python_packages()
    
    # Check model download
    model_ok = check_model_download()
    
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    if dirs_ok:
        print("✅ Directory structure: OK")
    else:
        print("⚠️ Directory structure: Fixed")
    
    if not missing_files:
        print("✅ Required files: All present")
    else:
        print("❌ Missing files:")
        for file_path, description in missing_files:
            print(f"   - {file_path} ({description})")
    
    if not missing_packages:
        print("✅ Python packages: All installed")
    else:
        print("❌ Missing Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        
        install_script = generate_install_script(missing_packages)
        if install_script:
            print(f"\n💡 Generated installation script: {install_script}")
            print(f"Run: chmod +x {install_script} && ./{install_script}")
    
    if model_ok:
        print("✅ Model file: Present")
    else:
        print("❌ Model file: Missing (see download instructions above)")
    
    # Overall status
    all_ready = (dirs_ok and not missing_files and not missing_packages and model_ok)
    
    print("\n" + "=" * 50)
    if all_ready:
        print("🎉 SYSTEM READY! You can now run the application:")
        print("   python3 app.py")
        print("   Then open rag-interface.html in your browser")
    else:
        print("⚠️ SETUP INCOMPLETE - Please resolve the issues above")
        
        if missing_packages:
            print("\n🔧 Quick fix for packages:")
            print(f"pip install {' '.join(missing_packages)}")
    
    return all_ready

if __name__ == "__main__":
    main()