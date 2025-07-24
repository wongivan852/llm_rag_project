#!/usr/bin/env python3
"""
Quick fix for the RAG system issues
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔧 Quick Fix for Project Management RAG")
    print("=" * 40)
    
    # Step 1: Make sure llama-cpp-python is properly installed
    print("1. Checking llama-cpp-python installation...")
    
    # Test if llama-cpp-python works
    test_code = """
try:
    from langchain_community.llms import LlamaCpp
    print("✅ LlamaCpp import successful")
except Exception as e:
    print(f"❌ LlamaCpp import failed: {e}")
"""
    
    success, stdout, stderr = run_command(f'python3 -c "{test_code}"')
    print(stdout)
    
    if "❌" in stdout:
        print("2. Reinstalling llama-cpp-python...")
        # Reinstall llama-cpp-python with specific settings for macOS
        run_command("pip uninstall llama-cpp-python -y")
        
        # For macOS with Metal support
        if sys.platform == "darwin":
            print("   Installing with Metal support for macOS...")
            success, stdout, stderr = run_command("CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
        else:
            print("   Installing standard version...")
            success, stdout, stderr = run_command("pip install llama-cpp-python --force-reinstall --no-cache-dir")
        
        if success:
            print("   ✅ llama-cpp-python reinstalled")
        else:
            print("   ❌ Failed to reinstall llama-cpp-python")
            print(f"   Error: {stderr}")
    
    # Step 2: Test the fixed app
    print("\n3. Testing imports with the new app...")
    
    test_imports = """
try:
    from langchain_community.llms import LlamaCpp
    print("✅ LlamaCpp import works")
except Exception as e:
    print(f"❌ LlamaCpp still failing: {e}")

try:
    from flask import Flask
    from flask_cors import CORS
    print("✅ Flask imports work")
except Exception as e:
    print(f"❌ Flask imports failed: {e}")
"""
    
    success, stdout, stderr = run_command(f'python3 -c "{test_imports}"')
    print(stdout)
    
    print("\n4. Checking file structure...")
    
    required_files = [
        "data/pmp_combined.txt",
        "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ Missing: {file_path}")
            all_files_exist = False
    
    print("\n" + "=" * 40)
    if "❌" not in stdout and all_files_exist:
        print("🎉 Everything looks good!")
        print("You should now be able to run:")
        print("   python3 app.py")
        print("\nThen open rag-interface.html in your browser")
    else:
        print("⚠️ Some issues remain:")
        if not all_files_exist:
            print("- Missing required files (see above)")
        if "❌" in stdout:
            print("- Import issues (see above)")
        print("\nThe new app.py should work even with some issues.")

if __name__ == "__main__":
    main()