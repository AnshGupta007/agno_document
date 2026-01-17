#!/usr/bin/env python3
"""
Main entry point for the Agno Document Q&A System
This script provides multiple ways to run the system:
1. Streamlit web interface
2. Command-line interface
3. Direct agent initialization
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is properly configured"""
    issues = []
    
    # Check GROQ_API_KEY
    if not os.getenv('GROQ_API_KEY'):
        issues.append("❌ GROQ_API_KEY environment variable not set")
    else:
        print("✅ GROQ_API_KEY found")
    
    # Check required packages
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        issues.append("❌ Streamlit not installed - run: pip install streamlit")
    
    try:
        import agno
        print("✅ Agno framework available")
    except ImportError:
        issues.append("❌ Agno framework not installed - run: pip install agno")
    
    try:
        import groq
        print("✅ Groq client available")
    except ImportError:
        issues.append("❌ Groq client not installed - run: pip install groq")
    
    try:
        import sentence_transformers
        print("✅ SentenceTransformers available")
    except ImportError:
        issues.append("❌ SentenceTransformers not installed - run: pip install sentence-transformers")
    
    try:
        import faiss
        print("✅ FAISS available")
    except ImportError:
        issues.append("❌ FAISS not installed - run: pip install faiss-cpu")
    
    return issues

def run_streamlit():
    """Run the Streamlit web interface"""
    print("🚀 Starting Streamlit web interface...")
    print("📱 The web app will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    
    os.system("streamlit run streamlit_app.py")

def run_cli():
    """Run the command-line interface"""
    print("🚀 Starting command-line interface...")
    
    try:
        from agno_agent import main
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_setup():
    """Run initial setup and checks"""
    print("🔧 Agno Document Q&A System Setup")
    print("=" * 50)
    
    issues = check_environment()
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n📋 Setup Instructions:")
        print("1. Set GROQ_API_KEY environment variable:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print("\n2. Install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n3. Add documents to the documents/ folder")
        print("   Supported: .txt, .md, .json, .docx, .py, .js, .html, .css")
        
        return False
    else:
        print("\n✅ All checks passed! System ready to use.")
        return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Agno Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run setup check
  python main.py --web             # Start Streamlit web interface
  python main.py --cli             # Start command-line interface
  python main.py --setup           # Run setup and environment check
        """
    )
    
    parser.add_argument(
        '--web', 
        action='store_true', 
        help='Start Streamlit web interface'
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true', 
        help='Start command-line interface'
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true', 
        help='Run setup and environment check'
    )
    
    parser.add_argument(
        '--model', 
        default='openai/gpt-oss-20b',
        help='Groq model to use (default: openai/gpt-oss-20b)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("📚 Agno Document Q&A System")
    print("=" * 40)
    print("🤖 Powered by Groq AI + Local Embeddings")
    print("🔍 RAG-based Document Question Answering")
    print("=" * 40)
    
    # Handle arguments
    if args.setup:
        run_setup()
    elif args.web:
        if run_setup():
            run_streamlit()
    elif args.cli:
        if run_setup():
            run_cli()
    else:
        # Default: run setup check and show options
        if run_setup():
            print("\n🎯 Choose how to run the system:")
            print("   python main.py --web     # Web interface (recommended)")
            print("   python main.py --cli     # Command-line interface")
            print("   python main.py --setup   # Re-run setup check")
        else:
            print("\n🔧 Please fix the issues above and run again")

if __name__ == "__main__":
    main()
