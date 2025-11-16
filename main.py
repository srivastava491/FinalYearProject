#!/usr/bin/env python3
"""
NIT Kurukshetra Web Scraping and RAG System
============================================

This script provides a complete pipeline for scraping the NIT Kurukshetra website,
extracting text content, generating vector embeddings, and implementing a RAG system
for question-answering.

Usage:
    python main.py [command]

Commands:
    scrape      - Scrape the NIT KKR website
    embed       - Generate vector embeddings from scraped content
    rag         - Run the RAG system in interactive mode
    full        - Run the complete pipeline (scrape + embed + rag)
    stats       - Show system statistics
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    # Core packages that are absolutely required
    core_packages = ['requests', 'beautifulsoup4', 'numpy']
    
    # Optional packages with fallbacks
    optional_packages = ['sentence_transformers', 'faiss']
    
    missing_core = []
    missing_optional = []
    
    for package in core_packages:
        try:
            if package == 'beautifulsoup4':
                __import__('bs4')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_core.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_core:
        print("❌ Missing required packages:")
        for package in missing_core:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print("pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print("⚠️  Optional packages not available (using fallbacks):")
        for package in missing_optional:
            print(f"   - {package}")
        print("System will work with reduced functionality.\n")
    
    print("✅ Core dependencies are installed")
    return True

def run_scraper():
    """Run the web scraper."""
    print("🕷️  Starting web scraping...")
    try:
        from scraper import main as scraper_main
        scraper_main()
        print("✅ Web scraping completed successfully")
        return True
    except Exception as e:
        print(f"❌ Web scraping failed: {e}")
        return False

def run_embeddings():
    """Run the vector embedding generation."""
    print("🧠 Generating vector embeddings...")
    try:
        from vector_embeddings import main as embeddings_main
        embeddings_main()
        print("✅ Vector embeddings generated successfully")
        return True
    except Exception as e:
        print(f"❌ Vector embedding generation failed: {e}")
        return False

def run_rag():
    """Run the RAG system."""
    print("🤖 Starting RAG system...")
    try:
        from rag_system import main as rag_main
        rag_main()
        return True
    except Exception as e:
        print(f"❌ RAG system failed: {e}")
        return False

def show_stats():
    """Show system statistics."""
    try:
        from vector_embeddings import VectorEmbeddingSystem
        embedding_system = VectorEmbeddingSystem()
        
        if embedding_system.load_vector_store():
            stats = embedding_system.get_stats()
            print("\n📊 System Statistics:")
            print("=" * 40)
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Unique documents: {stats['unique_documents']}")
            print(f"Total words: {stats['total_words']}")
            print(f"Average chunk length: {stats['average_chunk_length']:.1f} words")
            print(f"Model: {stats['model_name']}")
            print(f"Embedding dimension: {stats['embedding_dimension']}")
        else:
            print("❌ Vector store not found. Please run 'python main.py embed' first.")
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def check_data_exists():
    """Check if scraped data exists."""
    return Path("extracted_text").exists() and len(list(Path("extracted_text").glob("*.txt"))) > 0

def check_vector_store_exists():
    """Check if vector store exists."""
    return Path("vector_store/nitkkr_index.faiss").exists()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="NIT Kurukshetra Web Scraping and RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'command',
        choices=['scrape', 'embed', 'rag', 'full', 'stats'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    print("🚀 NIT Kurukshetra RAG System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Execute command
    if args.command == 'scrape':
        return 0 if run_scraper() else 1
        
    elif args.command == 'embed':
        if not check_data_exists():
            print("❌ No scraped data found. Please run 'python main.py scrape' first.")
            return 1
        return 0 if run_embeddings() else 1
        
    elif args.command == 'rag':
        if not check_vector_store_exists():
            print("❌ Vector store not found. Please run 'python main.py embed' first.")
            return 1
        return 0 if run_rag() else 1
        
    elif args.command == 'full':
        print("🔄 Running complete pipeline...")
        
        # Step 1: Scrape
        if not run_scraper():
            return 1
        
        # Step 2: Generate embeddings
        if not run_embeddings():
            return 1
        
        # Step 3: Start RAG system
        print("\n🎉 Pipeline completed! Starting RAG system...")
        return 0 if run_rag() else 1
        
    elif args.command == 'stats':
        show_stats()
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
