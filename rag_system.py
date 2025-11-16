import os
import json
from typing import List, Dict, Optional
import logging
from vector_embeddings import VectorEmbeddingSystem
from groq_llm import GroqLLMService, create_groq_service
from config import Config
import warnings

# --- NEW IMPORT ---
# Try to import the CrossEncoder for reranking
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    RERANKER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RERANKER_AVAILABLE = False
# --- END NEW IMPORT ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, vector_store_path="vector_store", use_groq=True, groq_model=None):
        """
        Initialize the RAG (Retrieval-Augmented Generation) system.
        
        Args:
            vector_store_path: Path to the vector store directory
            use_groq: Whether to use Groq LLM for response generation
            groq_model: Specific Groq model to use (optional)
        """
        self.vector_store_path = vector_store_path
        self.embedding_system = VectorEmbeddingSystem()
        self.use_groq = use_groq
        self.groq_service = None
        self.reranker = None  # --- NEW: Reranker model
        
        # Load the vector store
        if not self.embedding_system.load_vector_store():
            logger.error("Failed to load vector store. Please generate embeddings first.")
            raise FileNotFoundError("Vector store not found. Run vector_embeddings.py first.")
        
        # Initialize Groq LLM service if requested
        if self.use_groq:
            try:
                model = groq_model or Config.GROQ_MODEL
                self.groq_service = create_groq_service(model=model)
                if self.groq_service:
                    logger.info(f"Groq LLM service initialized with model: {model}")
                else:
                    logger.warning("Failed to initialize Groq service. Falling back to template-based responses.")
                    self.use_groq = False
            except Exception as e:
                logger.warning(f"Failed to initialize Groq service: {e}. Falling back to template-based responses.")
                self.use_groq = False
        
        # --- NEW: Initialize Reranker ---
        if RERANKER_AVAILABLE:
            try:
                # This is a lightweight but effective reranker
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Loaded Cross-Encoder reranker model successfully.")
            except Exception as e:
                logger.warning(f"Could not load reranker model: {e}. Reranking will be disabled.")
                self.reranker = None
        else:
            logger.warning("sentence-transformers not installed. Reranking will be disabled.")
        # --- END NEW: Reranker ---
            
        logger.info("RAG system initialized successfully")
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve and rerank relevant documents for a given query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks with metadata
        """
        logger.info(f"Searching for: '{query}'")
        
        # 1. Retrieve (get more results than needed, e.g., 20 or 4*k)
        initial_k = max(20, k * 4)
        results = self.embedding_system.search(query, k=initial_k)
        
        if not results:
            logger.info("Found 0 chunks.")
            return []
            
        # 2. Rerank (if model is available)
        if self.reranker:
            logger.info(f"Reranking {len(results)} results...")
            
            # Create pairs of (query, chunk_text)
            pairs = [(query, result['chunk_text']) for result in results]
            
            # Score the pairs
            try:
                scores = self.reranker.predict(pairs)
                
                # Add scores back to results and sort
                for i, result in enumerate(results):
                    result['rerank_score'] = scores[i]
                    
                results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # 3. Select the top K from the reranked list
                final_results = results[:k]
                logger.info(f"Found {len(final_results)} reranked results.")
                
                # Update similarity_score with the more accurate rerank_score for downstream use
                for res in final_results:
                    res['similarity_score'] = res['rerank_score']
                    
                return final_results
                
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Falling back to standard retrieval.")
                # Fallback to standard retrieval if reranking predict fails
                return results[:k]
            
        else:
            # Fallback to standard retrieval if no reranker
            logger.info(f"Found {len(results)} results (no reranking).")
            return results[:k]  # Return top k from original search
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown Title')
            url = result.get('url', 'Unknown URL')
            chunk_text = result.get('chunk_text', '')
            
            # Use rerank_score if available, else similarity_score
            score = result.get('rerank_score', result.get('similarity_score', 0))
            score_type = "Rerank Score" if 'rerank_score' in result else "Relevance Score"
            
            context_part = f"[Source {i}]\n"
            context_part += f"Title: {title}\n"
            context_part += f"URL: {url}\n"
            context_part += f"{score_type}: {score:.3f}\n"
            context_part += f"Content: {chunk_text}\n"
            context_part += "-" * 50 + "\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the retrieved context.
        Uses Groq LLM if available, otherwise falls back to template-based generation.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        if "No relevant information found" in context:
            return f"I couldn't find specific information about '{query}' in the NIT Kurukshetra website. Please try rephrasing your question or ask about different topics like academics, admissions, departments, or facilities."
        
        # Use Groq LLM if available
        if self.use_groq and self.groq_service:
            try:
                return self.groq_service.generate_rag_response(query, context, max_tokens=Config.MAX_RESPONSE_TOKENS)
            except Exception as e:
                logger.warning(f"Groq LLM generation failed: {e}. Falling back to template-based response.")
        
        # Fallback to template-based response
        response = f"Based on the information from NIT Kurukshetra's website, here's what I found regarding '{query}':\n\n"
        
        # Extract key information from context
        sources = context.split("[Source")
        relevant_info = []
        
        for source in sources[1:]:  # Skip the first empty split
            lines = source.strip().split('\n')
            content_line = None
            for line in lines:
                if line.startswith("Content:"):
                    content_line = line.replace("Content:", "").strip()
                    break
            if content_line and len(content_line) > 20:  # Only include substantial content
                relevant_info.append(content_line)
        
        # Combine relevant information
        if relevant_info:
            response += "\n".join(f"• {info}" for info in relevant_info[:3])  # Limit to top 3 points
            response += "\n\nFor more detailed information, you can visit the official NIT Kurukshetra website."
        else:
            response += "The retrieved information appears to be limited. Please try a more specific query or check the official website for comprehensive details."
        
        return response
    
    def answer_query(self, query: str, k: int = 5) -> Dict:
        """
        Answer a user query using the RAG system.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            logger.info(f"Starting query processing for: '{query}'")
            
            # Check if vector store is loaded
            if not self.embedding_system.metadata:
                return {
                    "query": query,
                    "response": "Vector store is not loaded. Please ensure embeddings have been generated first.",
                    "sources": [],
                    "num_sources": 0
                }
            
            # Retrieve relevant documents (now with reranking)
            logger.info("Retrieving relevant documents...")
            results = self.retrieve_relevant_documents(query, k)
            
            # Format context
            logger.info("Formatting context...")
            context = self.format_context(results)
            
            # Generate response
            logger.info("Generating response...")
            response = self.generate_response(query, context)
            
            return {
                "query": query,
                "response": response,
                "sources": [
                    {
                        "title": r.get('title', 'Unknown'),
                        "url": r.get('url', 'Unknown'),
                        # Use rerank_score if available, else similarity_score
                        "score": r.get('rerank_score', r.get('similarity_score', 0)),
                        "score_type": "rerank" if 'rerank_score' in r else "similarity",
                        "content_preview": r.get('chunk_text', '')[:200] + "..." if len(r.get('chunk_text', '')) > 200 else r.get('chunk_text', '')
                    }
                    for r in results
                ],
                "num_sources": len(results)
            }
            
        except Exception as e:
            error_msg = str(e) if str(e).strip() else f"Error of type {type(e).__name__}"
            logger.error(f"Error processing query '{query}': {error_msg}")
            
            return {
                "query": query,
                "response": f"Sorry, I encountered an error while processing your query: {error_msg}. Please try a different question or check if the vector store is properly set up.",
                "sources": [],
                "num_sources": 0
            }
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the RAG system."""
        stats = self.embedding_system.get_stats()
        
        # Add Groq LLM information
        stats.update({
            "groq_enabled": self.use_groq,
            "groq_model": self.groq_service.model if self.groq_service else None,
            "groq_available": self.groq_service is not None,
            "reranker_enabled": self.reranker is not None  # --- NEW STAT ---
        })
        
        return stats
    
    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        print("\n" + "="*60)
        print("🤖 NIT Kurukshetra RAG Assistant")
        print("="*60)
        print("Ask questions about NIT Kurukshetra!")
        print("Examples:")
        print("• What are the admission requirements?")
        print("• Tell me about the computer science department")
        print("• What facilities are available?")
        print("• How to apply for courses?")
        print("\nType 'quit' or 'exit' to stop.")
        
        # Show system status
        if self.use_groq and self.groq_service:
            print(f"🧠 Powered by Groq LLM ({self.groq_service.model})")
        else:
            print("📝 Using template-based responses")
        
        if self.reranker:  # --- NEW STATUS ---
            print("✨ Reranking enabled (for better accuracy)")
        else:
            print("⚠️  Reranking disabled (falling back to standard search)")
            
        print("="*60)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                print("\n🔍 Searching and Reranking...")
                result = self.answer_query(query)
                
                print(f"\n💬 Answer:")
                print("-" * 40)
                print(result['response'])
                
                if result['sources']:
                    print(f"\n📚 Sources ({result['num_sources']}):")
                    for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                        score_type = "Rerank Score" if source['score_type'] == 'rerank' else "Score"
                        print(f"{i}. {source['title']} ({score_type}: {source['score']:.3f})")
                        print(f"   URL: {source['url']}")
                        print(f"   Preview: {source['content_preview']}")
                        print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the RAG system."""
    try:
        # Check Groq configuration
        if not Config.validate_groq_config():
            print("\n⚠️  Groq LLM will not be available. The system will use template-based responses.")
            use_groq = False
        else:
            use_groq = True
        
        # Initialize RAG system
        rag = RAGSystem(use_groq=use_groq)
        
        # Show system stats
        stats = rag.get_system_stats()
        print(f"📊 System Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Unique documents: {stats['unique_documents']}")
        print(f"   Total words: {stats['total_words']}")
        print(f"   Embedding Model: {stats['model_name']}")
        print(f"   Groq LLM: {'✅ Enabled' if stats['groq_enabled'] and stats['groq_available'] else '❌ Disabled'}")
        if stats['groq_model']:
            print(f"   Groq Model: {stats['groq_model']}")
        # --- NEW STAT ---
        print(f"   Reranker:   {'✅ Enabled' if stats['reranker_enabled'] else '❌ Disabled'}")
        
        # Run interactive mode
        rag.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run the following commands first:")
        print("1. python scraper.py")
        print("2. python vector_embeddings.py")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")


if __name__ == "__main__":
    main()
