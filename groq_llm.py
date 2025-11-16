import os
import logging
from typing import Optional, Dict, Any

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except (ImportError, ModuleNotFoundError):
    # dotenv not available, use system environment variables only
    pass

try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("Groq SDK not available. Please install with: pip install groq")
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class GroqLLMService:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the Groq LLM service.
        
        Args:
            api_key: Groq API key (if not provided, will try to get from environment)
            model: Groq model to use (default: llama3-8b-8192)
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq SDK is not available. Please install with: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)  # type: ignore
        
        logger.info(f"Groq LLM service initialized with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate a response using Groq LLM.
        
        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                stream=False,
                stop=None
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            raise
    
    def generate_rag_response(self, query: str, context: str, max_tokens: int = 1024) -> str:
        """
        Generate a RAG response using retrieved context.
        
        Args:
            query: User query
            context: Retrieved context from vector search
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response based on context
        """
        # Create a comprehensive prompt for RAG
        prompt = f"""You are a helpful assistant for NIT Kurukshetra (National Institute of Technology Kurukshetra). 
Your role is to answer questions about the institute based on the provided context from their official website.

User Query: {query}

Context from NIT Kurukshetra website:
{context}

Instructions:
1. Answer the user's question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be accurate and factual - only use information from the provided context
4. If you find relevant information, present it in a clear and organized way
5. Include specific details like dates, requirements, procedures, etc. when available
6. If the query is about admissions, academics, departments, or facilities, focus on those aspects
7. Keep your response concise but informative
8. If you cannot answer based on the context, suggest the user visit the official website for more details

Response:"""

        try:
            return self.generate_response(prompt, max_tokens, temperature=0.3)  # Lower temperature for factual responses
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            # Fallback response
            return f"I apologize, but I'm experiencing technical difficulties. However, based on the retrieved information about NIT Kurukshetra, please visit their official website for detailed information about '{query}'."
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.generate_response("Hello, please respond with 'Connection successful'", max_tokens=10)
            logger.info("Groq API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Groq API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model and service.
        
        Returns:
            Dictionary with model information
        """
        return {
            "service": "Groq",
            "model": self.model,
            "api_available": GROQ_AVAILABLE,
            "api_key_set": bool(self.api_key)
        }


def create_groq_service(api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant") -> Optional[GroqLLMService]:
    """
    Factory function to create a Groq LLM service.
    
    Args:
        api_key: Groq API key
        model: Groq model to use
        
    Returns:
        GroqLLMService instance or None if creation fails
    """
    try:
        return GroqLLMService(api_key=api_key, model=model)
    except Exception as e:
        logger.error(f"Failed to create Groq service: {e}")
        return None


def main():
    """Test function for Groq LLM service."""
    try:
        # Create service
        service = create_groq_service()
        if not service:
            print("Failed to create Groq service. Please check your API key.")
            return
        
        # Test connection
        if service.test_connection():
            print("✅ Groq LLM service is working correctly!")
            
            # Test RAG response
            test_query = "What are the admission requirements?"
            test_context = "NIT Kurukshetra offers various undergraduate and postgraduate programs. Admission is based on JEE Main scores for B.Tech programs and GATE scores for M.Tech programs."
            
            response = service.generate_rag_response(test_query, test_context)
            print(f"\nTest RAG Response:")
            print(f"Query: {test_query}")
            print(f"Response: {response}")
        else:
            print("❌ Groq LLM service connection failed.")
            
    except Exception as e:
        print(f"Error testing Groq service: {e}")


if __name__ == "__main__":
    main()
