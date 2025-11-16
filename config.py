import os
from typing import Optional

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except (ImportError, ModuleNotFoundError):
    # dotenv not available, use system environment variables only
    pass

class Config:
    """Configuration management for the RAG system."""
    
    # Groq API Configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = "vector_store"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # RAG Configuration
    DEFAULT_RETRIEVAL_COUNT: int = 5
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_RESPONSE_TOKENS: int = 1024
    
    # Available Groq models
    AVAILABLE_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    @classmethod
    def validate_groq_config(cls) -> bool:
        """Validate Groq configuration."""
        if not cls.GROQ_API_KEY:
            print("❌ GROQ_API_KEY not found in environment variables.")
            print("Please set your Groq API key:")
            print("1. Get your API key from: https://console.groq.com/keys")
            print("2. Set environment variable: set GROQ_API_KEY=your_key_here")
            print("3. Or create a .env file with: GROQ_API_KEY=your_key_here")
            return False
        
        if cls.GROQ_MODEL not in cls.AVAILABLE_MODELS:
            print(f"⚠️  Model '{cls.GROQ_MODEL}' not in available models. Using default: llama-3.1-8b-instant")
            cls.GROQ_MODEL = "llama-3.1-8b-instant"
        
        return True
    
    @classmethod
    def get_model_info(cls) -> dict:
        """Get current model configuration."""
        return {
            "api_key_set": bool(cls.GROQ_API_KEY),
            "model": cls.GROQ_MODEL,
            "available_models": cls.AVAILABLE_MODELS
        }
