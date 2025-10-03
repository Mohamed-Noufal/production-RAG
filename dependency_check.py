import os
import logging
from pathlib import Path
from typing import Optional

import groq
import sentence_transformers
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class DependencyError(Exception):
    """Raised when a critical dependency check fails"""
    pass

def check_groq_ai() -> Optional[str]:
    """Check if Groq API key is valid and models are accessible"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "GROQ_API_KEY environment variable not set"
        
        if not api_key.startswith("gsk_"):
            return "Invalid Groq API key format. Key should start with 'gsk_'"
        
        try:
            client = groq.Groq(api_key=api_key)
            # Test connection with a small request
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                temperature=0,
                max_tokens=10
            )
            if not response:
                return "Failed to get response from Groq API"
            return None
        except Exception as model_error:
            if "unauthorized" in str(model_error).lower():
                return "Groq API key is not authorized. Please check your API key"
            elif "quota" in str(model_error).lower():
                return "Groq API quota exceeded. Please check your usage limits"
            else:
                return f"Failed to access Groq API: {str(model_error)}"
    except Exception as e:
        return f"Groq API configuration error: {str(e)}"

def check_qdrant() -> Optional[str]:
    """Check if Qdrant is accessible"""
    try:
        client = QdrantClient("localhost", port=6333)
        client.get_collections()
        return None
    except Exception as e:
        return f"Qdrant connection error: {str(e)}"

def check_embedding_model() -> Optional[str]:
    """Check if the embedding model is available"""
    try:
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["test"])
        if test_embedding.shape != (1, 384):
            return f"Unexpected embedding shape: {test_embedding.shape}"
        return None
    except Exception as e:
        return f"Embedding model error: {str(e)}"

def check_dependencies() -> None:
    """Check all critical dependencies and raise an error if any fail"""
    errors = []
    
    # Check Groq AI
    if error := check_groq_ai():
        errors.append(error)
        
    # Check Qdrant
    if error := check_qdrant():
        errors.append(error)
        
    # Check embedding model
    if error := check_embedding_model():
        errors.append(error)
    
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Dependency check failed:\n{error_msg}")
        raise DependencyError(f"System dependency check failed:\n{error_msg}")

def get_system_status() -> dict:
    """Get the status of all system dependencies"""
    return {
        "groq_ai": {"status": "ok" if not check_groq_ai() else "error", 
                   "error": check_groq_ai()},
        "qdrant": {"status": "ok" if not check_qdrant() else "error",
                  "error": check_qdrant()},
        "embedding_model": {"status": "ok" if not check_embedding_model() else "error",
                          "error": check_embedding_model()}
    }