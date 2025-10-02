import logging
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llama_index.core import Settings
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize the open-source embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

try:
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedding_model = SentenceTransformer(EMBED_MODEL)
    logger.info("Embedding model loaded successfully")
except Exception as e:
    error_msg = f"Failed to load embedding model: {str(e)}"
    logger.error(f"{error_msg}\n{traceback.format_exc()}")
    raise RuntimeError(error_msg)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    try:
        logger.info(f"Loading PDF from path: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found at path: {path}")
            
        docs = PDFReader().load_data(file=path)
        if not docs:
            raise ValueError(f"No content could be extracted from PDF: {path}")
            
        texts = [d.text for d in docs if getattr(d, "text", None)]
        if not texts:
            raise ValueError(f"No text content found in PDF: {path}")
            
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))
            
        logger.info(f"Successfully chunked PDF into {len(chunks)} segments")
        return chunks
        
    except Exception as e:
        error_msg = f"Failed to load or chunk PDF: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)

def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        if not texts:
            raise ValueError("No texts provided for embedding")
            
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = embedding_model.encode(texts)
        logger.info("Successfully generated embeddings")
        
        return embeddings.tolist()
        
    except Exception as e:
        error_msg = f"Failed to generate embeddings: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)