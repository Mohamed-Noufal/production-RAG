import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantStorage:
    def __init__(self, collection="docs", dim=384, max_retries=3):
        # Use environment variable or default to container name for Docker
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection
        self.max_retries = max_retries
        
        try:
            logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
            self.client = QdrantClient(url=self.qdrant_url, timeout=30)
            
            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            
            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection):
                logger.info(f"Creating collection: {collection}")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info(f"Collection {collection} created successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize Qdrant client: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    def upsert(self, ids, vectors, payloads):
        if not ids or not vectors or not payloads:
            raise ValueError("Empty input: ids, vectors, and payloads must not be empty")
        
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("Mismatched lengths: ids, vectors, and payloads must have the same length")
        
        try:
            points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
                     for i in range(len(ids))]
            
            for attempt in range(self.max_retries):
                try:
                    self.client.upsert(self.collection, points=points)
                    logger.info(f"Successfully upserted {len(points)} points")
                    return
                except UnexpectedResponse as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: Upsert failed: {str(e)}")
                    time.sleep(1)
                    
        except Exception as e:
            error_msg = f"Failed to upsert vectors: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def search(self, query_vector, top_k: int = 5):
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
            
        try:
            for attempt in range(self.max_retries):
                try:
                    results = self.client.search(
                        collection_name=self.collection,
                        query_vector=query_vector,
                        with_payload=True,
                        limit=top_k
                    )
                    
                    contexts = []
                    sources = set()

                    for r in results:
                        payload = getattr(r, "payload", None) or {}
                        text = payload.get("text", "")
                        source = payload.get("source", "")
                        if text:
                            contexts.append(text)
                            sources.add(source)

                    logger.info(f"Search completed. Found {len(contexts)} contexts from {len(sources)} sources")
                    return {"contexts": contexts, "sources": list(sources)}
                    
                except UnexpectedResponse as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: Search failed: {str(e)}")
                    time.sleep(1)
                    
        except Exception as e:
            error_msg = f"Failed to search vectors: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)