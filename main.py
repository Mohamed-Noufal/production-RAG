import logging
from fastapi import FastAPI, HTTPException
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
import traceback
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
import groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("GROQ_API_KEY"):
    error_msg = "GROQ_API_KEY environment variable is not set"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Configure Groq client
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Successfully configured Groq API")
except Exception as e:
    logger.error(f"Failed to configure Groq API: {str(e)}")
    raise

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # REMOVED throttle and rate_limit for simplicity
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        logger.info("Starting PDF ingestion process")
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        logger.info(f"Loading PDF from path: {pdf_path}")
        logger.info(f"Using source_id: {source_id}")
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"Successfully loaded and chunked PDF into {len(chunks)} chunks")
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        logger.info("Starting vector embedding and storage process")
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        vecs = embed_texts(chunks)
        logger.info("Successfully generated embeddings")
        
        logger.info("Preparing document IDs and payloads")
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        logger.info("Upserting vectors to Qdrant")
        store = QdrantStorage()
        store.upsert(ids, vecs, payloads)
        logger.info(f"Successfully ingested {len(chunks)} chunks into Qdrant")
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


from dependency_check import check_dependencies, DependencyError

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        try:
            # Check all dependencies first
            check_dependencies()
            
            # Check if we have any documents
            store = QdrantStorage()
            test_results = store.search([0.0] * 384, top_k=1)
            if not test_results["contexts"]:
                raise RuntimeError("No documents found in the vector store. Please upload a PDF first.")
            
            # Generate embeddings
            try:
                query_vec = embed_texts([question])[0]
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
            
            # Search for relevant content
            found = store.search(query_vec, top_k)
            if not found["contexts"]:
                logger.warning(f"No relevant contexts found for question: {question}")
            
            return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    try:
        question = ctx.event.data["question"]
        top_k = int(ctx.event.data.get("top_k", 5))
        logger.info(f"Processing question: {question} with top_k={top_k}")

        found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
        
        if not found.contexts:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "num_contexts": 0
            }

        context_block = "\n\n".join(f"- {c}" for c in found.contexts)
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )

        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."},
                {"role": "user", "content": user_content}
            ]
            
            response = groq_client.chat.completions.create(
                model="llama3-1-8b",
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                top_p=0.9,
                stream=False
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated response")
            return {
                "answer": answer,
                "sources": found.sources,
                "num_contexts": len(found.contexts)
            }
        except Exception as e:
            error_msg = f"Groq API error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)
            
    except Exception as e:
        error_msg = f"RAG query failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])