import asyncio
from pathlib import Path
import time
import logging
import traceback

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the dependency checking
from dependency_check import check_dependencies, DependencyError, get_system_status

# Load environment variables
load_dotenv()

# Check required environment variables
required_env_vars = ['GROQ_API_KEY', 'QDRANT_URL', 'QDRANT_HOST', 'INNGEST_API_BASE']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error(f"Missing required environment variables: {missing_vars}")

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

from dependency_check import get_system_status
from vector_db import QdrantStorage

def check_system_status():
    """Check all system components and return status"""
    status = get_system_status()
    
    # Check if we have any documents in Qdrant
    try:
        if status["qdrant"]["status"] == "ok":
            store = QdrantStorage()
            results = store.search([0.0] * 384, top_k=1)  # Dummy search to check for documents
            status["documents"] = {
                "status": "ok" if results.get("contexts") else "error",
                "error": "No documents found in vector store" if not results.get("contexts") else None
            }
    except Exception as e:
        status["documents"] = {
            "status": "error",
            "error": f"Failed to check documents: {str(e)}"
        }
    
    return status

# Show system status in sidebar
st.sidebar.title("System Status")
status = check_system_status()

# First check critical services
if status["groq_ai"]["status"] != "ok":
    st.sidebar.error("⚠️ Groq AI API Not Available")
    error_msg = status["groq_ai"].get("error", "Unknown error")
    st.sidebar.error("""
    The Groq AI API is not properly configured. Please check:
    1. Your API key is correctly set in the .env file
    2. The API key has access to Llama 3.1 8B model
    3. Your API quota is not exceeded
    
    Error: {}
    """.format(error_msg))

for component, details in status.items():
    if component != "groq_ai":  # Skip Groq AI as we handled it above
        component_name = component.replace('_', ' ').title()
        
        if details["status"] == "ok":
            st.sidebar.success(f"✅ {component_name}")
        else:
            st.sidebar.error(f"❌ {component_name}")
            if details.get("error"):
                st.sidebar.caption(f"Error: {details['error']}")

# Show documents status
if status.get("documents", {}).get("status") != "ok":
    st.sidebar.warning("No documents uploaded yet. Please upload a PDF first.")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    try:
        with st.spinner("Validating and saving PDF..."):
            path = save_uploaded_pdf(uploaded)
            logger.info(f"PDF saved to: {path}")
            
        with st.spinner("Triggering PDF ingestion..."):
            event_result = asyncio.run(send_rag_ingest_event(path))
            logger.info(f"Ingestion event sent with ID: {event_result}")
            
        # Wait for initial processing
        with st.spinner("Processing your document..."):
            time.sleep(2)  # Give some time for initial processing
            
            # Get system status
            status = check_system_status()
            if status["qdrant"]["status"] != "ok":
                st.error("Failed to connect to vector database. Please try again.")
                st.stop()
                
        st.success(f"Triggered ingestion for: {path.name}")
        st.info("Your document is being processed. It may take a few moments before it's available for searching.")
        st.caption("You can upload another PDF if you like.")
        
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Failed to process the PDF: {str(e)}")
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Cleaned up failed upload: {path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file: {cleanup_error}")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 600.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    error_details = None
    
    while True:
        try:
            runs = fetch_runs(event_id)
            if runs:
                run = runs[0]
                status = run.get("status")
                error_details = run.get("error")
                
                logger.info(f"Current function status: {status}")
                st.info(f"Processing status: {status}")
                
                last_status = status or last_status
                
                if status in ("Completed", "Succeeded", "Success", "Finished"):
                    return run.get("output") or {}
                    
                if status in ("Failed", "Cancelled"):
                    error_msg = f"Function run {status}"
                    if error_details:
                        error_msg += f"\nError details: {error_details}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except requests.RequestException as e:
            logger.error(f"Network error while fetching run status: {str(e)}")
            raise RuntimeError(f"Failed to fetch run status: {str(e)}")
            
        if time.time() - start > timeout_s:
            timeout_msg = f"Timed out waiting for run output (last status: {last_status})"
            logger.error(timeout_msg)
            raise TimeoutError(timeout_msg)
            
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        # Check system dependencies first
        try:
            check_dependencies()
        except DependencyError as e:
            st.error(str(e))
            st.stop()

        try:
            with st.spinner("Processing your question..."):
                logger.info(f"Processing question: {question}")
                # Fire-and-forget event to Inngest for observability/workflow
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                
            with st.spinner("Generating answer..."):
                # Poll the local Inngest API for the run's output
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

            st.subheader("Answer")
            if not answer:
                st.warning("No answer was generated. This might happen if no relevant content was found in the uploaded documents.")
            else:
                st.write(answer)
                
            if sources:
                st.caption("Sources")
                for s in sources:
                    st.write(f"- {s}")
            else:
                st.info("No source documents were referenced. Try uploading some PDFs first.")
                
        except ConnectionError as e:
            st.error(f"Failed to connect to required services: {str(e)}")
            logger.error(f"Connection error: {str(e)}")
        except RuntimeError as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            logger.error(f"Runtime error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")

