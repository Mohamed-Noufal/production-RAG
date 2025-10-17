# Production-Grade RAG App

This is a production-ready Retrieval-Augmented Generation (RAG) application designed for ingesting PDF documents, storing them in a local vector database, and enabling semantic search and question-answering over the content. It leverages modern tools for scalability, async processing, and user-friendly interaction.


## Overview

The app allows users to:
- Upload PDFs via a Streamlit web interface.
- Automatically process (chunk, embed, and store) the documents using event-driven workflows.
- Query the knowledge base with natural language questions, retrieving relevant contexts and generating concise answers powered by an LLM.

Key design principles:
- **Modular Architecture**: Separation of concerns with dedicated modules for data loading, vector storage, and workflows.
- **Async and Scalable**: Uses Inngest for durable, event-driven execution to handle ingestion and querying asynchronously.
- **Local-First**: Runs entirely locally with Qdrant for vector storage and no external dependencies beyond API keys.
- **Type-Safe**: Employs Pydantic models for structured data handling.

## Features

- **PDF Ingestion Pipeline**:
  - Load and parse PDFs using LlamaIndex's `PDFReader`.
  - Intelligent text chunking with `SentenceSplitter` (chunk size: 1000, overlap: 200) to preserve context.
  - Generate embeddings using Sentence Transformers' `all-MiniLM-L6-v2` model (384 dimensions).
  - Store vectors and payloads (text chunks + source metadata) in Qdrant with unique UUID-based IDs.
  - Robust error handling and retry mechanisms for reliable processing.

- **Semantic Retrieval**:
  - Embed user queries with the same Sentence Transformers model.
  - Perform cosine similarity search in Qdrant to retrieve top-K (default: 5) relevant chunks.
  - Extract and return contexts and unique sources for transparency.

- **Augmented Generation**:
  - Use Llama 3.1 8B via Groq API for generating concise, context-grounded answers.
  - System prompt enforces using only provided context to minimize hallucinations.
  - Configurable temperature (0.2) for consistent outputs and max tokens (512).

- **User Interface**:
  - Streamlit-based web app for PDF uploads and querying.
  - Real-time feedback during ingestion (spinner + success message).
  - Query form with adjustable top-K retrieval.
  - Display answers with cited sources for verifiability.
  - Handles multiple uploads sequentially.

- **Workflow Orchestration**:
  - Inngest functions for `rag/ingest_pdf` (ingestion) and `rag/query_pdf_ai` (querying).
  - Step-based execution with error handling and timeouts (120s for queries).
  - Polling mechanism in UI to wait for workflow completion and fetch results.
  - FastAPI integration for serving Inngest endpoints.

- **Production-Ready Elements**:
  - Environment management with `python-dotenv` for API keys (e.g., `OPENAI_API_KEY`).
  - Local Qdrant server integration (default: `http://localhost:6333`).
  - UUID generation for deduplication and traceability.
  - Logging via Uvicorn for observability.
  - Pydantic serialization for Inngest events and responses.

- **Extensibility**:
  - Custom types (`custom_types.py`) for RAG results (e.g., `RAGQueryResult`, `RAGSearchResult`).
  - Easy to extend for other document types or embedding/LLM providers via LlamaIndex abstractions.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │─── │  FastAPI/Inngest │─── │   Qdrant DB     │
│ (Upload/Query)  │    │ (Workflows)      │    │(Vectors/Payloads)
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │   Groq API       │    │ Sentence         │
                    │ (Llama 3.1 8B)   │    │ Transformers     │
                    └──────────────────┘    └──────────────────┘
                              │                     │
                              ▼                     ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │   Data Loader    │    │ LlamaIndex       │
                    │ (Chunk/Embed)    │◄───│ (PDF Parse/Split)│
                    └──────────────────┘    └──────────────────┘
```

- **Ingestion Flow**: UI → Inngest Event → Load/Chunk (LlamaIndex) → Embed (OpenAI) → Upsert (Qdrant).
- **Query Flow**: UI → Inngest Event → Embed Query (OpenAI) → Search (Qdrant) → Generate (OpenAI) → Return Answer + Sources.

## Tech Stack

- **Frontend/UI**: Streamlit (v1.49.1+)
- **Backend/API**: FastAPI (v0.116.1+), Uvicorn (v0.35.0+)
- **Workflows**: Inngest (v0.5.6+)
- **RAG Framework**: LlamaIndex (core v0.14.0+, file readers v0.5.4+)
- **Vector Database**: Qdrant Client (v1.15.1+)
- **LLM**: Groq API (llama-3.1-8b-instant model)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Utilities**: python-dotenv, Pydantic
- **Python**: >=3.13
- **Project Management**: pip, requirements.txt

## Setup

1. **Prerequisites**:
   - Python 3.13+
   - Install uv: `pip install uv`
   - Start local services:
     - Qdrant: Run the Qdrant server (e.g., via Docker: `docker run -p 6333:6333 qdrant/qdrant`).
     - Inngest: Run the dev server (`npx inngest-cli@latest dev`).

2. **Clone/Navigate**:
   ```
   cd Production-Grade-RAG-App
   ```

3. **Environment**:
   - Create `.env`:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     QDRANT_URL=http://localhost:6333
     QDRANT_HOST=localhost
     INNGEST_API_BASE=http://127.0.0.1:8288/v1
     ```

4. **Install Dependencies**:
   ```
   uv sync
   ```

5. **Run Backend**:
   ```
   uv run uvicorn main:app --reload --port 8000
   ```
   (Inngest workflows will be served here.)

6. **Run UI**:
   ```
   uv run streamlit run streamlit_app.py
   ```
   Access at `http://localhost:8501`.

## Usage

## Running the Services

You can run all services using Docker Compose or manually in separate terminals. Choose the option that best suits your needs:

### Option 1: Using Docker Compose

**Note on Storage Requirements**: 
- The Docker setup requires approximately 6-7GB of storage space due to Python packages and model downloads.
- If storage is a concern, consider using the manual development setup instead.

1. Make sure you have Docker and Docker Compose installed.

2. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Start all services:
   ```bash
   docker compose up --build
   ```

This will start:
- Qdrant on http://localhost:6333
- FastAPI on http://localhost:8000
- Streamlit on http://localhost:8501
- Inngest Dev Server on http://localhost:8288

To clean up and reclaim storage space when done:
```bash
docker compose down
docker system prune --volumes  # This will remove unused containers, networks, and volumes
```

### Option 2: Manual Setup (Development)

Start each service in separate terminals:

#### Terminal 1 - Qdrant Vector Database:
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Terminal 2 - FastAPI Backend:
```bash
uvicorn main:app --reload --port 8000
```

#### Terminal 3 - Inngest Dev Server:
```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

#### Terminal 4 - Streamlit Frontend:
```bash
streamlit run streamlit_app.py
```


1. **Ingest PDFs**:
   - Open the Streamlit app.
   - Upload a PDF in the "Upload a PDF to Ingest" section.
   - The app saves the file, triggers an Inngest event, and processes it asynchronously.
   - Success message confirms ingestion (chunks are embedded and stored).

2. **Query Knowledge Base**:
   - In the "Ask a question about your PDFs" section, enter a question.
   - Adjust top-K (number of chunks to retrieve).
   - Submit: The app triggers a query event, waits for the workflow, and displays the answer with sources.

Example Query: "What is the main topic of the document?" → Answer based on retrieved chunks + sources listed.

## Demo

Watch how the RAG application works in action:

https://github.com/Mohamed-Noufal/production-RAG/raw/main/assets/video.mp4

The video demonstrates:
- PDF document upload and processing
- Real-time document ingestion
- Query processing and response generation
- Context-aware answers with source citations

## Troubleshooting

- **Ingestion Fails**: Check Qdrant is running on port 6333. Verify PDF parsing (non-text PDFs may fail).
- **Query Timeout**: Increase timeout in `wait_for_run_output` or check Inngest logs.
- **Groq API Errors**: Ensure API key is set and valid (should start with 'gsk_').
- **Inngest Issues**: Run `npx inngest-cli@latest dev` and check for event processing.
- **No Contexts Retrieved**: Ensure documents are ingested; check Qdrant collection "docs" exists.
- **Embedding Issues**: Verify Sentence Transformers model is properly downloaded and accessible.

## Future Improvements

- Support multiple document types (e.g., TXT, DOCX via LlamaIndex).
- Hybrid search (keyword + semantic).
- Advanced prompting/chaining for complex queries.
- Authentication and persistent storage.
- Deployment: Dockerize for cloud (e.g., Vercel for Inngest, Render for Qdrant).

## License

MIT License (or specify as needed).
