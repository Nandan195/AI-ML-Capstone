# Zepto Support Assistant (Module 3)

This module implements a full Retrieval-Augmented Generation (RAG) backend utilizing LangGraph, ChromaDB, sentence-transformers, and FastAPI.

## Architecture

1. **INGESTION (`ingestion.py`)**: Reads the 8 policy text files from `docs/`.
2. **CHUNKING (`ingestion.py`)**: Splits documents line-by-line (each document acts as its own chunk for these small files).
3. **EMBEDDING (`embeddings.py`)**: Uses local `sentence-transformers` (`all-MiniLM-L6-v2`) to embed chunks natively.
4. **CHROMADB (`vector_store.py`)**: Stores embeddings in a local persistent ChromaDB collection using Cosine Similarity.
5. **RETRIEVAL (`graph.py:retrieve_and_answer`)**: Queries the top 3 similar chunks for policy questions.
6. **GENERATION (`graph.py` & `gemini_client.py`)**: 
   - Uses `MOCK_LLM=1` by default for local offline mock-generation.
   - Can use optional `MOCK_LLM=0` to route to Google Gemini (SDK) using the prompt template in `prompts.py`.
7. **PYDANTIC RESPONSE (`models.py`)**: Enforces the `{answer: str, sources: list[str], confidence: float}` structure.
8. **FASTAPI (`main.py`)**: Exposes the `POST /ask` endpoint utilizing the compiled LangGraph execution graph.

### Mock Mode vs Real Mode
The application reads the `MOCK_LLM` environment variable. 
- **MOCK_LLM=1 or Unset**: The application operates in Mock mode. It uses keyword-based intent classification in `classify_intent` and returns answers directly in python code inside `retrieve_and_answer` or `direct_answer`. No external LLM calls are made.
- **MOCK_LLM=0**: The application attempts to use Gemini 2.5 Flash via `gemini_client.py`. It requires `GEMINI_API_KEY` to be exported in the environment. It leverages the prompt in `prompts.py`. A 3-attempt retry loop is included to ensure output constraints are respected.

### Docker Status
The `Dockerfile` is fully implemented to build the FastAPI application on port 8000. However, the Docker Engine was unavailable on the local machine during verification, so it could not be actively run.

## Actual Captured Test Responses (Mock Mode)

**TEST: Policy Question (Triggers Retrieval)**
**Query:** "What is the standard delivery fee for orders below INR 149?"
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
  "sources": [
    "doc_01.txt_chunk_0"
  ],
  "confidence": 1.0
}
```

**TEST: General Question (Direct Answer)**
**Query:** "Tell me a joke."
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
