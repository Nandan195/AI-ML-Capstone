from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from .models import QueryRequest, QueryResponse
from .graph import app_graph

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        # Initialize state
        initial_state = {
            "query": request.query,
            "intent": "",
            "retrieved_chunks": [],
            "sources": [],
            "answer": "",
            "confidence": 0.0,
            "error": False
        }
        
        # Run graph
        final_state = app_graph.invoke(initial_state)
        
        # Pydantic validation handles constraints internally if returning this model.
        # But we construct it manually to ensure validation occurs before returning.
        response = QueryResponse(
            answer=final_state["answer"],
            sources=final_state["sources"],
            confidence=final_state["confidence"]
        )
        return response
        
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"Response validation failed: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
