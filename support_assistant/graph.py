import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from .vector_store import retrieve_top_k
from .gemini_client import generate_gemini_response
from .prompts import RAG_PROMPT_TEMPLATE
import json

class SupportState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    sources: List[str]
    answer: str
    confidence: float
    error: bool

def is_mock_mode() -> bool:
    val = os.environ.get("MOCK_LLM", "1").strip()
    return val == "1" or val == ""

def classify_intent(state: SupportState) -> SupportState:
    query_lower = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours", "policy", "fee", "damaged", "grocery", "pass", "support", "delayed", "phone", "order"]
    
    intent = "general_question"
    for kw in keywords:
        if kw in query_lower:
            intent = "policy_question"
            break
            
    state["intent"] = intent
    return state

def direct_answer(state: SupportState) -> SupportState:
    state["sources"] = []
    state["confidence"] = 1.0
    
    if is_mock_mode():
        state["answer"] = "I can only answer questions about Zepto policies right now."
    else:
        try:
            state["answer"] = generate_gemini_response(f"Answer this general question concisely: {state['query']}")
        except Exception as e:
            state["answer"] = f"Error calling Gemini: {e}"
            state["error"] = True
            
    return state

def retrieve_and_answer(state: SupportState) -> SupportState:
    results = retrieve_top_k(state["query"], k=3)
    
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    ids = results["ids"][0] if results["ids"] else []
    
    state["retrieved_chunks"] = docs
    
    if not docs:
        state["answer"] = "No relevant policies found."
        state["sources"] = []
        state["confidence"] = 0.0
        return state
        
    best_doc = docs[0]
    best_id = ids[0]
    best_source = metas[0].get("source", best_id)
    
    state["sources"] = [best_id]
    state["confidence"] = 1.0
    
    if is_mock_mode():
        state["answer"] = f"Based on the retrieved context: {best_doc}"
    else:
        # Build context from all retrieved docs
        context_str = "\n".join([f"Source {metas[i]['source']}: {docs[i]}" for i in range(len(docs))])
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, query=state["query"])
        
        # Retry logic up to 2 additional times (3 total tries)
        max_tries = 3
        success = False
        
        for attempt in range(max_tries):
            try:
                if attempt > 0:
                    prompt += f"\n[CORRECTIVE INSTRUCTION]: Ensure the answer relies ONLY on context and is under 100 words."
                
                llm_response = generate_gemini_response(prompt)
                
                # Basic validation
                if len(llm_response.split()) > 150:
                    raise ValueError("Response is too long.")
                if "do not know" in llm_response.lower() and len(llm_response) < 50:
                    # Valid refusal
                    pass
                
                state["answer"] = llm_response
                success = True
                break
            except Exception as e:
                continue
                
        if not success:
            state["answer"] = "[Error Response]: Failed to generate a valid grounded response after 3 attempts."
            state["error"] = True
            state["confidence"] = 0.0
            
    return state

def route_intent(state: SupportState) -> str:
    return state["intent"]

# Build the Graph
workflow = StateGraph(SupportState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()
