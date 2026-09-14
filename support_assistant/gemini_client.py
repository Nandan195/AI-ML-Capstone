import os
from google import genai

_client = None

def get_gemini_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _client = genai.Client(api_key=api_key)
    return _client

def generate_gemini_response(prompt: str) -> str:
    """
    Generates a response using Gemini API.
    Raises exception if API key is not set or API call fails.
    """
    client = get_gemini_client()
    if not client:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    # We use a standard model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text
