import os
import toml
from google import genai

# Load secrets
try:
    if os.path.exists(".streamlit/secrets.toml"):
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
        api_key = secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            print("Loaded API key from secrets.toml")
except Exception as e:
    print(f"Warning: Could not load secrets: {e}")

# Test Embedding
print("--- Starting Embedding Test ---")
try:
    client = genai.Client()
    
    text = "Hello world"
    # Using the model name from logic.py
    model = "text-embedding-004" 
    
    print(f"Attempting to embed using {model}...")
    result = client.models.embed_content(
        model=model,
        contents=text
    )
    
    # Check if we got embeddings
    # The SDK structure might be different, let's inspect result
    print("Result type:", type(result))
    if hasattr(result, 'embeddings'):
        vals = result.embeddings[0].values
        print(f"Success! Got embedding vector of length {len(vals)}")
    else:
        print("Success? But structure is unexpected:", result)

    print("--- Embedding Test Success ---")

except Exception as e:
    print(f"--- Embedding Test Failed ---")
    print(e)
