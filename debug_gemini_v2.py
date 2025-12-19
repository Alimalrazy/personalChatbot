import os
import toml
from google import genai

# Try to load secrets to set env var for the snippet to work as-is
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

# User's snippet starts here
print("--- Starting User Snippet ---")
try:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Explain how AI works in a few words"
    )
    print(response.text)
    print("--- User Snippet Success ---")
except Exception as e:
    print(f"--- User Snippet Failed ---")
    print(e)
