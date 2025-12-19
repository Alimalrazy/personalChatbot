import google.generativeai as genai
import toml
import os

try:
    if os.path.exists(".streamlit/secrets.toml"):
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
        api_key = secrets.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print("Starting model check...")
            for m in genai.list_models():
                print(f"Name: {m.name}")
                print(f"Methods: {m.supported_generation_methods}")
            print("Done.")
        else:
            print("GEMINI_API_KEY not found in secrets.toml")
    else:
        print(".streamlit/secrets.toml not found")
            
except Exception as e:
    print(f"Error: {e}")
