from google import genai
from src.config import config

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=config.api_key)

    def generate_response(self, system_prompt: str, user_query: str, model: str = "gemini-2.5-flash") -> str:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=[system_prompt + "\n\n" + user_query]
            )
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error. Please try again. {str(e)}"
