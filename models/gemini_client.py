import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
