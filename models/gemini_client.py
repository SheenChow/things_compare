import google.generativeai as genai
from typing import Generator, Optional, Callable
from config import GEMINI_API_KEY, GEMINI_MODEL

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_text_stream(
        self, 
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        response = self.model.generate_content(prompt, stream=True)
        full_text = ""
        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                if on_chunk:
                    on_chunk(chunk.text)
                yield chunk.text
        return full_text
    
    def generate_text_with_full_result(
        self, 
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        stream: bool = True
    ) -> tuple[str, dict]:
        full_text = ""
        metadata = {}
        
        if stream:
            for chunk in self.generate_text_stream(prompt, on_chunk=on_chunk):
                full_text += chunk
        else:
            full_text = self.generate_text(prompt)
            if on_chunk:
                on_chunk(full_text)
        
        return full_text, metadata
