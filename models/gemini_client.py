from google import genai
from typing import Generator, Optional, Callable
from config import GEMINI_API_KEY, GEMINI_MODEL_EXTRACT, GEMINI_MODEL_ANALYSIS
from PIL import Image
from io import BytesIO
import base64

GEMINI_MODEL_IMAGE = "gemini-2.5-flash-image"

class GeminiClient:
    def __init__(self, model_name: str = GEMINI_MODEL_ANALYSIS):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
    def generate_text_stream(
        self, 
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            stream=True
        )
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
        metadata = {'model': self.model_name}
        
        if stream:
            for chunk in self.generate_text_stream(prompt, on_chunk=on_chunk):
                full_text += chunk
        else:
            full_text = self.generate_text(prompt)
            if on_chunk:
                on_chunk(full_text)
        
        return full_text, metadata
    
    def generate_image(
        self, 
        prompt: str,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> tuple[bytes, dict]:
        if on_progress:
            on_progress("正在生成图像...")
        
        response = self.client.models.generate_content(
            model=GEMINI_MODEL_IMAGE,
            contents=prompt
        )
        
        metadata = {'model': GEMINI_MODEL_IMAGE}
        
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                if on_progress:
                    on_progress("图像生成完成！")
                return image_bytes, metadata
        
        raise ValueError("未能从响应中获取图像")
    
    def save_image(self, image_bytes: bytes, output_path: str) -> str:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return output_path
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')
