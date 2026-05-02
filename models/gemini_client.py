from google import genai
from google.genai import types
from typing import Generator, Optional, Callable
from config import GEMINI_API_KEY, GEMINI_MODEL_EXTRACT, GEMINI_MODEL_ANALYSIS, GEMINI_MODEL_IMAGE
from PIL import Image
from io import BytesIO
import base64

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
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt
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
    
    def translate_to_english(self, chinese_prompt: str) -> str:
        translate_prompt = f"""请将以下中文图像描述翻译成英文，保持原意不变，适用于AI图像生成。
只返回英文翻译结果，不要添加其他解释。

中文描述：
{chinese_prompt}"""
        
        result = self.generate_text(translate_prompt)
        return result.strip()
    
    def generate_image(
        self, 
        prompt: str,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> tuple[bytes, dict]:
        if on_progress:
            on_progress("正在翻译图像描述为英文...")
        
        english_prompt = self.translate_to_english(prompt)
        
        if on_progress:
            on_progress(f"正在生成图像...")
        
        response = self.client.models.generate_content(
            model=GEMINI_MODEL_IMAGE,
            contents=english_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                thinking_config=types.ThinkingConfig(
                    thinking_level="HIGH",
                    include_thoughts=False
                )
            )
        )
        
        metadata = {'model': GEMINI_MODEL_IMAGE}
        
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'as_image') and part.as_image():
                    image = part.as_image()
                    if image:
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        image_bytes = buf.getvalue()
                        
                        if on_progress:
                            on_progress("图像生成完成！")
                        return image_bytes, metadata
                
                elif hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    if on_progress:
                        on_progress("图像生成完成！")
                    return image_bytes, metadata
        
        text_result = ""
        if hasattr(response, 'text') and response.text:
            text_result = response.text
        
        raise ValueError(f"未能从响应中获取图像。模型返回: {text_result[:200] if text_result else '(无文本)'}")
    
    def save_image(self, image_bytes: bytes, output_path: str) -> str:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return output_path
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')
