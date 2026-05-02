from google import genai
from google.genai import types
from typing import Generator, Optional, Callable
from config import GEMINI_API_KEY, GEMINI_MODEL_EXTRACT, GEMINI_MODEL_ANALYSIS, GEMINI_MODEL_IMAGE
from PIL import Image
from io import BytesIO
import base64
import time
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

class GeminiClient:
    def __init__(self, model_name: str = GEMINI_MODEL_ANALYSIS):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name
        logger.info(f"GeminiClient 初始化，模型: {self.model_name}")
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _retry_on_error(self, func, func_name: str = "unknown", *args, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"[{func_name}] 尝试调用 (第 {attempt + 1}/{MAX_RETRIES} 次)")
                result = func(*args, **kwargs)
                logger.info(f"[{func_name}] 调用成功")
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"[{func_name}] 调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {error_msg[:200]}")
                
                if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    logger.error(f"[{func_name}] 配额限制错误，停止重试")
                    break
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (attempt + 1)
                    logger.info(f"[{func_name}] 等待 {delay} 秒后重试...")
                    time.sleep(delay)
        
        logger.error(f"[{func_name}] 重试 {MAX_RETRIES} 次后仍然失败: {last_error}")
        raise last_error
    
    def generate_text(self, prompt: str) -> str:
        prompt_length = len(prompt)
        logger.info(f"[generate_text] Prompt 长度: {prompt_length} 字符 (前100: {prompt[:100]}...)")
        
        def _call():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            result = response.text
            logger.info(f"[generate_text] 结果长度: {len(result)} 字符")
            return result
        
        return self._retry_on_error(_call, "generate_text")
    
    def generate_text_stream(
        self, 
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        prompt_length = len(prompt)
        logger.info(f"[generate_text_stream] Prompt 长度: {prompt_length} 字符 (前100: {prompt[:100]}...)")
        
        try:
            logger.info(f"[generate_text_stream] 开始流式调用...")
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt
            )
            full_text = ""
            chunk_count = 0
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    chunk_count += 1
                    if on_chunk:
                        on_chunk(chunk.text)
                    yield chunk.text
            
            logger.info(f"[generate_text_stream] 完成，共 {chunk_count} 个 chunk，总长度: {len(full_text)} 字符")
            return full_text
            
        except Exception as e:
            logger.warning(f"[generate_text_stream] 流式调用失败: {e}")
            logger.info(f"[generate_text_stream] 回退到非流式调用...")
            try:
                result = self.generate_text(prompt)
                if on_chunk:
                    on_chunk(result)
                yield result
                return result
            except Exception as e2:
                logger.error(f"[generate_text_stream] 非流式调用也失败: {e2}")
                raise e
    
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
        prompt_length = len(prompt)
        logger.info(f"[generate_image] Prompt 长度: {prompt_length} 字符")
        logger.info(f"[generate_image] 模型: {GEMINI_MODEL_IMAGE}")
        logger.info(f"[generate_image] Prompt: {prompt[:300]}...")
        
        if on_progress:
            on_progress("正在准备生成图像...")
        
        def _generate():
            logger.info(f"[generate_image] 配置 response_modalities=['IMAGE', 'TEXT']")
            
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
            
            logger.info(f"[generate_image] 调用 generate_content...")
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_IMAGE,
                contents=prompt,
                config=config
            )
            
            logger.info(f"[generate_image] 收到响应，检查响应结构...")
            logger.info(f"[generate_image] response 类型: {type(response)}")
            
            if hasattr(response, 'parts') and response.parts:
                logger.info(f"[generate_image] 响应有 {len(response.parts)} 个 parts")
                
                for i, part in enumerate(response.parts):
                    logger.info(f"[generate_image] 检查 part {i}:")
                    logger.info(f"  - part 类型: {type(part)}")
                    logger.info(f"  - hasattr as_image: {hasattr(part, 'as_image')}")
                    
                    if hasattr(part, 'as_image'):
                        try:
                            image = part.as_image()
                            if image:
                                logger.info(f"[generate_image] 通过 as_image() 获取到图像: {type(image)}")
                                buf = BytesIO()
                                image.save(buf, format='PNG')
                                image_bytes = buf.getvalue()
                                logger.info(f"[generate_image] 图像大小: {len(image_bytes)} 字节")
                                
                                if on_progress:
                                    on_progress("图像生成完成！")
                                return image_bytes, {'model': GEMINI_MODEL_IMAGE}
                        except Exception as e:
                            logger.warning(f"[generate_image] as_image() 调用失败: {e}")
                    
                    if hasattr(part, 'inline_data') and part.inline_data:
                        logger.info(f"[generate_image] 通过 inline_data 获取到图像数据")
                        image_bytes = part.inline_data.data
                        if image_bytes:
                            logger.info(f"[generate_image] 图像大小: {len(image_bytes)} 字节")
                            if on_progress:
                                on_progress("图像生成完成！")
                            return image_bytes, {'model': GEMINI_MODEL_IMAGE}
                    
                    if hasattr(part, 'text') and part.text:
                        logger.info(f"[generate_image] part {i} 包含文本: {part.text[:200]}...")
            
            if hasattr(response, 'text') and response.text:
                logger.warning(f"[generate_image] 响应没有图像，只有文本: {response.text[:300]}")
                raise ValueError(f"模型返回了文本而非图像，可能是内容被过滤。返回内容: {response.text[:300]}")
            
            logger.error("[generate_image] 响应中没有找到图像或文本")
            raise ValueError("未能从响应中获取图像或文本，响应格式异常")
        
        return self._retry_on_error(_generate, "generate_image")
    
    def save_image(self, image_bytes: bytes, output_path: str) -> str:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"[save_image] 图像已保存到: {output_path}")
        return output_path
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')
