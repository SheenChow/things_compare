from models import GeminiClient
from typing import Callable, Optional, Tuple

IMAGE_GENERATOR_PROMPT = '''你是一位创意视觉设计师，擅长将抽象的对比概念转化为生动的视觉图像。

请根据以下对比分析的核心结论，生成一个详细的中文图像描述提示词，用于AI图像生成。

对比对象 A: {thing_a}
对比对象 B: {thing_b}

核心分析结论:
{summary_text}

请生成一个详细的中文图像描述提示词，要求：
1. 图像应该能够直观体现【{thing_a}】与【{thing_b}】的本质差异
2. 风格：专业、现代、科技感，适合用于报告和演示
3. 构图：左右分屏对比，或中心对称展示
4. 色调：专业商务风格，避免过于花哨
5. 分辨率：高清晰度，适合打印

请仅输出中文图像描述提示词，不要添加其他解释。'''

class AgentImageGenerator:
    def __init__(self):
        self.gemini = GeminiClient()
    
    def get_model_name(self) -> str:
        return self.gemini.get_model_name()
    
    def generate_image_prompt(
        self, 
        thing_a: str, 
        thing_b: str, 
        summary_text: str
    ) -> str:
        prompt = IMAGE_GENERATOR_PROMPT.format(
            thing_a=thing_a,
            thing_b=thing_b,
            summary_text=summary_text[:3000]
        )
        result = self.gemini.generate_text(prompt)
        return result.strip()
    
    def generate_image(
        self, 
        image_prompt: str,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> Tuple[bytes, dict]:
        return self.gemini.generate_image(image_prompt, on_progress=on_progress)
    
    def run(
        self,
        thing_a: str,
        thing_b: str,
        summary_text: str,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> Tuple[bytes, str, dict]:
        if on_progress:
            on_progress("正在生成图像描述...")
        
        image_prompt = self.generate_image_prompt(thing_a, thing_b, summary_text)
        
        if on_progress:
            on_progress(f"图像描述已生成，正在绘制图像...")
        
        image_bytes, metadata = self.generate_image(image_prompt, on_progress=on_progress)
        
        return image_bytes, image_prompt, metadata
