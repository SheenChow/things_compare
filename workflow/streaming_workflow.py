import json
import base64
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO

from extractors import ObjectExtractor
from agents import (
    AgentCompareAB, 
    AgentAViewB, 
    AgentBViewA, 
    AgentSummarizer
)
from agents.agent_compare_ab import COMPARE_AB_PROMPT
from agents.agent_a_view_b import A_VIEW_B_PROMPT
from agents.agent_b_view_a import B_VIEW_A_PROMPT
from agents.agent_summarizer import SUMMARIZER_PROMPT
from agents.agent_image_generator import AgentImageGenerator, IMAGE_GENERATOR_PROMPT
from config import GEMINI_MODEL_EXTRACT, GEMINI_MODEL_IMAGE

@dataclass
class StepEvent:
    step: str
    step_name: str
    status: str
    content: Optional[str] = None
    prompt: Optional[str] = None
    full_result: Optional[str] = None
    error: Optional[str] = None
    model_name: Optional[str] = None
    image_base64: Optional[str] = None
    image_prompt: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"

class StreamingCompareWorkflow:
    STEP_EXTRACT = "extract"
    STEP_COMPARE_AB = "compare_ab"
    STEP_A_VIEW_B = "a_view_b"
    STEP_B_VIEW_A = "b_view_a"
    STEP_SUMMARIZE = "summarize"
    STEP_GENERATE_IMAGE = "generate_image"
    
    STEP_NAMES = {
        STEP_EXTRACT: "提取对比对象",
        STEP_COMPARE_AB: "系统性对比分析 (Agent 1)",
        STEP_A_VIEW_B: "A视角看B (Agent 2)",
        STEP_B_VIEW_A: "B视角看A (Agent 3)",
        STEP_SUMMARIZE: "汇总分析 (Agent 4)",
        STEP_GENERATE_IMAGE: "生成可视化图像",
    }
    
    def __init__(self):
        self.object_extractor = ObjectExtractor()
        self.agent_compare_ab = AgentCompareAB()
        self.agent_a_view_b = AgentAViewB()
        self.agent_b_view_a = AgentBViewA()
        self.agent_summarizer = AgentSummarizer()
        self.agent_image_generator = AgentImageGenerator()
        
        self.results: Dict[str, Any] = {}
        self.debug_info: Dict[str, Any] = {}
        self.image_bytes: Optional[bytes] = None
        self.image_prompt_generated: Optional[str] = None
    
    def _send_event(
        self, 
        step: str, 
        status: str, 
        content: Optional[str] = None,
        prompt: Optional[str] = None,
        full_result: Optional[str] = None,
        error: Optional[str] = None,
        model_name: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_prompt: Optional[str] = None,
        on_event: Optional[Callable[[StepEvent], None]] = None
    ):
        event = StepEvent(
            step=step,
            step_name=self.STEP_NAMES.get(step, step),
            status=status,
            content=content,
            prompt=prompt,
            full_result=full_result,
            error=error,
            model_name=model_name,
            image_base64=image_base64,
            image_prompt=image_prompt
        )
        if on_event:
            on_event(event)
        return event
    
    def run_streaming(
        self, 
        user_input: str, 
        on_event: Optional[Callable[[StepEvent], None]] = None,
        include_debug: bool = True
    ) -> Dict[str, Any]:
        self.results = {
            "user_input": user_input,
            "thing_a": None,
            "thing_b": None,
            "compare_ab_result": None,
            "a_view_b_result": None,
            "b_view_a_result": None,
            "summary_result": None,
        }
        self.debug_info = {}
        self.image_bytes = None
        self.image_prompt_generated = None
        
        extract_model_name = self.object_extractor.get_model_name()
        
        self._send_event(
            self.STEP_EXTRACT, "start", 
            content=f"正在解析输入: \"{user_input}\"",
            model_name=extract_model_name,
            on_event=on_event
        )
        
        try:
            extract_prompt = f"用户输入: {user_input}\n\n使用ObjectExtractor提取两个对比对象"
            thing_a, thing_b = self.object_extractor.extract(user_input)
            
            self.results["thing_a"] = thing_a
            self.results["thing_b"] = thing_b
            
            extract_content = f"成功提取对比对象:\n- 事物A: {thing_a}\n- 事物B: {thing_b}"
            
            self._send_event(
                self.STEP_EXTRACT, "complete",
                content=extract_content,
                full_result=f"{thing_a},{thing_b}",
                prompt=extract_prompt if include_debug else None,
                model_name=extract_model_name,
                on_event=on_event
            )
            
            self.debug_info[self.STEP_EXTRACT] = {
                "prompt": extract_prompt,
                "result": f"{thing_a},{thing_b}",
                "model": extract_model_name
            }
            
        except Exception as e:
            self._send_event(
                self.STEP_EXTRACT, "error",
                error=str(e),
                model_name=extract_model_name,
                on_event=on_event
            )
            raise
        
        compare_prompt = COMPARE_AB_PROMPT.format(thing_a=thing_a, thing_b=thing_b)
        self._run_agent_streaming(
            step=self.STEP_COMPARE_AB,
            prompt=compare_prompt,
            result_key="compare_ab_result",
            agent_fn=lambda on_chunk: self.agent_compare_ab.gemini.generate_text_with_full_result(
                compare_prompt, on_chunk=on_chunk, stream=True
            ),
            model_name=self.agent_compare_ab.gemini.get_model_name(),
            on_event=on_event,
            include_debug=include_debug
        )
        
        a_view_b_prompt = A_VIEW_B_PROMPT.format(thing_a=thing_a, thing_b=thing_b)
        self._run_agent_streaming(
            step=self.STEP_A_VIEW_B,
            prompt=a_view_b_prompt,
            result_key="a_view_b_result",
            agent_fn=lambda on_chunk: self.agent_a_view_b.gemini.generate_text_with_full_result(
                a_view_b_prompt, on_chunk=on_chunk, stream=True
            ),
            model_name=self.agent_a_view_b.gemini.get_model_name(),
            on_event=on_event,
            include_debug=include_debug
        )
        
        b_view_a_prompt = B_VIEW_A_PROMPT.format(thing_a=thing_a, thing_b=thing_b)
        self._run_agent_streaming(
            step=self.STEP_B_VIEW_A,
            prompt=b_view_a_prompt,
            result_key="b_view_a_result",
            agent_fn=lambda on_chunk: self.agent_b_view_a.gemini.generate_text_with_full_result(
                b_view_a_prompt, on_chunk=on_chunk, stream=True
            ),
            model_name=self.agent_b_view_a.gemini.get_model_name(),
            on_event=on_event,
            include_debug=include_debug
        )
        
        summarize_prompt = SUMMARIZER_PROMPT.format(
            thing_a=thing_a,
            thing_b=thing_b,
            compare_ab_result=self.results["compare_ab_result"],
            a_view_b_result=self.results["a_view_b_result"],
            b_view_a_result=self.results["b_view_a_result"]
        )
        self._run_agent_streaming(
            step=self.STEP_SUMMARIZE,
            prompt=summarize_prompt,
            result_key="summary_result",
            agent_fn=lambda on_chunk: self.agent_summarizer.gemini.generate_text_with_full_result(
                summarize_prompt, on_chunk=on_chunk, stream=True
            ),
            model_name=self.agent_summarizer.gemini.get_model_name(),
            on_event=on_event,
            include_debug=include_debug
        )
        
        self._run_image_generation(
            thing_a=thing_a,
            thing_b=thing_b,
            summary_text=self.results["summary_result"],
            on_event=on_event,
            include_debug=include_debug
        )
        
        return {
            "results": self.results,
            "debug_info": self.debug_info,
            "image_bytes": self.image_bytes,
            "image_prompt": self.image_prompt_generated
        }
    
    def _run_agent_streaming(
        self,
        step: str,
        prompt: str,
        result_key: str,
        agent_fn,
        model_name: str,
        on_event: Optional[Callable[[StepEvent], None]] = None,
        include_debug: bool = True
    ):
        full_content = ""
        
        def on_chunk(chunk: str):
            nonlocal full_content
            full_content += chunk
            self._send_event(
                step, "streaming",
                content=chunk,
                model_name=model_name,
                on_event=on_event
            )
        
        self._send_event(
            step, "start",
            content=f"正在执行 {self.STEP_NAMES[step]}...",
            prompt=prompt if include_debug else None,
            model_name=model_name,
            on_event=on_event
        )
        
        try:
            result, metadata = agent_fn(on_chunk)
            self.results[result_key] = result
            
            self._send_event(
                step, "complete",
                full_result=result,
                model_name=model_name,
                on_event=on_event
            )
            
            self.debug_info[step] = {
                "prompt": prompt,
                "result": result,
                "model": model_name
            }
            
        except Exception as e:
            self._send_event(
                step, "error",
                error=str(e),
                model_name=model_name,
                on_event=on_event
            )
            raise
    
    def _run_image_generation(
        self,
        thing_a: str,
        thing_b: str,
        summary_text: str,
        on_event: Optional[Callable[[StepEvent], None]] = None,
        include_debug: bool = True
    ):
        self._send_event(
            self.STEP_GENERATE_IMAGE, "start",
            content="正在准备生成可视化图像...",
            model_name=GEMINI_MODEL_IMAGE,
            on_event=on_event
        )
        
        try:
            def on_progress(msg: str):
                self._send_event(
                    self.STEP_GENERATE_IMAGE, "streaming",
                    content=msg,
                    model_name=GEMINI_MODEL_IMAGE,
                    on_event=on_event
                )
            
            image_bytes, image_prompt, metadata = self.agent_image_generator.run(
                thing_a=thing_a,
                thing_b=thing_b,
                summary_text=summary_text or "",
                on_progress=on_progress
            )
            
            self.image_bytes = image_bytes
            self.image_prompt_generated = image_prompt
            
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            self._send_event(
                self.STEP_GENERATE_IMAGE, "complete",
                content="图像生成完成！",
                image_base64=image_base64,
                image_prompt=image_prompt,
                model_name=GEMINI_MODEL_IMAGE,
                prompt=image_prompt if include_debug else None,
                on_event=on_event
            )
            
            self.debug_info[self.STEP_GENERATE_IMAGE] = {
                "prompt": image_prompt,
                "result": "图像已生成",
                "model": GEMINI_MODEL_IMAGE
            }
            
        except Exception as e:
            self._send_event(
                self.STEP_GENERATE_IMAGE, "error",
                error=f"图像生成失败: {str(e)}",
                model_name=GEMINI_MODEL_IMAGE,
                on_event=on_event
            )
    
    def get_image_bytes(self) -> Optional[bytes]:
        return self.image_bytes
    
    def get_image_prompt(self) -> Optional[str]:
        return self.image_prompt_generated
    
    def generate_markdown(self) -> str:
        results = self.results
        
        md_lines = [
            f"# 事物对比分析报告",
            f"",
            f"> 分析对象: {results.get('thing_a', 'N/A')} vs {results.get('thing_b', 'N/A')}",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## 1. 系统性对比分析 (Agent 1)",
            f"",
            results.get('compare_ab_result', '(无结果)'),
            f"",
            f"---",
            f"",
            f"## 2. {results.get('thing_a', 'A')} 视角看 {results.get('thing_b', 'B')} (Agent 2)",
            f"",
            results.get('a_view_b_result', '(无结果)'),
            f"",
            f"---",
            f"",
            f"## 3. {results.get('thing_b', 'B')} 视角看 {results.get('thing_a', 'A')} (Agent 3)",
            f"",
            results.get('b_view_a_result', '(无结果)'),
            f"",
            f"---",
            f"",
            f"## 4. 汇总分析 (Agent 4)",
            f"",
            results.get('summary_result', '(无结果)'),
            f"",
        ]
        
        if self.image_prompt_generated:
            md_lines.extend([
                f"---",
                f"",
                f"## 5. 可视化图像",
                f"",
                f"**图像描述提示词:**",
                f"",
                f"{self.image_prompt_generated}",
                f"",
                f"> 注：图像已生成，将在PDF报告中展示",
                f"",
            ])
        
        if self.debug_info:
            md_lines.extend([
                f"---",
                f"",
                f"## 附录: 调试信息",
                f"",
            ])
            
            for step, info in self.debug_info.items():
                step_name = self.STEP_NAMES.get(step, step)
                model = info.get('model', '未知')
                md_lines.extend([
                    f"### {step_name}",
                    f"",
                    f"**使用模型:** {model}",
                    f"",
                    f"**Prompt:**",
                    f"```",
                    info.get('prompt', '')[:2000] + ('...' if len(info.get('prompt', '')) > 2000 else ''),
                    f"```",
                    f"",
                    f"**结果 (摘要):**",
                    f"```",
                    info.get('result', '')[:500] + ('...' if len(info.get('result', '')) > 500 else ''),
                    f"```",
                    f"",
                ])
        
        return "\n".join(md_lines)
    
    def generate_pdf(self, output_path: str) -> str:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
            from reportlab.lib.units import inch
            from reportlab.lib.colors import HexColor
            
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=20,
                textColor=HexColor('#667eea')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceBefore=15,
                spaceAfter=10,
                textColor=HexColor('#4a5568')
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                leading=18,
                spaceAfter=10,
                textColor=HexColor('#2d3748')
            )
            
            story = []
            
            results = self.results
            
            story.append(Paragraph("事物对比分析报告", title_style))
            story.append(Spacer(1, 10))
            
            info_text = f"""
            <b>分析对象:</b> {results.get('thing_a', 'N/A')} vs {results.get('thing_b', 'N/A')}<br/>
            <b>生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            story.append(Paragraph(info_text, normal_style))
            story.append(Spacer(1, 20))
            
            sections = [
                ("系统性对比分析", self.results.get('compare_ab_result')),
                (f"{self.results.get('thing_a', 'A')} 视角看 {self.results.get('thing_b', 'B')}", self.results.get('a_view_b_result')),
                (f"{self.results.get('thing_b', 'B')} 视角看 {self.results.get('thing_a', 'A')}", self.results.get('b_view_a_result')),
                ("汇总分析", self.results.get('summary_result')),
            ]
            
            for title, content in sections:
                if content:
                    story.append(Paragraph(title, heading_style))
                    story.append(Spacer(1, 5))
                    
                    paragraphs = content.split('\n')
                    for para in paragraphs:
                        if para.strip():
                            para = para.replace('#', '')
                            para = para.replace('*', '')
                            story.append(Paragraph(para.strip(), normal_style))
                    
                    story.append(Spacer(1, 15))
            
            if self.image_bytes:
                story.append(PageBreak())
                story.append(Paragraph("可视化图像", heading_style))
                story.append(Spacer(1, 10))
                
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(BytesIO(self.image_bytes))
                    
                    max_width = 6 * inch
                    max_height = 8 * inch
                    
                    img_width, img_height = img.size
                    ratio = min(max_width / img_width, max_height / img_height)
                    new_width = img_width * ratio
                    new_height = img_height * ratio
                    
                    temp_img_path = output_path + '_temp.png'
                    img.save(temp_img_path)
                    
                    img_platypus = Image(temp_img_path, width=new_width, height=new_height)
                    story.append(img_platypus)
                    
                    try:
                        import os
                        os.remove(temp_img_path)
                    except:
                        pass
                        
                except Exception as e:
                    story.append(Paragraph(f"图像展示失败: {str(e)}", normal_style))
            
            doc.build(story)
            return output_path
            
        except ImportError:
            raise ImportError("请安装 reportlab 库: pip install reportlab")
