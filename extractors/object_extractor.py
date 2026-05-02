from models import GeminiClient
from config import GEMINI_MODEL_EXTRACT

EXTRACT_PROMPT = '''你是一个对比对象提取专家。用户会输入一个包含两个事物对比的句子，请你从中提取出这两个**完整的对比对象**。

重要规则：
1. 不要只提取简单的名词，要提取完整的对比主体（包括关键的限定词、修饰词、领域等）
2. 如果对比的是两个主体在某个领域/方面的差异，请把领域/方面也包含进去
3. 只返回两个对比对象，用逗号分隔
4. 不要添加任何其他文字、解释或格式

示例分析：
- 输入：分析一下中国和美国对AI发展的路径和思路的差异
- 错误提取：中国,美国 ❌（缺少关键限定词）
- 正确提取：中国的AI发展路径,美国的AI发展路径 ✅

- 输入：对比一下深度学习和机器学习在图像处理中的效果差异
- 错误提取：深度学习,机器学习 ❌（缺少应用场景）
- 正确提取：深度学习在图像处理中的应用,机器学习在图像处理中的应用 ✅

- 输入：CPU和GPU在游戏中的性能差异
- 错误提取：CPU,GPU ❌
- 正确提取：CPU在游戏中的性能,GPU在游戏中的性能 ✅

- 输入：对比CPU和GPU的差异
- 简单提取：CPU,GPU ✅（没有限定词时可以简化）

现在请处理以下输入：
{user_input}'''

class ObjectExtractor:
    def __init__(self):
        self.gemini = GeminiClient(model_name=GEMINI_MODEL_EXTRACT)
    
    def get_model_name(self) -> str:
        return self.gemini.get_model_name()
    
    def extract(self, user_input: str) -> tuple[str, str]:
        prompt = EXTRACT_PROMPT.format(user_input=user_input)
        result = self.gemini.generate_text(prompt)
        result = result.strip()
        objects = [obj.strip() for obj in result.split(',') if obj.strip()]
        
        if len(objects) >= 2:
            return objects[0], objects[1]
        else:
            raise ValueError(f"无法从输入中提取两个对比对象: {user_input}")
