from models import GeminiClient

EXTRACT_PROMPT = '''你是一个对比对象提取专家。用户会输入一个包含两个事物对比的句子，请你从中提取出这两个需要对比的事物名称。

要求：
1. 只返回两个事物的名称，用逗号分隔
2. 不要添加任何其他文字、解释或格式
3. 如果句子中有多个对比对象，只提取最主要的两个

示例输入：对比CPU和GPU的差异
示例输出：CPU,GPU

示例输入：请分析一下HTTP和HTTPS的区别
示例输出：HTTP,HTTPS

现在请处理以下输入：
{user_input}'''

class ObjectExtractor:
    def __init__(self):
        self.gemini = GeminiClient()
    
    def extract(self, user_input: str) -> tuple[str, str]:
        prompt = EXTRACT_PROMPT.format(user_input=user_input)
        result = self.gemini.generate_text(prompt)
        result = result.strip()
        objects = [obj.strip() for obj in result.split(',') if obj.strip()]
        
        if len(objects) >= 2:
            return objects[0], objects[1]
        else:
            raise ValueError(f"无法从输入中提取两个对比对象: {user_input}")
