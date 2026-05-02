from models import GeminiClient

SUMMARIZER_PROMPT = '''你是一位资深的对比分析专家。以下是关于【{thing_a}】和【{thing_b}】的多角度对比分析结果，请你将这些内容进行综合汇总，提炼出最有价值的洞察。

================ 【Agent 1】系统性对比结果 ================
{compare_ab_result}

================ 【Agent 2】{thing_a}视角看{thing_b} ================
{a_view_b_result}

================ 【Agent 3】{thing_b}视角看{thing_a} ================
{b_view_a_result}

================ 请你进行汇总 ================

请按照以下结构输出汇总结果：

一、核心洞察（3-5条最关键的发现）

二、本质差异总结
（用最精炼的语言概括两者的根本区别）

三、场景选择指南
- 什么时候选择{thing_a}？
- 什么时候选择{thing_b}？
- 什么时候两者配合使用？

四、未来展望
- 两者的发展趋势如何？
- 在AI时代的地位变化？

输出要求：
- 语言精炼、有洞察、不冗余
- 保留各Agent中的独特观点和精彩见解
- 形成一个统一、连贯的整体分析'''

class AgentSummarizer:
    def __init__(self):
        self.gemini = GeminiClient()
    
    def run(self, thing_a: str, thing_b: str, 
            compare_ab_result: str, 
            a_view_b_result: str, 
            b_view_a_result: str) -> str:
        prompt = SUMMARIZER_PROMPT.format(
            thing_a=thing_a,
            thing_b=thing_b,
            compare_ab_result=compare_ab_result,
            a_view_b_result=a_view_b_result,
            b_view_a_result=b_view_a_result
        )
        return self.gemini.generate_text(prompt)
