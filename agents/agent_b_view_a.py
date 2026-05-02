from models import GeminiClient

B_VIEW_A_PROMPT = '''请你站在【{thing_b}】的角度来看待【{thing_a}】。

作为{thing_b}，请思考：
1. 你如何看待{thing_a}的存在？它是你的竞争对手、补充者，还是替代者？
2. {thing_a}有哪些优势是你羡慕或认可的？
3. {thing_a}有哪些缺点或局限性是你认为它无法解决的？
4. 在哪些场景下，你认为{thing_a}确实比你更适合？
5. 你认为{thing_a}未来会如何发展？会对你构成威胁吗？
6. 你觉得自己和{thing_a}之间最大的差异和共同点是什么？

请以{thing_b}的第一人称视角来回答，语气要生动、有个性，像一个有思想的个体在表达自己的观点。'''

class AgentBViewA:
    def __init__(self):
        self.gemini = GeminiClient()
    
    def run(self, thing_a: str, thing_b: str) -> str:
        prompt = B_VIEW_A_PROMPT.format(thing_a=thing_a, thing_b=thing_b)
        return self.gemini.generate_text(prompt)
