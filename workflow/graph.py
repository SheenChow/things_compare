from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from extractors import ObjectExtractor
from agents import AgentCompareAB, AgentAViewB, AgentBViewA, AgentSummarizer

class CompareState(TypedDict):
    user_input: str
    thing_a: Optional[str]
    thing_b: Optional[str]
    compare_ab_result: Optional[str]
    a_view_b_result: Optional[str]
    b_view_a_result: Optional[str]
    summary_result: Optional[str]

class CompareWorkflow:
    def __init__(self):
        self.object_extractor = ObjectExtractor()
        self.agent_compare_ab = AgentCompareAB()
        self.agent_a_view_b = AgentAViewB()
        self.agent_b_view_a = AgentBViewA()
        self.agent_summarizer = AgentSummarizer()
    
    def extract_objects(self, state: CompareState) -> dict:
        user_input = state["user_input"]
        thing_a, thing_b = self.object_extractor.extract(user_input)
        return {
            "thing_a": thing_a,
            "thing_b": thing_b
        }
    
    def compare_ab(self, state: CompareState) -> dict:
        result = self.agent_compare_ab.run(state["thing_a"], state["thing_b"])
        return {"compare_ab_result": result}
    
    def a_view_b(self, state: CompareState) -> dict:
        result = self.agent_a_view_b.run(state["thing_a"], state["thing_b"])
        return {"a_view_b_result": result}
    
    def b_view_a(self, state: CompareState) -> dict:
        result = self.agent_b_view_a.run(state["thing_a"], state["thing_b"])
        return {"b_view_a_result": result}
    
    def summarize(self, state: CompareState) -> dict:
        result = self.agent_summarizer.run(
            state["thing_a"],
            state["thing_b"],
            state["compare_ab_result"],
            state["a_view_b_result"],
            state["b_view_a_result"]
        )
        return {"summary_result": result}
    
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(CompareState)
        
        workflow.add_node("extract_objects", self.extract_objects)
        workflow.add_node("compare_ab", self.compare_ab)
        workflow.add_node("a_view_b", self.a_view_b)
        workflow.add_node("b_view_a", self.b_view_a)
        workflow.add_node("summarize", self.summarize)
        
        workflow.set_entry_point("extract_objects")
        
        workflow.add_edge("extract_objects", "compare_ab")
        workflow.add_edge("extract_objects", "a_view_b")
        workflow.add_edge("extract_objects", "b_view_a")
        
        workflow.add_edge(["compare_ab", "a_view_b", "b_view_a"], "summarize")
        
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    def run(self, user_input: str) -> dict:
        graph = self.build_graph()
        initial_state = {
            "user_input": user_input,
            "thing_a": None,
            "thing_b": None,
            "compare_ab_result": None,
            "a_view_b_result": None,
            "b_view_a_result": None,
            "summary_result": None
        }
        result = graph.invoke(initial_state)
        return result
