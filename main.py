#!/usr/bin/env python3
import sys
from workflow import CompareWorkflow

def print_separator(title: str = ""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n【{title}】\n{line}")
    else:
        print(f"\n{line}")

def main():
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("请输入需要对比的内容（例如：对比CPU和GPU的差异）：")
        user_input = input("> ").strip()
    
    if not user_input:
        print("错误：请输入有效的对比内容")
        sys.exit(1)
    
    print_separator("开始对比分析")
    print(f"用户输入: {user_input}")
    
    workflow = CompareWorkflow()
    
    try:
        result = workflow.run(user_input)
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_separator("提取的对比对象")
    print(f"事物A: {result.get('thing_a', 'N/A')}")
    print(f"事物B: {result.get('thing_b', 'N/A')}")
    
    print_separator("Agent 1 - 系统性对比分析 (A vs B)")
    if result.get('compare_ab_result'):
        print(result['compare_ab_result'])
    else:
        print("(无输出)")
    
    print_separator(f"Agent 2 - {result.get('thing_a', 'A')}视角看{result.get('thing_b', 'B')}")
    if result.get('a_view_b_result'):
        print(result['a_view_b_result'])
    else:
        print("(无输出)")
    
    print_separator(f"Agent 3 - {result.get('thing_b', 'B')}视角看{result.get('thing_a', 'A')}")
    if result.get('b_view_a_result'):
        print(result['b_view_a_result'])
    else:
        print("(无输出)")
    
    print_separator("Agent 4 - 汇总分析")
    if result.get('summary_result'):
        print(result['summary_result'])
    else:
        print("(无输出)")
    
    print_separator()
    print("对比分析完成！")

if __name__ == "__main__":
    main()
