from typing import Annotated, List, TypedDict
import operator
import os
import subprocess
import datetime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START

# 定义状态类型
class SearchState(TypedDict):
    history_search_result_list: Annotated[List[str], operator.add]  # 历史搜索结果列表
    current_command_list: List[str]  # 当前命令列表
    searched_files: List[str]  # 已搜索过的文件
    plan_list: List[str]  # 计划列表
    max_iteration: int  # 最大迭代次数
    current_iteration: int  # 当前迭代次数
    target: str  # 搜索目标
    directory: str  # 搜索目录

# Linux命令工具
def get_file_content(file_path: str) -> str:
    """使用cat命令获取文件内容"""
    try:
        result = subprocess.run(["cat", file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error reading file: {str(e)}"

def execute_ripgrep(command: str) -> List[str]:
    """执行ripgrep命令并返回结果"""
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        return result.stdout.splitlines()
    except Exception as e:
        return []

# 创建规划提示模板
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个任务规划助手。根据当前的搜索结果和目标，判断是否需要继续搜索。
        如果需要继续，请提供下一步的搜索关键词。
        
        当前目标: {target}
        当前搜索结果: {history_search_result_list}
        当前迭代次数: {current_iteration}
        最大迭代次数: {max_iteration}
        
        如果已经找到完整答案或达到最大迭代次数，请回复 FINAL。
        否则，请提供下一步搜索关键词。""",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# 创建执行提示模板
executor_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个命令生成助手。根据搜索关键词，生成一个或多个ripgrep命令。

        搜索目录: {directory}
        搜索关键词: {keyword}
        已搜索文件: {searched_files}
        已搜索文件对应的内容: {history_search_result_list}
        
        请生成ripgrep命令列表，每行一个命令。每个命令最多只搜索 10 个文件。每个命令格式为:
        rg "关键词" 目录路径 --type 文件类型
        
        返回格式示例:
        rg "keyword1" ./path --max-count 10 --type md
        rg "keyword2" ./path --max-count 10 --type py
        rg "keyword3" ./path --max-count 10 --type txt
        """,
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# 初始化 LLM
llm = ChatOpenAI(model="gemini-1.5-pro-latest")
planner_chain = planner_prompt | llm
executor_chain = executor_prompt | llm

def planner(state: SearchState):
    """规划节点：判断是否继续搜索，并更新搜索关键词"""
    if state["current_iteration"] >= state["max_iteration"]:
        # 返回完整的状态更新
        return {
            "plan_list": ["FINAL"],
            "current_iteration": state["current_iteration"],  # 保持当前迭代次数
            "history_search_result_list": state["history_search_result_list"],  # 保持现有结果
            "current_command_list": state["current_command_list"],
            "searched_files": state["searched_files"]
        }

    result = planner_chain.invoke({
        "messages": [HumanMessage(content="请分析当前进度")],
        "target": state["target"],
        "history_search_result_list": state["history_search_result_list"],
        "current_iteration": state["current_iteration"],
        "max_iteration": state["max_iteration"],
    })

    if "FINAL" in result.content:
        # 返回完整的状态更新
        return {
            "plan_list": ["FINAL"],
            "current_iteration": state["current_iteration"],
            "history_search_result_list": state["history_search_result_list"],
            "current_command_list": state["current_command_list"],
            "searched_files": state["searched_files"]
        }

    # 返回完整的状态更新
    return {
        "plan_list": [result.content],
        "current_iteration": state["current_iteration"] + 1,
        "history_search_result_list": state["history_search_result_list"],
        "current_command_list": state["current_command_list"],
        "searched_files": state["searched_files"]
    }

def executor(state: SearchState):
    """执行节点：生成并执行ripgrep命令，获取搜索结果"""
    keyword = state["plan_list"][-1]
    current_command_list = state["current_command_list"].copy()
    
    # 生成ripgrep命令
    result = executor_chain.invoke({
        "messages": [HumanMessage(content="请生成ripgrep命令")],
        "directory": state["directory"],
        "keyword": keyword,
        "searched_files": state["searched_files"],
        "history_search_result_list": state["history_search_result_list"],
    })
    
    # 将返回的命令文本分割成命令列表
    commands = [cmd.strip() for cmd in result.content.strip().split('\n') if cmd.strip()]
    current_command_list.extend(commands)
    
    # 执行所有命令获取结果
    all_search_results = []
    for command in commands:
        search_results = execute_ripgrep(command)
        if not search_results:
            continue
        all_search_results.extend(search_results)
    
    # 把 all_search_results 中相同的文件去重，先过滤出文件路径
    unique_files = list(set(result.split(":")[0] for result in all_search_results))

    # 获取文件内容
    new_files = []
    contents = []
    for file_path in unique_files:
        if file_path not in state["searched_files"]:
            content = get_file_content(file_path)
            contents.append(f"File: {file_path}\n{content}")
            new_files.append(file_path)

    # 返回完整的状态更新
    return {
        "history_search_result_list": state["history_search_result_list"] + contents,
        "searched_files": state["searched_files"] + new_files,
        "current_command_list": current_command_list,
        "plan_list": state["plan_list"],
        "current_iteration": state["current_iteration"]
    }

def should_continue(state: SearchState):
    """判断是否继续搜索"""
    if state["plan_list"] and state["plan_list"][-1] == "FINAL":
        return "writer"
    return "executor"

def writer(state: SearchState):
    """输出节点：将结果写入文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, f"{timestamp}_{state['target'][:30]}.md")
    
    with open(output_file, "w") as f:
        f.write("# Search Results\n\n")
        f.write("## Searched Files\n")
        for file in state["searched_files"]:
            f.write(f"- {file}\n")
        
        f.write("\n## File Contents\n")
        for result in state["history_search_result_list"]:
            f.write(f"\n{result}\n")
    
    return state

# 创建工作流图
workflow = StateGraph(SearchState)

# 添加节点
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("writer", writer)

# 添加边
workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "executor": "executor",
        "writer": "writer"
    }
)
workflow.add_edge("executor", "planner")
workflow.add_edge(START, "planner")
workflow.add_edge("writer", END)

# 编译图
graph = workflow.compile()

def search_in_directory(directory: str, target: str, max_iteration: int = 2):
    """在指定目录中搜索相关文档"""
    # 初始化状态
    initial_state = {
        "history_search_result_list": [],
        "current_command_list": [],
        "searched_files": [],
        "plan_list": [],
        "max_iteration": max_iteration,
        "current_iteration": 0,
        "target": target,
        "directory": directory
    }

    # 执行搜索工作流
    for event in graph.stream(initial_state):
        if "planner" in event:
            print(f"\n规划迭代 {event['planner'].get('current_iteration', 0)}:")
            print(f"下一步计划: {event['planner'].get('plan_list', [])}")
        elif "executor" in event:
            print("\n执行结果:")
            print(f"执行的命令: {event['executor'].get('current_command_list', [])}")
            print(f"搜索的文件: {event['executor'].get('searched_files', [])}")
        elif "writer" in event:
            print("\n输出结果:")
            print(f"输出文件: {event['writer'].get('output_file', '')}")

if __name__ == "__main__":
    directory = "/home/michael/ubuntu-repos/docsets/chroma"
    target = "如何使用 Chroma 向量数据库，只搜索 md 文件即可"
    search_in_directory(directory, target)
