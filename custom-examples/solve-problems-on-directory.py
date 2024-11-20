from typing import Annotated, List, TypedDict
import operator
import os
import subprocess
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader


# 定义状态类型
class SearchState(TypedDict):
    answer_list: Annotated[List[str], operator.add]  # 答案列表
    current_query: str  # 当前搜索查询
    searched_files: List[str]  # 已搜索过的文件
    plan_list: List[str]  # 计划列表
    max_iteration: int  # 最大迭代次数
    current_iteration: int  # 当前迭代次数
    vectorstore: Chroma  # 向量存储
    target: str  # 搜索目标


# Linux命令工具
def get_file_content(file_path: str) -> str:
    """使用cat命令获取文件内容"""
    try:
        result = subprocess.run(["cat", file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error reading file: {str(e)}"


# 创建规划提示模板
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个任务规划助手。根据当前的答案列表和目标，判断是否需要继续搜索。
    如果需要继续，请提供下一步的搜索查询。
    
    当前目标: {target}
    当前答案列表: {answer_list}
    当前迭代次数: {current_iteration}
    最大迭代次数: {max_iteration}
    
    如果已经找到完整答案或达到最大迭代次数，请回复 FINAL。
    否则，请提供下一步搜索查询。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 创建执行提示模板
executor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个文件分析助手。根据搜索到的文件内容，提供相关答案。
    
    搜索查询: {query}
    目标要回答的问题: {target}
    相关文件内容: {file_contents}
    
    请分析文件内容并提供答案。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 初始化 LLM
llm = ChatOpenAI(model="gemini-1.5-pro-latest")
planner_chain = planner_prompt | llm
executor_chain = executor_prompt | llm


def load_documents(directory: str) -> Chroma:
    """加载目录中的文档并创建向量存储"""
    loader = DirectoryLoader(directory, glob="**/*.*")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore


def planner(state: SearchState):
    """规划节点：判断是否继续搜索，并更新搜索查询"""
    if state["current_iteration"] >= state["max_iteration"]:
        return {"plan_list": ["FINAL"]}

    result = planner_chain.invoke(
        {
            "messages": [HumanMessage(content="请分析当前进度")],
            "target": state["target"],
            "answer_list": state["answer_list"],
            "current_iteration": state["current_iteration"],
            "max_iteration": state["max_iteration"],
        }
    )

    if "FINAL" in result.content:
        return {"plan_list": ["FINAL"]}

    return {
        "current_query": result.content,
        "current_iteration": state["current_iteration"] + 1,
    }


def executor(state: SearchState):
    """执行节点：搜索文件并获取答案"""
    # 使用向量存储搜索相关文档
    results = state["vectorstore"].similarity_search(
        state["current_query"],
        k=20,
        # filter={"source": {"$nin": state["searched_files"]}}
    )

    # 获取文件内容
    file_contents = []
    new_searched_files = []
    for doc in results:
        file_path = doc.metadata["source"]
        if file_path not in state["searched_files"]:
            content = get_file_content(file_path)
            file_contents.append(f"File: {file_path}\n{content}")
            new_searched_files.append(file_path)

    # 分析内容获取答案
    result = executor_chain.invoke(
        {
            "messages": [HumanMessage(content="请分析文件内容")],
            "query": state["current_query"],
            "target": state["target"],
            "file_contents": "\n\n".join(file_contents),
        }
    )

    # 将新答案添加到现有答案列表中
    answer_list = state["answer_list"] + [result.content]
    return {"answer_list": answer_list, "searched_files": new_searched_files}

def should_continue(state: SearchState):
    """判断是否继续搜索"""
    if state["plan_list"] and state["plan_list"][-1] == "FINAL":
        return END
    return "executor"


# 创建工作流图
workflow = StateGraph(SearchState)

# 添加节点
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)

# 添加边
workflow.add_conditional_edges(
    "planner", should_continue, {"executor": "executor", END: END}
)
workflow.add_edge("executor", "planner")
workflow.add_edge(START, "planner")

# 编译图
graph = workflow.compile()


def search_in_directory(directory: str, target: str, max_iteration: int = 3):
    """在指定目录中搜索相关文档"""
    # 加载文档到向量存储
    vectorstore = load_documents(directory)

    # 初始化状态
    initial_state = {
        "answer_list": [],
        "current_query": "",
        "searched_files": [],
        "plan_list": [],
        "max_iteration": max_iteration,
        "current_iteration": 0,
        "target": target,
        "vectorstore": vectorstore,
    }

    # 执行搜索工作流
    final_answers = []
    for event in graph.stream(initial_state):
        if "planner" in event:
            print(f"\n规划迭代 {event['planner'].get('current_iteration', 0)}:")
            print(f"下一步查询: {event['planner'].get('current_query', '')}")
        elif "executor" in event:
            print("\n执行结果:")
            answers = event['executor'].get('answer_list', [])
            final_answers = answers
            print(f"搜索的文件: {event['executor'].get('searched_files', [])}")
    
    # Pretty print final answers
    print("\n最终答案:")
    for i, answer in enumerate(final_answers, 1):
        print(f"\n=== 答案 {i} ===")
        print(answer)

if __name__ == "__main__":
    directory = "/home/michael/ubuntu-repos/docsets/lobe-chat/"
    target = "如何使用 docker 部署带有数据库功能的 Lobe Chat，帮我输出完整的步骤"
    search_in_directory(directory, target)
