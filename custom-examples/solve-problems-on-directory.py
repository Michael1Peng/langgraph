from typing import Annotated, List, TypedDict
import operator
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader

# 定义状态类型
class SearchState(TypedDict):
    messages: Annotated[List, operator.add]  # 消息历史
    query: str  # 搜索查询
    iteration: int  # 当前迭代次数
    results: List[dict]  # 搜索结果
    vectorstore: Chroma  # 向量存储

# 创建搜索提示模板
search_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个文件搜索助手。根据用户的查询,帮助找到最相关的文件。
    当前迭代次数: {iteration}
    请分析已找到的文件,并提供下一步搜索建议。
    如果已经找到足够相关的文件或达到最大迭代次数,请回复 FINAL ANSWER。"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 初始化 LLM
llm = ChatOpenAI(model="chatgpt-4o-latest")
search_chain = search_prompt | llm

def load_documents(directory: str) -> Chroma:
    """加载目录中的文档并创建向量存储"""
    loader = DirectoryLoader(directory, glob="**/*.*")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

def search_documents(state: SearchState):
    """搜索文档并更新状态"""
    # 使用向量存储搜索相关文档
    results = state["vectorstore"].similarity_search(state["query"], k=3)
    
    # 准备消息内容
    content = f"找到以下相关文件:\n"
    for doc in results:
        content += f"\n文件路径: {doc.metadata['source']}\n"
        content += f"内容预览: {doc.page_content[:200]}...\n"
    
    # 让 LLM 分析结果
    result = search_chain.invoke({
        "messages": [HumanMessage(content=content)],
        "iteration": state["iteration"]
    })
    
    return {
        "messages": [result],
        "results": [{"path": doc.metadata["source"], "content": doc.page_content} for doc in results],
        "iteration": state["iteration"] + 1
    }

def should_continue(state: SearchState):
    """判断是否继续搜索"""
    if state["iteration"] >= 3:  # 最多迭代3次
        return END
    
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return END
        
    return "search"

# 创建工作流图
workflow = StateGraph(SearchState)

# 添加搜索节点
workflow.add_node("search", search_documents)

# 添加条件边
workflow.add_conditional_edges(
    "search",
    should_continue,
    {
        "search": "search",
        END: END
    }
)

workflow.add_edge(START, "search")

# 编译图
graph = workflow.compile()

def search_in_directory(directory: str, query: str):
    """在指定目录中搜索相关文档"""
    # 加载文档到向量存储
    vectorstore = load_documents(directory)
    
    # 初始化状态
    initial_state = {
        "messages": [],
        "query": query,
        "iteration": 0,
        "results": [],
        "vectorstore": vectorstore
    }
    
    # 执行搜索工作流
    for event in graph.stream(initial_state):
        event_search = event["search"]
        if "messages" in event_search:
            for message in event_search["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\n迭代 {event_search['iteration']}:")
                    print(message.content)
                    if event_search.get("results"):
                        print("\n找到的文件:")
                        for result in event_search["results"]:
                            print(f"- 路径: {result['path']}")

if __name__ == "__main__":
    directory = "/mnt/f/iCloudDrive/workshop/副业/享梦游/私域"
    query = "如何使用 LangChain?"  # 搜索查询
    search_in_directory(directory, query)
