# %% [markdown]
# ## 第5部分：手动更新状态
# 
# 在上一节中，我们展示了如何中断图以便人类可以检查其操作。这让人类可以`读取`状态，但如果他们想改变代理的路线，他们需要有`写入`权限。
# 
# 幸运的是，LangGraph允许你**手动更新状态**！更新状态让你可以通过修改代理的行为（甚至修改过去！）来控制代理的轨迹。当你想纠正代理的错误、探索替代路径或引导代理朝特定目标前进时，这个功能特别有用。
# 
# 我们将在下面展示如何更新检查点状态。和之前一样，首先定义你的图。我们将重用与之前完全相同的图。

# %%
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt **after** actions, if desired.
    # interrupt_after=["tools"]
)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# %%
snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()

# %% [markdown]
# 到目前为止，这完全是上一节的重复。LLM刚刚请求使用搜索引擎工具，我们的图被中断了。如果我们像之前一样继续，工具将被调用来搜索网络。
# 
# 但如果用户想要干预呢？如果我们认为聊天机器人不需要使用工具呢？
# 
# 让我们直接提供正确的回答！

# %%
from langchain_core.messages import AIMessage, ToolMessage

answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
)
new_messages = [
    # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
    ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
    # And then directly "put words in the LLM's mouth" by populating its response.
    AIMessage(content=answer),
]

new_messages[-1].pretty_print()
graph.update_state(
    # Which state to update
    config,
    # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
    # to the existing state. We will review how to update existing messages in the next section!
    {"messages": new_messages},
)

print("\n\nLast 2 messages;")
print(graph.get_state(config).values["messages"][-2:])

# %% [markdown]
# 现在图已经完成，因为我们已经提供了最终的响应消息！由于状态更新模拟了图的一个步骤，它们甚至生成相应的跟踪。检查上面`update_state`调用的[LangSmith跟踪](https://smith.langchain.com/public/6d72aeb5-3bca-4090-8684-a11d5a36b10c/r)，看看发生了什么。
# 
# **注意**我们的新消息被_追加_到状态中已有的消息之后。还记得我们是如何定义`State`类型的吗？
# 
# ```python
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
# ```
# 
# 我们用预构建的`add_messages`函数注释了`messages`。这指示图总是将值追加到现有列表，而不是直接覆盖列表。这里应用了相同的逻辑，所以我们传递给`update_state`的消息以相同的方式被追加！
# 
# `update_state`函数的操作就像它是图中的一个节点一样！默认情况下，更新操作使用最后执行的节点，但你可以在下面手动指定它。让我们添加一个更新，并告诉图将其视为来自"chatbot"。

# %%
graph.update_state(
    config,
    {"messages": [AIMessage(content="I'm an AI expert!")]},
    # Which node for this function to act as. It will automatically continue
    # processing as if this node just ran.
    as_node="chatbot",
)

# %% [markdown]
# 查看提供的链接中这个更新调用的[LangSmith跟踪](https://smith.langchain.com/public/2e4d92ca-c17c-49e0-92e5-3962390ded30/r)。**注意**从跟踪中可以看到，图继续进入`tools_condition`边缘。我们刚刚告诉图将更新视为`as_node="chatbot"`。如果我们按照下面的图从`chatbot`节点开始，我们自然会进入`tools_condition`边缘，然后是`__end__`，因为我们更新的消息没有工具调用。

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %% [markdown]
# 像之前一样检查当前状态，以确认检查点反映了我们的手动更新。

# %%
snapshot = graph.get_state(config)
print(snapshot.values["messages"][-3:])
print(snapshot.next)

# %% [markdown]
# **注意**：我们继续向状态添加AI消息。由于我们作为`chatbot`行动并用不包含`tool_calls`的AIMessage响应，图知道它已经进入了完成状态（`next`为空）。
# 
# #### 如果你想**覆盖**现有消息怎么办？
# 
# 我们用来注释图的`State`的[`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)函数控制如何对`messages`键进行更新。这个函数查看新`messages`列表中的任何消息ID。如果ID与现有状态中的消息匹配，[`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages)用新内容覆盖现有消息。
# 
# 作为例子，让我们更新工具调用，以确保我们从搜索引擎获得好的结果！首先，开始一个新线程：

# %%
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "2"}}  # we'll use thread_id = 2 here
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# %% [markdown]
# **接下来，**让我们更新我们代理的工具调用。也许我们想特别搜索人机交互工作流程。

# %%
from langchain_core.messages import AIMessage

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])
new_tool_call = existing_message.tool_calls[0].copy()
new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tool_call],
    # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
    id=existing_message.id,
)

print("Updated")
print(new_message.tool_calls[0])
print("Message ID", new_message.id)
graph.update_state(config, {"messages": [new_message]})

print("\n\nTool calls")
graph.get_state(config).values["messages"][-1].tool_calls

# %% [markdown]
# **注意**我们已经修改了AI的工具调用，搜索"LangGraph human-in-the-loop workflow"而不是简单的"LangGraph"。
# 
# 查看[LangSmith跟踪](https://smith.langchain.com/public/cd7c09a6-758d-41d4-8de1-64ab838b2338/r)以查看状态更新调用 - 你可以看到我们的新消息已成功更新了之前的AI消息。
# 
# 通过使用`None`作为输入和现有配置进行流式处理来恢复图。

# %%
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# %% [markdown]
# 查看[跟踪](https://smith.langchain.com/public/2d633326-14ad-4248-a391-2757d01851c4/r/6464f2f2-edb4-4ef3-8f48-ee4e249f2ad0)以查看工具调用和后续的LLM响应。**注意**现在图使用我们更新的查询词查询搜索引擎 - 我们能够手动覆盖LLM的搜索！
# 
# 所有这些都反映在图的检查点内存中，这意味着如果我们继续对话，它将回忆起所有_修改过的_状态。

# %%
events = graph.stream(
    {
        "messages": (
            "user",
            "Remember what I'm learning about?",
        )
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# %% [markdown]
# **恭喜！**你已经使用`interrupt_before`和`update_state`作为人机交互工作流程的一部分手动修改了状态。中断和状态修改让你可以控制代理的行为。结合持久性检查点，这意味着你可以在任何点`暂停`操作并`恢复`。你的用户不必在图中断时立即可用！
# 
# 这一节的图代码与之前的相同。要记住的关键片段是添加`.compile(..., interrupt_before=[...])`（或`interrupt_after`），如果你想在图每次到达一个节点时明确暂停。然后你可以使用`update_state`来修改检查点并控制图应如何继续。

