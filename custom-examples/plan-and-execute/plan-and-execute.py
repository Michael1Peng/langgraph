# %% [markdown]
# # Plan-and-Execute
# 
# # 计划与执行
# 
# This notebook shows how to create a "plan-and-execute" style agent. This is heavily inspired by the [Plan-and-Solve](https://arxiv.org/abs/2305.04091) paper as well as the [Baby-AGI](https://github.com/yoheinakajima/babyagi) project.
# 
# 本笔记本展示了如何创建一个“计划和执行”风格的智能体。这受到[Plan-and-Solve](https://arxiv.org/abs/2305.04091)论文以及[Baby-AGI](https://github.com/yoheinakajima/babyagi)项目的强烈启发。
# 
# The core idea is to first come up with a multi-step plan, and then go through that plan one item at a time.
# 
# 核心思想是首先制定一个多步骤的计划，然后逐项执行这个计划。
# After accomplishing a particular task, you can then revisit the plan and modify as appropriate.
# 
# 在完成某项任务后，您可以重新审视计划并根据需要进行修改。
# 
# 
# The general computational graph looks like the following:
# 
# 一般的计算图如下所示：
# 
# ![plan-and-execute diagram](attachment:86cf6404-3d9b-41cb-ab97-5e451f576620.png)
# 
# 抱歉，我无法直接查看或翻译图像内容。如果您可以提供图像中的文本或描述内容，我会很乐意帮助您翻译。
# 
# 
# This compares to a typical [ReAct](https://arxiv.org/abs/2210.03629) style agent where you think one step at a time.
# 
# 这与典型的 [ReAct](https://arxiv.org/abs/2210.03629) 风格的智能体相比，后者是逐步思考的。
# The advantages of this "plan-and-execute" style agent are:
# 
# 这种“计划与执行”风格的代理的优点是：
# 
# 1. Explicit long term planning (which even really strong LLMs can struggle with)
# 
# 明确的长期规划（即使是非常强大的大型语言模型也可能会面临困难）
# 2. Ability to use smaller/weaker models for the execution step, only using larger/better models for the planning step
# 
# 2. 能够在执行步骤中使用较小/较弱的模型，仅在规划步骤中使用较大/较好的模型。
# 
# 
# The following walkthrough demonstrates how to do so in LangGraph. The resulting agent will leave a trace like the following example: ([link](https://smith.langchain.com/public/d46e24d3-dda6-44d5-9550-b618fca4e0d4/r)).
# 
# 以下操作指南演示了如何在 LangGraph 中进行操作。生成的代理将留下类似以下示例的痕迹: ([链接](https://smith.langchain.com/public/d46e24d3-dda6-44d5-9550-b618fca4e0d4/r))。
# 

# %% [markdown]
# ## Setup
# 
# ## 设置
# 
# First, we need to install the packages required.
# 
# 首先，我们需要安装所需的包。
# 

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain-community langchain-openai tavily-python

# %% [markdown]
# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# 
# 接下来，我们需要为 OpenAI（我们将使用的大型语言模型）和 Tavily（我们将使用的搜索工具）设置 API 密钥。
# 

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

# %% [markdown]
# <div class="admonition tip">
# 
# <div class="admonition tip">  
# 提示  
# </div>
#     <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
# 
# <p class="admonition-title">为 LangGraph 开发设置 <a href="https://smith.langchain.com">LangSmith</a></p>
#     <p style="padding-top: 5px;">
# 
# <p style="padding-top: 5px;">
#         Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
# 
# 注册LangSmith，快速发现问题并提升您使用LangGraph项目的性能。LangSmith使您能够利用追踪数据来调试、测试和监控您基于LangGraph构建的LLM应用程序——想了解如何开始，请点击<a href="https://docs.smith.langchain.com">这里</a>。
#     </p>
# 
# 看起来你提供的文本仅包含一个标记（`</p>`），这是HTML中的一个结束标签，代表段落的结束。请提供更多内容或文本，我将很乐意为你翻译。
# </div>
# 
# 该文本没有提供可翻译的内容。请提供需要翻译的具体文本。
# 

# %% [markdown]
# ## Define Tools
# 
# ## 定义工具
# 
# We will first define the tools we want to use. For this simple example, we will use a built-in search tool via Tavily. However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/how_to/custom_tools) on how to do that.
# 
# 我们将首先定义我们想要使用的工具。对于这个简单的例子，我们将通过 Tavily 使用内置的搜索工具。不过，创建自己的工具其实很简单 - 请参阅 [这里](https://python.langchain.com/docs/how_to/custom_tools) 的文档了解如何操作。
# 

# %%
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]

# %% [markdown]
# ## Define our Execution Agent
# 
# ## 定义我们的执行代理商
# 
# Now we will create the execution agent we want to use to execute tasks. 
# 
# 现在我们将创建一个执行代理，用于执行任务。
# Note that for this example, we will be using the same execution agent for each task, but this doesn't HAVE to be the case.
# 
# 请注意，在这个例子中，我们将为每个任务使用相同的执行代理，但这并不是必须的。
# 

# %%
from langchain import hub
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4o")
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

# %%
agent_executor.invoke({"messages": [("user", "who is the winnner of the lol s14?")]})

# %% [markdown]
# ## Define the State
# 
# ## 定义状态
# 
# Let's now start by defining the state the track for this agent.
# 
# 现在我们先来定义这个代理的轨迹状态。
# 
# First, we will need to track the current plan. Let's represent that as a list of strings.
# 
# 首先，我们需要跟踪当前的计划。我们将其表示为一个字符串列表。
# 
# Next, we should track previously executed steps. Let's represent that as a list of tuples (these tuples will contain the step and then the result)
# 
# 接下来，我们应该跟踪之前执行的步骤。我们可以将其表示为一个元组列表（这些元组将包含步骤以及结果）。
# 
# Finally, we need to have some state to represent the final response as well as the original input.
# 
# 最后，我们需要有一些状态来表示最终响应以及原始输入。
# 

# %%
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# %% [markdown]
# ## Planning Step
# 
# ## 规划步骤
# 
# Let's now think about creating the planning step. This will use function calling to create a plan.
# 
# 现在让我们考虑创建计划步骤。这将使用函数调用来制定一个计划。
# 

# %% [markdown]
# <div class="admonition note">
# 
# <div class="admonition note"> 
#     <p class="admonition-title">Using Pydantic with LangChain</p>
# 
# <p class="admonition-title">在 LangChain 中使用 Pydantic</p>
#     <p>
# 
# 请提供您希望翻译的具体文本内容，我将很乐意为您翻译。
#         This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
# 
# 该笔记本使用了 Pydantic v2 的 <code>BaseModel</code>，这需要 <code>langchain-core >= 0.3</code>。使用 <code>langchain-core < 0.3</code> 会导致由于同时使用 Pydantic v1 和 v2 的 <code>BaseModels</code> 而产生错误。
#     </p>
# 
# 您提供的文本仅包含一个 HTML 标签，表示段落的结束，没有实际内容。请提供需要翻译的文本内容，我会很乐意为您翻译。
# </div>
# 
# 该文本包含一个 HTML 标签 "</div>"，在中文中没有特定的意义，通常翻译为“结束的 div 标签”。
# 

# %%
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

# %%
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)

# %%
planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)

# %% [markdown]
# ## Re-Plan Step
# 
# ## 重新规划步骤
# 
# Now, let's create a step that re-does the plan based on the result of the previous step.
# 
# 现在，让我们创建一个步骤，根据前一步的结果重新制定计划。
# 

# %%
from typing import Union


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

If you have all the information needed to answer the user's question, respond with a final answer in the format:
{{"action": {{"response": "your final answer here"}}}}

If more steps are needed, respond with the remaining steps in the format:
{{"action": {{"steps": ["step 1", "step 2", ...]}}}}

Only include steps that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

# %% [markdown]
# ## Create the Graph
# 
# ## 创建图表
# 
# We can now create the graph!
# 
# 我们现在可以创建图表了！
# 

# %%
from typing import Literal
from langgraph.graph import END


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

# %%
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# %%
from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %%
config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the all team members of lol s13 winner?"}
for event in app.stream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)

# %% [markdown]
# ## Conclusion
# 
# ## 结论
# 
# Congrats on making a plan-and-execute agent! One known limitations of the above design is that each task is still executed in sequence, meaning embarrassingly parallel operations all add to the total execution time. You could improve on this by having each task represented as a DAG (similar to LLMCompiler), rather than a regular list.
# 
# 恭喜你制作了一个计划与执行的智能体！上述设计的一个已知限制是每个任务仍然是顺序执行的，这意味着尴尬的并行操作都会增加总执行时间。你可以通过将每个任务表示为有向无环图（DAG，类似于LLMCompiler），而不是常规列表，来改进这一点。
# 


