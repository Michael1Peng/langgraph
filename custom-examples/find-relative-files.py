from typing import Annotated, List
from typing_extensions import TypedDict
import operator
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START

# Define state to track messages and file contents
class FileReaderState(TypedDict):
    messages: Annotated[List, operator.add]
    files: List[str]
    current_file: str
    file_contents: dict

# Create prompt for processing files
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant that reads and analyzes files.
        When given file contents, summarize them briefly and note any key points.
        Current file being processed: {current_file}"""
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm

def process_files(state: FileReaderState):
    """Process the current file and update state"""
    if not state["files"]:
        # No more files to process
        return {
            "messages": [AIMessage(content="Finished processing all files.")],
            "current_file": ""
        }
    
    # Get next file to process
    current_file = state["files"][0]
    remaining_files = state["files"][1:]
    
    # Get file contents
    content = state["file_contents"].get(current_file, "")
    
    # Process with LLM
    result = chain.invoke({
        "messages": [HumanMessage(content=f"Contents of {current_file}:\n{content}")],
        "current_file": current_file
    })
    
    return {
        "messages": [result],
        "files": remaining_files,
        "current_file": current_file
    }

def should_continue(state: FileReaderState):
    """Determine if we should continue processing files"""
    if state["files"]:
        return "process"
    return END

# Create graph
workflow = StateGraph(FileReaderState)
workflow.add_node("process", process_files)
workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "process": "process",
        END: END
    }
)
workflow.add_edge(START, "process")

# Compile graph
graph = workflow.compile()

def read_directory(directory_path: str):
    """Read all files in a directory and process them"""
    # Get list of files
    files = []
    file_contents = {}
    
    # Walk through directory
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    files.append(filepath)
                    file_contents[filepath] = content
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    # Initialize state
    initial_state = {
        "messages": [],
        "files": files,
        "current_file": "",
        "file_contents": file_contents
    }
    
    # Process files through graph
    for event in graph.stream(initial_state):
        if "messages" in event['process']:
            for message in event["process"]["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\nProcessing: {event.get('current_file', '')}")
                    print(message.content)

if __name__ == "__main__":
    # Example usage
    directory = "./custom-examples"
    read_directory(directory)
