from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_core.output_parsers import PydanticOutputParser
from tools import search_tool, wiki_tool, save_tool
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from google.cloud import aiplatform
from typing_extensions import TypedDict
from text_funcs import log, clear_log, read_log
from judgeval.common.tracer import Tracer
from judgeval.scorers import AnswerRelevancyScorer
from judgeval.data import Example


judgment = Tracer(project_name="SaffAgent")

load_dotenv()
aiplatform.init(project="746472204967", location="us-west1")

#load agent prompts
def load_system_prompt(prompt_file):
    with open("prompts/" + prompt_file, "r") as f:
        return f.read()

classifier_prompt = load_system_prompt("prompt_classifier.md")
logical_prompt = load_system_prompt("prompt_logical.md")
summary_prompt = load_system_prompt("prompt_summary.md")
therapist_prompt = load_system_prompt("prompt_therapist.md")
 
#Pydantic output 
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

#Pydantic input
class ChatRequest(BaseModel):
    message: str

#load llm model
llm = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)
#init output parser
parser = PydanticOutputParser(pydantic_object = ResearchResponse)

#tools = [search_tool, wiki_tool, save_tool]
#llm.bind_tools(tools)

#Nested dictionary to store to classified message 
#Example:
#state = {
#    "messages": [
#       {"role": "user", 
#       "content": "Hi"},
#       {"role": "assistant", 
#       "content": "Hello! How can I help?"}
#   ]
#   "message_type": "logical"
#

class MessageClassifier(BaseModel):
    #force message type to be "emotional" or "logical"
    message_type: Literal ["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires emotional or logical response"
    )

#State class with list of messages, updates after each node in graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    #return type
    message_type: str | None
    
#Summary agent
def summary_agent(state: State):
    #obtain user input
    last_message = state["messages"][-1]
    clear_log("summary.txt")      
    #prompt setup
    messages = [
            {   
                "role": "system",
                "content": summary_prompt
            },
            {
                "role": "user",
                "content": read_log() + " " + last_message.content
            }
        ]
    #call llm with prompt
    reply = llm.invoke(messages)
    log(reply.content, "summary.txt")
    #eval
    """Summarize user chat history"""
    ans =  "The user asked questions and was responded to with an answer. The current query is asked by the user"
    example = Example(
        input="Summarize the chat history",
        actual_output=ans
    )
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        example=example,
        model = "gpt-4.1"
    )
    #prompt setup
    return {"messages": [{"role": "assistant", "content" : reply.content}]}

#classfier node
#provides llm with prompt to classify user message and get either emotional or logical in response
def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    #prompt invoaation
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": classifier_prompt
            },
            #combine with user message
            {"role": "user", "content": last_message.content}    
        ])
    
    #updates messages in state
    print(result.message_type)
    return {"message_type": result.message_type}


#pass message to appropriate llm
def router(state: State):
    message_type = state.get("message_type", "logical")#default to logical
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}

#emotional agent
def therapist_agent(state: State):
    #obtain user input
    last_message = state["messages"][-1]

    #prompt setup
    messages = [
            {   
                "role": "system",
                "content": therapist_prompt
            },
            {
                "role": "user",
                "content": last_message.content
            }
        ]

    #call llm with prompt
    reply = llm.invoke(messages)
    #return llm response
    return {"messages": [{"role": "assistant", "content" : reply.content}]}


#logical agent
def logical_agent(state: State):
    #obtain user input
    last_message = state["messages"][-1]

    #prompt setup
    messages = [
            {   
                "role": "system",
                "content": logical_prompt
                              
            },
            {
                "role": "user",
                "content": last_message.content
            }
        ]
    #call llm with prompt
    reply = llm.invoke(messages)
    #prompt setup
    return {"messages": [{"role": "assistant", "content" : reply.content}]}



#init graph builder using state
graph_builder = StateGraph(State)
#add nodes
graph_builder.add_node("summarizer", summary_agent)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

#connect nodes using edges
graph_builder.add_edge(START, "summarizer")
graph_builder.add_edge("summarizer", "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    #get value of state
    lambda state: state.get("next"),
    #set path based on state value {value: destination node name}
    {"therapist": "therapist", "logical": "logical"}
    )
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)


#compile graph
graph = graph_builder.compile()