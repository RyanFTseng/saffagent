from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_nvidia import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

load_dotenv()


def log(text):
    with open("log.txt", "a") as f:
        f.write(text + "\n")
    f = open("log.txt", "r")
    f.close()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

#tools = [search_tool, wiki_tool, save_tool]
llm = init_chat_model("qwen/qwen3-235b-a22b", model_provider = 'nvidia')
parser = PydanticOutputParser(pydantic_object = ResearchResponse)

#llm.bind_tools(tools)

#State to classify message into 2 states
class MessageClassifier(BaseModel):
    #force message type to be "emotional" or "logical"
    message_type: Literal ["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires emotional or logical response"
    )

#State with list of messages, updates after each node in graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    #return type
    message_type: str | None
    

#classfier node
#provides llm with prompt to classify user message and get either emotional or logical in response
def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    #combine prompt with user message



    #view raw result for debugging
    '''
    raw_result = llm.invoke([
    {"role": "system", "content": "Classify the user message as either:- 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems- 'logical': if it asks for facts, information, logical analysis, or practical solutions"},
    {"role": "user", "content": last_message.content}
    ])

    print("Raw LLM result:", raw_result)
    '''

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            You must respond with a JSON object matching this format:
                                            {
                                              "message_type": "emotional" | "logical"
                                            }
            """
            },
            {"role": "user", "content": last_message.content}    #input user message
        ])
    #return either emotional or logical based on user input and llm response 
    #updates messages in state

    print(result.message_type)
    return {"message_type": result.message_type}


#pass message to appropriate llm
def router(state: State):
    message_type = state.get("message type", "logical")#default to logical
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
                "content": """
                                            You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                                            Show empathy, validate their feelings, and help them process their emotions.
                                            Ask thoughtful questions to help them explore their feelings more deeply.
                                            Avoid giving logical solutions unless explicitly asked."""
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
                "content": """You are a purely logical assistant. Focus only on facts and information.
                              Provide clear, concise answers based on logic and evidence.
                              Do not address emotions or provide emotional support.
                              Be direct and straightforward in your responses."""
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
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)


#connect nodes using edges
graph_builder.add_edge(START, "classifier")
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





def run_chatbot():
    #initialize state
    state = {"messages": [], "message_type": None}

    while True:
        #get user input
        user_input = input("User Message: ")

        #user controlled exit
        if user_input  == "exit":
            print("Bye")
            break

        #add user message to state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
            ]

        #call llm using state
        state =graph.invoke(state)

        #print messages in state added by llm/user
        if state.get("messages") and len(state["messages"]) >0:
            last_message = state["messages"][-1]
            log("User Message" + user_input)
            log(f"Assistant: {last_message.content}")
            print(f"Assistant: {last_message.content}")



if __name__ == "__main__":
    run_chatbot()



'''
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a pro gamer with multiple championship titles.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


#init agent and tools
#tools = []
#tools = [search_tool, wiki_tool, save_tool]
tools = [search_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)
query = input("What can I help you with? ")
raw_response = agent_executor.invoke({"query":query}) #match with human inputs in prompt def
print(raw_response)

#parse raw output
try:
    structured_response = parser.parse(raw_response.get("output"))
except Exception as e:
    print("error parsing response", e, " Raw Respone - ", raw_response)



#print(structured_response)
'''