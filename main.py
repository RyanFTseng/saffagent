from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_nvidia import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]
    

llm = ChatNVIDIA(model = "nvidia/llama-3.1-nemotron-nano-4b-v1.1")
parser = PydanticOutputParser(pydantic_object = ResearchResponse)

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
tools = [search_tool, wiki_tool, save_tool]

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
