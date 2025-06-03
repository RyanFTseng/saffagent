from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_nvidia import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(model = "nvidia/llama-3.1-nemotron-nano-4b-v1.1")
response = llm.invoke("who is zidane?")
print(response)
