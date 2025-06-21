from agents import graph, ChatRequest
from text_funcs import  log, clear_log
from auth.throttling import apply_rate_limit
from auth.dependencies import get_user_identifier
from fastapi import Depends, FastAPI


#uvicorn main:app --reload

#init app
app = FastAPI()


#dict to track user messages
chat_state = {"messages": [], "message_type": None}

@app.get("/")
async def root():
    return {"app: running"}

@app.post("/agent")
async def agent(request: ChatRequest, user_id: str = Depends(get_user_identifier)):
    apply_rate_limit(user_id)
    user_input = request.message

    if user_input == "clear":
        clear_log("log.txt")
        return {"response": "Chat log cleared."}

    #add current query to message state
    chat_state["messages"] = chat_state.get("messages", []) + [
        {"role": "user", "content": user_input}
    ]

    new_state = graph.invoke(chat_state)
    chat_state.update(new_state)

    if new_state.get("messages"):
        last_message = new_state["messages"][-1]
        log(f"User Message: {user_input}\nAssistant: {last_message.content}\n---\n", "log.txt")

        return {"response": last_message.content}
                                                    
    return {"response": "No response generated."}






