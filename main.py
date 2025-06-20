from agents import graph, BaseModel
from text_funcs import  log, clear_log
from fastapi import FastAPI

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

chat_state = {"messages": [], "message_type": None}

@app.get("/")
def root():
    return {"wah wah": "world"}

@app.post("/agent")
def agent(request: ChatRequest):
    user_input = request.message

    if user_input == "clear":
        clear_log("log.txt")
        return {"response": "Chat log cleared."}

    chat_state["messages"] = chat_state.get("messages", []) + [
        {"role": "user", "content": user_input}
    ]

    new_state = graph.invoke(chat_state)
    chat_state.update(new_state)

    if new_state.get("messages"):
        last_message = new_state["messages"][-1]
        log("User Message: " + user_input, "log.txt")
        log(f"Assistant: {last_message.content}", "log.txt")
        log("---\n", "log.txt")
        return {"response": last_message.content}
                                                    
    return {"response": "No response generated."}






#local run
'''
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

        if user_input == "clear":
            clear_log("log.txt")
            print("Chat log cleared")
            user_input = input("User Message: ")

        #add user message to state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
            ]

        #call llm using state
        state =graph.invoke(state)

        #print messages in state added by llm/user
        if state.get("messages") and len(state["messages"]) >0:
            last_message = state["messages"][-1]
            log("User Message: " + user_input, "log.txt")
            log(f"Assistant: {last_message.content}", "log.txt")
            log("---\n", "log.txt")
            print(f"Assistant: {last_message.content}")





if __name__ == "__main__":
    run_chatbot()
'''