from agents import graph, BaseModel
from text_funcs import  log, clear_log

#terminal run

def run_chatbot():
    #initialize state to hold chat history
    #Example state:
    #{
    #"messages": [
    #    {"role": "user", "content": "Hi"},
    #    {"role": "assistant", "content": "Hello! How can I help you?"},
    #    {"role": "user", "content": "What's the weather today?"}
    #]
    #}
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
