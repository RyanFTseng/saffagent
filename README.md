# Saffagent


Custom LLM agent workflow for handling user queries with context persistence using the Qwen3 LLM.


Agent Nodes:


Summarizer - Summarizes previous conversations in chat history for other agents.


Classifier - Classifies current query into for the router to direct to the appropriate agent.


Logical -  Responds to logic-based queries.


Therapist - Responds to emotional-based queries.



Graph Structure:

![image](https://github.com/user-attachments/assets/870a0c17-6d28-4d2d-98c7-5415aab7cca8)


