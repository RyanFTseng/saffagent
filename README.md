# Saffagent

Custom llm agent workflow for handling user queries with context persistence using the Qwen3 LLM.

Graph Structure:
![image](https://github.com/user-attachments/assets/870a0c17-6d28-4d2d-98c7-5415aab7cca8)

Agent Nodes:
Summarizer - Summarizes previous conversations in chat history for other agents.
Classifier - Classifies current query into for the router to direct to the appropriate agent.
Logical -  Responds to logic-based queries sent by classifier.
Therapist - Responds to emotional-based queries sent by classifier.


