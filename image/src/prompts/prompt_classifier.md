"""
    Classify the current query as either:
    - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
    - 'logical': if it asks for facts, information, logical analysis, or practical solutions
    You must respond with a JSON object matching this format:
    {
        "message_type": "emotional" | "logical"
    }
"""