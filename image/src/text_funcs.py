#text functions

MAX_LOG_QUERIES = 10

def log(text, file_name):
    with open(file_name, "a", encoding = "utf-8") as f:
        f.write(text + "\n")

def clear_log(file_name):
    with open(file_name, "w", encoding = "utf-8") as f:
        f.write('')
    f.close()

def read_log():
    with open("log.txt", "r+",encoding = "utf-8") as f:
        #check if chat history exceeds max log queries
        #delete first query in chat history if exceeded
        lines = f.readlines()
        if lines.count("---\n") >= MAX_LOG_QUERIES:
            first = lines.index("---\n")
            lines = lines[first+1:]
            f.seek(0)
            f.writelines(lines)
            f.truncate()
        f.seek(0)
        log = f.read()
        f.close()
    return log