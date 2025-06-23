import os

MAX_LOG_QUERIES = 10

def get_tmp_file_path(filename):
    return os.path.join("/tmp", filename)

def log(text, file_name):
    log_file = get_tmp_file_path(file_name)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def clear_log(file_name):
    log_file = get_tmp_file_path(file_name)
    with open(log_file, "w", encoding="utf-8") as f:
        pass

def read_log():
    log_file = get_tmp_file_path("log.txt")
    if not os.path.exists(log_file):
        return ""

    with open(log_file, "r+", encoding="utf-8") as f:
        lines = f.readlines()
        if lines.count("---\n") >= MAX_LOG_QUERIES:
            first = lines.index("---\n")
            lines = lines[first+1:]
            f.seek(0)
            f.writelines(lines)
            f.truncate()
        f.seek(0)
        return f.read()
