import ast


def extract_list_from_string(s):
    start = s.find("[")
    end = s.rfind("]") + 1
    if start != -1 and end != -1:
        list_str = s[start:end]
        try:
            return ast.literal_eval(list_str)
        except:
            return "Failed to extract list."
    return "No list format found."
