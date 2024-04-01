import ast
import cv2
import numpy as np
import base64


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


def image_to_base64(cv2_image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", cv2_image)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str


def base64_to_cv2_image(base64_str: str) -> np.ndarray:
    decoded = base64.b64decode(base64_str)
    np_arr = np.fromstring(decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image
