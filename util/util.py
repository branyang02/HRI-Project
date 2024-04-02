import ast
import cv2
import numpy as np
import base64
import re


def extract_coords(s):
    """
    Extracts coordinates from a string and removes leading zeros from numbers.
    Args:
        s (str): [[x0,y0,x1,y1]]
    """
    # Find the substring containing the coordinates
    start = s.find("[[")
    end = s.find("]]") + 2
    if start != -1 and end != -1:
        coords_str = s[start:end]
        # Remove leading zeros from numbers using a regular expression
        coords_str_no_leading_zeros = re.sub(r"\b0+([0-9]+)", r"\1", coords_str)
        try:
            print(
                f"Extracted substring (with leading zeros removed): {coords_str_no_leading_zeros}"
            )
            return ast.literal_eval(coords_str_no_leading_zeros)
        except Exception as e:
            print(f"Error: {e}")
            return "Failed to extract coordinates."
    return "No coordinates format found."


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


def visualize_bbox_on_image(image: np.ndarray, coords, image_size=(1120, 1120)):
    """
    Args:
        image (np.ndarray): Input image.
        coords (list): Coordinates of the bounding box in the format [[x0,y0,x1,y1]].
    """
    x0, y0, x1, y1 = coords[0]
    image = cv2.resize(image, image_size)
    image_with_bbox = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return image_with_bbox
