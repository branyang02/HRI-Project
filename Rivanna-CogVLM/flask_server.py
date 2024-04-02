from flask import Flask, request, jsonify
import numpy as np
from CogVLM import CogVLM
import cv2
import base64

app = Flask(__name__)

# Pre-initialize VLM
VLM = CogVLM()


def image_to_base64(cv2_image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", cv2_image)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str


def base64_to_cv2_image(base64_str: str) -> np.ndarray:
    decoded = base64.b64decode(base64_str)
    np_arr = np.fromstring(decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


@app.route("/inference", methods=["POST"])
def inference():
    # Extract data from request
    data = request.json
    prompt = data["prompt"]
    base64_image = data["image"]

    # Perform inference
    result = VLM.inference(base64_image, prompt)

    # Return the result
    return jsonify(result)


@app.route("/")
def home():
    return "The API endpoint works!!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
