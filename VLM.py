from openai import OpenAI
import numpy as np
import base64
import os
import cv2
import requests
from abc import ABC, abstractmethod


class VLM(ABC):
    @abstractmethod
    def inference(self, image: np.ndarray, prompt: str):
        pass


class OPENAI_VLM(VLM):
    def __init__(self):
        self.client = OpenAI()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }

    def inference(self, image: np.ndarray, prompt: str):
        resized_image = cv2.resize(image, (256, 256))
        base64_image = self._image_to_base64(resized_image)
        payload = self._build_payload(base64_image, prompt)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        response = response.json()
        return response["choices"][0]["message"]["content"]

    def _image_to_base64(self, cv2_image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", cv2_image)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return base64_str

    def _build_payload(self, base64_image: str, prompt: str, max_tokens: int = 300):
        return {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
