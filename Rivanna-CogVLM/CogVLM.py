import numpy as np
import os
import base64
import io
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
import torch

CACHE_DIR = "/scratch/" + os.environ["USER"] + "/huggingface_cache"


class CogVLM:
    def __init__(self, torch_dtype=torch.bfloat16, quant=None):
        # self.model_path = "THUDM/cogvlm-grounding-generalist-hf"
        self.model_path = "THUDM/cogagent-chat-hf"
        self.tokenizer_path = "lmsys/vicuna-7b-v1.5"
        self.device = "cuda"
        self.torch_dtype = torch_dtype
        self.quant = quant

        # initialize tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.tokenizer_path, cache_dir=CACHE_DIR
        )

        # initialize model
        if self.quant:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            ).eval()
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    load_in_4bit=self.quant is not None,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR,
                )
                .to(self.device)
                .eval()
            )

    def inference(self, base64_image: str, prompt: str):
        text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        print("User: ", prompt)
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        query = text_only_template.format(prompt)

        # Assuming empty history prompts
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=query, history=[], images=[image]
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": (
                [[input_by_model["images"][0].to(self.device).to(self.torch_dtype)]]
                if image is not None
                else None
            ),
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.device).to(self.torch_dtype)]
            ]

        gen_kwargs = {
            "max_length": 2048,
            "do_sample": False,
            "temperature": 0.0,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
            return response
