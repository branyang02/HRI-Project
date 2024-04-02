import numpy as np
from VLM.VLM import OPENAI_VLM
from VLM.VLM import CogVLM

OPEN_AI_PROMPT = """
You are an intelligent robot situated in a room filled with humans, each engaged in different activities. Given the image of the room, your objective is to gain the attention of a human for assistance. To approach this strategically, you need to assess and rank the humans based on the likelihood of their availability and willingness to help, judging by their current actions.

Your task is to:

1. Observe the activities each human is involved in.
2. Rank these humans from the most to the least likely to give you their attention and help, based on their activities.
3. Compile your findings into a list, starting with the person most likely to assist you and ending with the least likely.

Describe each human with a brief sentence that captures their main activity and any distinctive features, like the color of their shirt or their location within the room.

Your output should be a Python list in the following format:
[
    "A person in a red shirt sitting quietly on a chair, appearing to be waiting for something.",
    "Someone in a blue shirt standing by the door, frequently checking their watch.",
    ...
]

Below is an example of how you should structure your response. You can use this as a template and fill in the details based on the image you observe:
In the room, There are four individuals who can potentially help you. First, we analyze the activities of each person in the room. Here are their descriptions:

Person 1: Wearing a yellow shirt, deeply engrossed in conversation on a phone, pacing back and forth.
Person 2: In a red shirt, sitting at a table, looking relaxed, occasionally glancing around the room.
Person 3: Sporting a blue jacket, intently working on a laptop, wearing headphones.
Person 4: Dressed in a green dress, standing by a window, looking out with a thoughtful expression, occasionally sipping coffee.

Based on these descriptions, here's how I would rank the individuals in terms of their likelihood to assist me. The person most likely to help is listed first, followed by others in descending order of their perceived willingness to assist:
[
    "Person in red shirt",
    "Person in green dress",
    "Person in yellow shirt",
    "Person in blue jacket"
]

Provide your answer:

"""

# CogVLM_PROMPT = """
# Your job is to figure out which person is most likely to help you given the image. First, analyze each person in the room and rank them based on their likelihood of helping you.

# Next, after analyzing each person in the image, give me the bounding box of the person who is most likely to help you in the format [[x0,y0,x1,y1]].
# """

CogVLM_PROMPT = """
In this task, you are to identify the individual within an image who seems most willing and capable of assisting you. Start by carefully observing each person in the scene. Evaluate their potential willingness and ability to offer help, taking into account factors such as their posture, facial expression, and any objects they might be interacting with or activities they are engaged in. Importantly, also consider how busy each person appears to be, as this may affect their availability to assist.

Rank each person based on these criteria to determine who is most likely to offer assistance.

Upon completing your analysis, identify the person you have determined is the most likely to help. Provide the bounding box. (with grounding)
"""


class LMRobot:
    def __init__(self, model: str = "openai"):
        if model == "gpt4":
            self.model = OPENAI_VLM()
            self.prompt = OPEN_AI_PROMPT
        elif model == "cogvlm":
            self.model = CogVLM()
            self.prompt = CogVLM_PROMPT
        else:
            raise ValueError("Invalid model type. Choose between 'gpt4' and 'cogvlm'.")

    def detect_and_rank_humans(self, image: np.ndarray, prompt: str) -> str:
        print(self.prompt + "\n" + prompt)
        response = self.model.inference(image, self.prompt + "\n" + prompt)
        return response
