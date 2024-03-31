import cv2
from VLM.LMRobot import LMRobot
from util.util import extract_list_from_string
from double.double import DRDoubleSDK
import numpy as np
import os
import sys


# # 1. Capture image
# # image = cv2.VideoCapture(0)
# image = cv2.imread("photo.jpg")

# # 2. Create LMRobot instance
# LM_robot = LMRobot()

# # 3. Call detect_and_rank_humans method
# response = LM_robot.detect_and_rank_humans(image=image, prompt="")
# print(response)
# ranked_human_descriptions = extract_list_from_string(response)
# print(ranked_human_descriptions)


robot_ip = "192.168.1.200"
port = int(os.getenv("PORT", 12345))
d3 = DRDoubleSDK(robot_ip=robot_ip, port=port)

try:
    # 1. Get image from D3
    image = d3.capture_photo((120, 120))
    cv2.imwrite("test_photo.jpg", image)
    print("Image captured and saved.")
    # 2. Create LMRobot instance
    LM_robot = LMRobot()
    # 3. Call detect_and_rank_humans method
    response = LM_robot.detect_and_rank_humans(image=image, prompt="")
    print(response)
    ranked_human_descriptions = extract_list_from_string(response)
    print(ranked_human_descriptions)
except KeyboardInterrupt:
    d3.sendCommand("camera.disable")
    d3.sendCommand("screensaver.nudge")
    d3.close()
    print("cleaned up")
    sys.exit(0)
