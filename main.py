import cv2
from LMRobot import LMRobot

# 1. Capture image
# image = cv2.VideoCapture(0)
image = cv2.imread("photo.jpg")

# 2. Create LMRobot instance
LM_robot = LMRobot()

# 3. Call detect_and_rank_humans method
response = LM_robot.detect_and_rank_humans(image=image, prompt="")
print(response)
