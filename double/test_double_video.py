import double
import cv2
import numpy as np
import os

# Example Video
robot_ip = "192.168.1.200"
port = int(os.getenv("PORT", 12345))
d3 = double.DRDoubleSDK(robot_ip=robot_ip, port=port)
d3.stream_video(frame_size=(120, 120)) 
d3.close()