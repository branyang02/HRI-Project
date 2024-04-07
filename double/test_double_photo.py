import double
import cv2
import numpy as np
import os
import sys

robot_ip = "192.168.1.200"
port = int(os.getenv("PORT", 12345))
d3 = double.DRDoubleSDK(robot_ip=robot_ip, port=port)
print("Connected to robot")

try:
    image = d3.capture_photo((120, 120))
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    d3.sendCommand("camera.disable")
    d3.sendCommand("screensaver.nudge")
    d3.close()
    print("cleaned up")
    sys.exit(0)
