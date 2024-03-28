import double
import cv2
import numpy as np
import os

robot_ip = "192.168.1.200"
port = int(os.getenv("PORT"))
d3 = double.DRDoubleSDK(robot_ip=robot_ip, port=port)

image = d3.capture_photo((120, 120))
cv2.imshow("Captured Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

d3.close()
