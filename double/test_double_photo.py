import double
import cv2
import numpy as np
import os
# Photo test
robot_ip = "192.168.1.200"
port = int(os.getenv("PORT", 12345))  
d3 = double.DRDoubleSDK(robot_ip=robot_ip, port=port)
image = d3.capture_photo((120, 120))
if image is not None:
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to capture or display the image.")
d3.close()


