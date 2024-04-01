import cv2
from VLM.LMRobot import LMRobot
from util.util import extract_coords, visualize_bbox_on_image

# 1. Capture image
image = cv2.imread("photo.jpg")
image = cv2.resize(image, (1120, 1120))  # 1120 x 1120 is the expected image size

# 2. Create LMRobot instance
LM_robot = LMRobot(model="cogvlm")

# 3. Call detect_and_rank_humans method
response = LM_robot.detect_and_rank_humans(
    image=image,
    prompt="",
)
print("VLM output: ", response)

# 4. Extract coordinates from the response
coords = extract_coords(response)
print("Coordinates: ", coords)

# 5. Draw bounding box around the person
image_with_bbox = visualize_bbox_on_image(image, coords)

# 6. Display the image with bounding box
cv2.imshow("Image with bounding box", image_with_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()
