import threading
import socket
import json
import cv2
import requests
import numpy as np


class DRDoubleSDK:
    def __init__(self, robot_ip, port):
        self.robot_ip = robot_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((robot_ip, port))

    def close(self):
        self.sock.close()

    def sendCommand(self, command, data=None):
        packet = {"c": command}
        if data is not None:
            packet["d"] = data
        jsonString = json.dumps(packet)
        self.sock.send(jsonString.encode("utf-8"))

    def recv(self):
        packet = self.sock.recv(4096).decode("utf-8")
        if not packet:
            exit("Error: received None from D3SDK")
        object = None
        try:
            object = json.loads(packet)
        except ValueError as e:
            print("JSON Parse error", packet)
        return object

    def capture_photo(self, img_size=(520, 520)):
        self.sendCommand(
            "camera.enable",
            {
                "width": img_size[0],
                "height": img_size[1],
                "template": "preheat",
                "gstreamer": "appsrc name=d3src ! autovideosink",
            },
        )
        self.sendCommand("events.subscribe", {"events": ["DRCamera.photo"]})
        self.sendCommand("camera.capturePhoto")

        while True:
            packet = self.recv()
            print(packet)
            if packet != None:
                event = packet["class"] + "." + packet["key"]
                if event == "DRBase.status":
                    print(packet["data"])
                elif event == "DRCamera.enable":
                    print("camera enabled")
                elif event == "DRCamera.photo":
                    print("photo captured")
                    # URL from where to fetch the captured image
                    image_url = f"http://{self.robot_ip}:8080/d3-camera-photo.jpg"
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        nparr = np.frombuffer(response.content, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return image
