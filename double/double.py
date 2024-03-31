import socket
import json
import cv2
import numpy as np
import requests

class DRDoubleSDK:
    def __init__(self, robot_ip, port):
        self.robot_ip = robot_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((robot_ip, port))
        self.http_session = requests.Session()  # Use persistent HTTP connection

    def close(self):
        self.sock.close()
        self.http_session.close()  # Close the HTTP session

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

    def fetch_image(self, image_url):
        response = self.http_session.get(image_url)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                return image
            else:
                print("Failed to decode image.")
                return None
        else:
            print(f"Failed to fetch image, Status Code: {response.status_code}")
            return None

    def capture_photo(self, img_size=(520, 520)):
        self.sendCommand("camera.enable", {
            "width": img_size[0], "height": img_size[1], 
            "template": "preheat", "gstreamer": "appsrc name=d3src ! autovideosink",
        })
        self.sendCommand("events.subscribe", {"events": ["DRCamera.photo"]})
        self.sendCommand("camera.capturePhoto")

        while True:
            packet = self.recv()
            if packet is not None:
                event = packet["class"] + "." + packet["key"]
                if event == "DRCamera.photo":
                    print("Photo captured, fetching...")
                    image_url = f"http://{self.robot_ip}:8080/d3-camera-photo.jpg"
                    return self.fetch_image(image_url)
                
    def stream_video(self, frame_size=(520, 520)):
        # Streams video by continuously capturing images until 'q' is pressed.  
        print("Streaming video. Press 'q' to quit.")
        while True:
            self.sendCommand("camera.enable", {
                "width": frame_size[0], "height": frame_size[1], 
                "template": "preheat", "gstreamer": "appsrc name=d3src ! autovideosink",
            })
            self.sendCommand("events.subscribe", {"events": ["DRCamera.photo"]})
            self.sendCommand("camera.capturePhoto")

            image = None
            while image is None:
                packet = self.recv()
                if packet is not None:
                    event = packet["class"] + "." + packet["key"]
                    if event == "DRCamera.photo":
                        #print("Frame captured, fetching...")
                        image_url = f"http://{self.robot_ip}:8080/d3-camera-photo.jpg"
                        image = self.fetch_image(image_url)
                        if image is not None:
                            cv2.imshow("Video Stream", image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
                                cv2.destroyAllWindows()
                                return  



