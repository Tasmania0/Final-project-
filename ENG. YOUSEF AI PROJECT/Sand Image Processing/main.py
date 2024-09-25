import io
import cv2
import requests
import threading
from PIL import Image, ImageTk
import tkinter as tk
from requests_toolbelt.multipart.encoder import MultipartEncoder

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")
        
        self.video_source = 0
        self.cap = None
        self.running = False

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        self.start_button = tk.Button(self.btn_frame, text="Open Camera", bg="blue", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.btn_frame, text="Close Camera", bg="blue", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(self.video_source)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.canvas.delete("all")

    def update_frame(self):
        if self.running:
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)
                self.process_frame(img)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.root.after(10, self.update_frame)

    def process_frame(self, img):
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pilImage.save(buffered, quality=100, format="JPEG")

        try:
            m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
            response = requests.post("https://detect.roboflow.com/silica_sand/1?api_key=RlvzQiXpEn5NZTeHJ4QK", 
                                     data=m, headers={'Content-Type': m.content_type})
            data = response.json()

            if data['predictions']:
                imWidth = int(data['predictions'][0]['width'])
                imHeight = int(data['predictions'][0]['height'])
                xp = int(data['predictions'][0]['x'])
                yp = int(data['predictions'][0]['y'])
                clasName = data['predictions'][0]['class']

                start_x = xp - (imWidth / 2)
                start_y = yp - (imHeight / 2)
                start_point = (int(start_x), int(start_y))
                end_x = xp + (imWidth / 2)
                end_y = yp + (imHeight / 2)
                end_point = (int(end_x), int(end_y))

                cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, clasName, (int(start_x), int(start_y) - 10), font, 1, color, thickness)

        except Exception as e:
            print('Error:', e)

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
