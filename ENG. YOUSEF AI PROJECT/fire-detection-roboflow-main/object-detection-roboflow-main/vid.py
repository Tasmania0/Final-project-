import io
import cv2
import requests
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from requests_toolbelt.multipart.encoder import MultipartEncoder

class FireDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Fire Detection")
        self.master.config(bg='black')

        self.video_source = 0
        self.cap = None
        self.is_camera_active = False

        self.canvas = tk.Canvas(master, width=640, height=480, bg='black')
        self.canvas.pack()

        self.start_button = tk.Button(master, text="Start Camera", command=self.start_camera, bg='blue', fg='white')
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(master, text="Stop Camera", command=self.stop_camera, bg='blue', fg='white')
        self.stop_button.pack(side=tk.LEFT)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_camera(self):
        if not self.is_camera_active:
            self.cap = cv2.VideoCapture(self.video_source)
            self.is_camera_active = True
            self.update()

    def stop_camera(self):
        if self.is_camera_active:
            self.cap.release()
            self.is_camera_active = False
            self.canvas.delete("all")

    def update(self):
        if self.is_camera_active:
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)
                self.detect_fire(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=pil_image)

                self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
                self.canvas.img_tk = img_tk  # keep a reference
            self.master.after(10, self.update)

    def detect_fire(self, img):
        buffered = io.BytesIO()
        pilImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pilImage.save(buffered, quality=100, format="JPEG")

        try:
            m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
            response = requests.post("https://detect.roboflow.com/fire-and-smoke-detection-vhrkg/1?api_key=VKZx2mEpTLBu8cpFh4h3", 
                                     data=m, headers={'Content-Type': m.content_type})
            data = response.json()

            if data['predictions']:
                self.draw_predictions(img, data['predictions'][0])
        except Exception as e:
            print('No fire detected:', e)

    def draw_predictions(self, img, prediction):
        imWidth = int(prediction['width'])
        imHeight = int(prediction['height'])
        xp = int(prediction['x'])
        yp = int(prediction['y'])
        clasName = prediction['class']

        start_x = xp - (imWidth / 2)
        start_y = yp - (imHeight / 2)
        end_x = xp + (imWidth / 2)
        end_y = yp + (imHeight / 2)

        cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color=(0, 255, 0), thickness=2)
        cv2.putText(img, clasName, (int(start_x), int(start_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def on_closing(self):
        self.stop_camera()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()
