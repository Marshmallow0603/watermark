import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
from model import DetectInpaint


class PaintApp:
    def __init__(self, master):
        self.master = master
        self.load_image()
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_mask)
        self.mask_button = tk.Button(master, text="Create Mask", command=self.create_mask)
        self.mask_button.pack()
        self.mask_button = tk.Button(master, text="Start", command=self.detectInpaint)
        self.mask_button.pack()
        self.brush_size = 5
        self.brush_size_scale = tk.Scale(master, from_=1, to=100, orient=tk.HORIZONTAL, command=self.change_brush_size)
        self.brush_size_scale.pack()
        self.mask = None 

    def load_image(self):
        image_path = filedialog.askopenfilename(title="Select Image")
        self.image = Image.open(image_path)
        self.canvas = tk.Canvas(self.master, width=self.image.width, height=self.image.height, scrollregion=(0,0,500,500))
        self.canvas.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor=tk.NW)

    def draw_mask(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        if self.mask is None:
            self.mask = np.zeros((self.image.height, self.image.width, 4), dtype=np.uint8)
        cv2.circle(self.mask, (x, y), r, (255, 255, 255, 255), -1)

    def create_mask(self):
        if self.mask is not None:
            cv2.imwrite("mask.png", self.mask)
            
    def detectInpaint(self):
        detect_inpaint = DetectInpaint(
            image=self.image,
            mask = self.mask,
            use_cuda_if_available=False 
        )
        image_inpaint = detect_inpaint.run()
        image_inpaint.show()

    def change_brush_size(self, value):
        self.brush_size = int(value)


root = tk.Tk()
app = PaintApp(root)
root.mainloop()







