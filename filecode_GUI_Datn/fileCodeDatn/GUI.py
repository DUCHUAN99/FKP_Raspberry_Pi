
import subprocess
from PIL import Image,ImageTk
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import PIL
import time
from time import sleep
import RPi.GPIO as GPIO    
import imutils
import numpy as np
import scipy.spatial
from scipy.signal import convolve2d
import pickle
import tkinter as tk
import threading
from queue import Queue
from Matching import check
from coi import coi_keu,coi_keu_sai

def run_code1():
    global print_queue
    print_queue.put("\nĐang xử lí..... Vui lòng chờ trong giây lát !")
    text,value = check() 
    if(value ==1):
        coi_keu(3)
        print_queue.put(text)
    else:
        coi_keu_sai()
        print_queue.put(text)
def print_output():
    global print_queue
    while True:
        message = print_queue.get()
        text_output.insert(tk.END, message + '\n')
        text_output.see(tk.END)  # Cuộn xuống cuối cùng

def on_button_click():
    thread_video = threading.Thread(target=run_code1)
    thread_print = threading.Thread(target=print_output) 
    thread_video.start()
    thread_print.start()
#Tạo cửa sổ
root = tk.Tk()
root.title("Hệ thống nhận diện khớp ngón tay")

# Chỉnh kích thước của cửa sổ
window_width = 1024
window_height = 768
# chiều rộng và chiều cao của màn hình máy tính
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

#khoảng cách từ cạnh màn hình đến cạnh của cửa sổ đồ họa
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

#Tạo Frame để chứa các button
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP)

# Tạo button để chạy file code 1
button1 = tk.Button(button_frame, text="Kiểm Tra Khớp Ngón Tay", command= on_button_click,font=("Arial", 15, "bold"),bg="#c7a47e")
button1.pack(side=tk.LEFT, padx=10, pady=10)
# Tạo button để chạy file code 2
#button2 = tk.Button(button_frame, text="Thêm Ảnh Vào Cơ Sở Dữ Liệu", command=run_code2,font=("Arial", 15, "bold"),bg="#c7a47e")
#button2.pack(side=tk.LEFT, padx=10, pady=10)

# Tạo cửa sổ để hiển thị kết quả
text_output = tk.Text(root, height=100, width=50,font=("Arial", 30))
text_output.pack()
print_queue = Queue()
# Bắt đầu vòng lặp chạy cửa sổ
root.mainloop()




    
    
    
    
