import PIL
import tkinter as tk
from tkinter import filedialog
import RPi.GPIO as GPIO
import time

def coi_keu(so_lan):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    pin_number = 15
    GPIO.setup(pin_number,GPIO.OUT)
    i = 0 
    while(i<so_lan):
        GPIO.output(pin_number, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(pin_number, GPIO.LOW)
        i = i +1
        time.sleep(0.1)

def coi_keu_sai():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    pin_number = 15
    GPIO.setup(pin_number,GPIO.OUT)
    i = 0 
    while(i<1):
        GPIO.output(pin_number, GPIO.HIGH)
        time.sleep(3)
        GPIO.output(pin_number, GPIO.LOW)
        i = i +1
        



