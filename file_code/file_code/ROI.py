import cv2
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import PIL
import dlib
import time
from time import sleep
import time
import RPi.GPIO as GPIO    
from time import sleep
import imutils
import numpy as np
BUTTON = 16
GPIO.setwarnings(False)    
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT, initial=GPIO.LOW)   
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON,GPIO.IN,pull_up_down=GPIO.PUD_UP)
user_id = input("\n Enter user id :")
print("\n [INFO] Intilialzing finger capture. Please wait ... ")
os.chdir("/home/pi/Desktop/FKP/dataRaw")
os.mkdir(user_id)
os.chdir("/home/pi/Desktop/FKP/dataset")
os.mkdir(user_id)
path = os.path.join("/home/pi/Desktop/FKP/dataset",user_id)
os.chdir(path)
def record_video():
    print('Recording...Please wait...')
    #data = time.strftime("%d_%b_%Y\%H:%M:%S")
    camera.start_preview()
    camera.start_recording('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
    camera.wait_recording(2)
    camera.stop_recording()
    camera.stop_preview()
    print('Video recorded successfully')
def sharp(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
camera = PiCamera()
camera.resolution = (960,720)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(960,720))

#(380, 500, 3)
def scale(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    return new_img


def crop_image(row,img):
    img  = np.array(img, dtype=np.uint8)
    img_crop = img[row-300: row,:]
    return img_crop

def step_2(img_path):
    img = cv2.imread(img_path,0)
    origin = img
    y0=img.shape[0]
    img_crop = crop_image(y0-30,img)
    #cv2_imshow(im)
    return img_crop,y0,origin
def step_3(img, min_val, ksize=3):
    #3. Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    edge_gradient = np.sqrt(Gx*Gx + Gy*Gy)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi 
    # round angle to 4 directions
    angle = np.abs(angle)
    for x in range(angle.shape[0]):
      for y in range(angle.shape[1]):
        if ((angle[x,y] <=22.5) or (angle[x,y] >= 157.5)):
          angle[x,y] = 0
        elif ((angle[x,y] > 22.5) and (angle[x,y] < 67.5)):
          angle[x,y] = 45
        elif ((angle[x,y] >= 67.5) and (angle[x,y] <= 112.5)):
          angle[x,y] = 90
        else :
          angle[x,y] = 135    
    #4. Non-maximum Suppression
    canny_raw = np.zeros(img.shape, np.uint8)
    for x in range(1, edge_gradient.shape[0]-1):
        for y in range(1, edge_gradient.shape[1]-1):
            if angle[x,y] == 0:
                if edge_gradient[x,y] <= max(edge_gradient[x,y-1], edge_gradient[x,y+1]):
                    edge_gradient[x,y] = 0
            elif angle[x,y] == 45:
                if edge_gradient[x,y] <= max(edge_gradient[x+1,y-1], edge_gradient[x-1,y+1]):
                    edge_gradient[x,y] = 0
            elif angle[x,y] == 90:
                if edge_gradient[x,y] <= max(edge_gradient[x-1,y], edge_gradient[x+1,y]):
                    edge_gradient[x,y] = 0
            elif angle[x,y] == 135:
                if edge_gradient[x,y] <= max(edge_gradient[x-1,y-1], edge_gradient[x+1,y+1]):
                    edge_gradient[x,y] = 0
    #5. Hysteresis Thresholding    
    canny_mask = np.zeros(img.shape, np.uint8)
    for x in range (img.shape[0]):
      for y in range (img.shape[1]):
        if edge_gradient[x,y]>=min_val:
          canny_mask[x,y]=255
    return scale(canny_mask),canny_mask

def step_4(ie):
    img = ie
    ymid = img.shape[0] / 2
    icd = np.zeros((img.shape[0], img.shape[1]))
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if ((int(img[i, j]) == 0) or (int(img[i + 1, j - 1]) == 1 and int(img[i + 1, j + 1] == 1))):
                icd[i, j] = 0
            elif ((int(img[i + 1, j - 1]) == 1) and (i <= ymid)) or (int(img[i + 1, j + 1]) == 1 and i > ymid):
                icd[i, j] = 1
            elif ((int(img[i + 1, j + 1]) == 1) and (i <= ymid))  or (int(img[i + 1, j - 1]) == 1 and i > ymid):
                icd[i, j] = -1
    return icd

def calc_conMag(x):
    res = np.absolute(np.sum(icd[:,x-17:x+18]))
    return res
def step_5():
    conMag = np.ones(icd.shape[1])
    conMag*=1000
    for i in range(200,350):
            conMag[i] = calc_conMag(i)
    x0 = np.argmin(conMag)
    return x0,conMag
def step_6(origin,x0,y0):
    IROI = origin[y0-280-30:y0-30,x0-140:x0+140]
    return IROI


for frame in camera.capture_continuous(rawCapture,format = "bgr",use_video_port = True):
    image = frame.array
    show = image
    cv2.imshow("Crop",show[270:650,200:700])

    ButtonState = GPIO.input(BUTTON)
    if (ButtonState == 0) or (cv2.waitKey(1) & 0xff == ord("q")):
        cv2.destroyAllWindows()
        record_video()
        print('Spliting...Please wait...')
        start_time = time.time()
        cap = cv2.VideoCapture('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
        currentFrame = 0
        count = 0
        while(True):
            ret,frame = cap.read()
            if ret:
                currentFrame +=1
                if(currentFrame%10==5):
                    count+=1
                    print("Splitting {}/5 images ...".format(count))
                    name = '/home/pi/Desktop/FKP/dataset/' + user_id + '/' + user_id + '_' + str(count) + '.jpg'
                    name2 = '/home/pi/Desktop/FKP/dataRaw/' + user_id + '/' + user_id + 'Raw_' + str(count) + '.jpg'

                    #frame = frame[x:x+h,y:y+w]
                    frame = sharp(frame[270:650,200:700])
                    cv2.imwrite(name2,frame)
                    img_crop,y0,origin =step_2(name2)
                    ie,canny_mask = step_3(img_crop,10)
                    icd = step_4(ie)
                    x0, conMag = step_5()
                    print(x0)
                    IROI = step_6(origin,x0,y0)
                    #draw = cv2.line(frame, (0,y0-30), (500,y0-30), (0,0,255), thickness=2)
                    #draw = cv2.line(draw, (x0,0), (x0,380), (0,0,255), thickness=2)
                    #cv2.imshow(IROI)
                    #cv2.imwrite("/home/pi/Desktop/FKP/"+user_id + '_'+str(count)+".jpg",draw)
                    #cv2.imwrite(name, IROI)
            else:
                break
        print("All Done!")
        print("--- %s seconds ---" % (time.time() - start_time))
        cap.release()
        cv2.destroyAllWindows()
        exit()
    if cv2.waitKey(1) & 0xff == ord("e"):
        exit()
    rawCapture.truncate(0)


