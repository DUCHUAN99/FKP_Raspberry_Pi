from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import PIL
#import dlib
import time
from time import sleep
import RPi.GPIO as GPIO    
import imutils
import numpy as np
import scipy.spatial
from scipy.signal import convolve2d
import pickle
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import PIL
#import dlib
import time
from time import sleep
import RPi.GPIO as GPIO    
import imutils
import numpy as np
import pickle
import scipy.spatial
from scipy.signal import convolve2d

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

def feature_extraction(input_image, no_theta=6, sigma=5.3, delta=3.3):
    theta = np.arange(1, no_theta + 1) * np.pi / no_theta
    (x, y) = np.meshgrid(np.arange(0, 35, 1), np.arange(0, 35, 1))
    xo, yo = np.shape(x)[0] / 2, np.shape(x)[0] / 2
    kappa = np.sqrt(2. * np.log(2.)) * ((np.power(2, delta) + 1.) / ((np.power(2, delta) - 1)))
    omega = kappa / sigma

    Psi = {}  # where the filters are stored
    gabor_responses = []
    for i in range(0, len(theta)):
        xp = (x - xo) * np.cos(theta[i]) + (y - yo) * np.sin(theta[i])
        yp = -(x - xo) * np.sin(theta[i]) + (y - yo) * np.cos(theta[i])
        # Directional Gabor Filter
        Psi[str(i)] = (omega / (np.sqrt(2 * np.pi* kappa)) ) * \
                      np.exp(
                          (-np.power(omega, 2) / (8 * np.power(kappa, 2))) * (4 * np.power(xp, 2) + np.power(yp, 2))) * \
                      (np.cos(omega * xp) - np.exp(-np.power(kappa, 2) / 2))
        filtered = convolve2d(input_image, Psi[str(i)], mode='same', boundary='symm')
        gabor_responses.append(filtered)
    #Extract ImcompCode
    gabor_res = np.array(gabor_responses)
    return gabor_res

def BOCV_cal(gabor_res):
  BOCV_res = []
  for i in range (6) :
    res = np.zeros((280,280))
    for x in range(280) :
      for y in range(280) :
        #print(gabor[i][x,y])
        if gabor_res[i][x,y] < 0 :
          res[x,y] = 1
    res = res.astype(np.int)
    #print(res)
    BOCV_res.append(res)
  return BOCV_res

def distance_cal(BOCV1,BOCV2) :
  res = np.bitwise_and(np.bitwise_xor(BOCV1,BOCV2),1)
  D = np.sum(res)/(6*280*280)
  return D


def record_video():
    print('Recording...Please wait...')
    #data = time.strftime("%d_%b_%Y\%H:%M:%S")
    camera.start_preview()
    camera.start_recording('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
    camera.wait_recording(2)
    camera.stop_recording()
    camera.stop_preview()
    print('Video recorded successfully')