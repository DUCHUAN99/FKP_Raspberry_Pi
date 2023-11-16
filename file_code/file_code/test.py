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
import pickle
import scipy.spatial
from scipy.signal import convolve2d
start_time = time.time()

BUTTON = 16
GPIO.setwarnings(False)    
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT, initial=GPIO.LOW)   
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON,GPIO.IN,pull_up_down=GPIO.PUD_UP)

user_id = input("Enter Username: ")
os.chdir("/home/pi/Desktop/FKP/dataset")
os.mkdir(user_id)
path = os.path.join("/home/pi/Desktop/FKP/dataset",user_id)
os.chdir(path)
def record_video():
    print('[INFO] Start The Data Collection Process. Please Wait...')
    #data = time.strftime("%d_%b_%Y\%H:%M:%S")
    camera.start_preview()
    camera.start_recording('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
    camera.wait_recording(2)
    camera.stop_recording()
    camera.stop_preview()
    print('Data Collection Process Is Complete!')
    print('----------------------------------')
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
    res = np.zeros((230,230))
    for x in range(230) :
      for y in range(230) :
        #print(gabor[i][x,y])
        if gabor_res[i][x,y] < 0 :
          res[x,y] = 1
    res = res.astype(np.int)
    #print(res)
    BOCV_res.append(res)
  return BOCV_res
pklfile = pickle.loads(open("/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle","rb").read())
BOCVs = pklfile['BOCV']
labels = pklfile['label']
All_label = []
for frame in camera.capture_continuous(rawCapture,format = "bgr",use_video_port = True):
    image = frame.array
    y=350; x=320; w=230; h=230 
    crop = image[x:x+h,y:y+w]
    cv2.imshow("Crop",crop)
    ButtonState = GPIO.input(BUTTON)
    if (ButtonState == 0) or (cv2.waitKey(1) & 0xff == ord("q")):
        #cap.release()
        cv2.destroyAllWindows()
        record_video()
        print('[INFO] Start The Feature Extraction Process. Please Wait...')
        cap = cv2.VideoCapture('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
        currentFrame = 0
        count=0
        while(True):
            ret,frame = cap.read()
            if ret:
                currentFrame +=1
                if(currentFrame%10==5):
                    count+=1
                    name = '/home/pi/Desktop/FKP/dataset/' + user_id + '/' + user_id + '_' + str(count) + '.jpg'
                    frame = frame[x:x+h,y:y+w]
                    frame = sharp(frame)
                    cv2.imwrite(name, frame)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    print("Extracting Feature Of Image "+ user_id + '_' + str(count) + '.jpg')
                    gabor_res = feature_extraction(image)
                    BOCV_res = BOCV_cal(gabor_res)
                    BOCVs.append(BOCV_res)
                    labels.append(user_id)
            else:
                break
        print('Feature Extraction Process Is Complete!')
        data = {"BOCV" : BOCVs, "label" : labels}
        f = open('/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle',"wb")
        f.write(pickle.dumps(data))
        f.close
        for i in range(len(labels)):
            if i % 5 == 0:
                All_label.append(labels[i])
        print('----------------------------------')
        print("All Label: "+str(All_label))
        print("Total Processing Time: "+str(np.round((time.time() - start_time),2))+'(s)')
        #cap.release()
        exit()
    if cv2.waitKey(1) & 0xff == ord("e"):
        exit()
    rawCapture.truncate(0)





