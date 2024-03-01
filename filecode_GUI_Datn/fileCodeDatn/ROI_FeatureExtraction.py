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
from myFunction import scale,sharp,crop_image,step_2,step_3,distance_cal
from myFunction import step_4,step_6,feature_extraction,BOCV_cal

def record_video():
    print('Recording...Please wait...')
    #data = time.strftime("%d_%b_%Y\%H:%M:%S")
    camera.start_preview()
    camera.start_recording('/home/pi/Desktop/FKP/dataset/' + user_id + '.h264')
    camera.wait_recording(2)
    camera.stop_recording()
    camera.stop_preview()
    print('Video recorded successfully')

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

start_time = time.time()
BUTTON = 16
GPIO.setwarnings(False)    
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT, initial=GPIO.LOW)   
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON,GPIO.IN,pull_up_down=GPIO.PUD_UP)

user_id = input("Enter UserId: ")
print("\n[INFO] Intilialzing finger capture. Please wait ... ")
os.chdir("/home/pi/Desktop/FKP/dataRaw")
os.mkdir(user_id)
os.chdir("/home/pi/Desktop/FKP/dataset")
os.mkdir(user_id)
path = os.path.join("/home/pi/Desktop/FKP/dataset",user_id)
os.chdir(path)


camera = PiCamera()
camera.resolution = (960,720)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(960,720))



pklfile = pickle.loads(open("/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle","rb").read())
BOCVs = pklfile['BOCV']
labels = pklfile['label']
All_label = []
for frame in camera.capture_continuous(rawCapture,format = "bgr",use_video_port = True):
    image = frame.array
    show = image
    cv2.imshow("Crop",show[270:650,200:700])
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
                if((currentFrame%4==2) and (currentFrame>2) and (currentFrame<45)):
                    count+=1
                    print("ROI & Feature Extractioning {}/10 images ...".format(count))
                    name = '/home/pi/Desktop/FKP/dataset/' + user_id + '/' + user_id + '_' + str(count) + '.jpg'
                    name2 = '/home/pi/Desktop/FKP/dataRaw/' + user_id + '/' + user_id + 'Raw_' + str(count) + '.jpg'

                    #frame = frame[x:x+h,y:y+w]
                    frame = sharp(frame[270:650,200:700])
                    cv2.imwrite(name2,frame)
                    img_crop,y0,origin =step_2(name2)

                    ie,canny_mask = step_3(img_crop,10)
                    icd = step_4(ie)
                    x0, conMag = step_5()
                    #print(x0)
                    IROI = step_6(origin,x0,y0)
                    #cv2.imshow(IROI)
                    cv2.imwrite(name, IROI)
                    image = IROI
                    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #print("Extracting...")
                    gabor_res = feature_extraction(image)
                    BOCV_res = BOCV_cal(gabor_res)
                    BOCVs.append(BOCV_res)
                    labels.append(user_id)
            else:
                break
        print('Complete!')
        data = {"BOCV" : BOCVs, "label" : labels}
        f = open('/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle',"wb")
        f.write(pickle.dumps(data))
        f.close
        print(len(labels))
        for i in range(len(labels)):
            if i % 10 == 0:
                All_label.append(labels[i])
        print('----------------------------------')
        print("All Label: "+str(All_label))
        print("Total Processing Time: "+str(np.round((time.time() - start_time),2))+'(s)')
        #cap.release()
        exit()
    if cv2.waitKey(1) & 0xff == ord("e"):
        exit()
    rawCapture.truncate(0)