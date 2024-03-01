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
from myFunction import scale,sharp,crop_image,step_2,step_3,distance_cal
from myFunction import step_4,step_6,feature_extraction,BOCV_cal
from coi import coi_keu,coi_keu_sai
from queue import Queue

def check():
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
    pklfile = pickle.loads(open("/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle","rb").read())
    LED1 = 11
    LED2 = 13
    BUTTON = 16
    GPIO.setwarnings(False)    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED1, GPIO.OUT, initial=GPIO.LOW)   
    GPIO.setup(LED2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON,GPIO.IN,pull_up_down=GPIO.PUD_UP)


    camera = PiCamera()
    camera.resolution = (960,720)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(960,720))

    #print('[INFO] Start The FKP Capturing Process. Please Wait...')
    for frame in camera.capture_continuous(rawCapture,format = "bgr",use_video_port = True):
        image = frame.array
        cv2.imwrite("/home/pi/Desktop/FKP/test/test.jpg",image[270:650,200:700])
        show = image
        show = show[270:650,200:700]
        #cv2.line(show,(0,380-15),(500,380-15),(255,0,0),thickness=2)
        cv2.imshow("Crop",cv2.flip(show,1))

        ButtonState = GPIO.input(BUTTON)
        if (ButtonState == 0) or (cv2.waitKey(1) & 0xff == ord("q")):
            
            start_time = time.time()
            #print('FKP Capturing Process Is Complete! ')
            #cv2.imwrite("/home/pi/Desktop/FKP/test/test.jpg",crop)
            im_show = image
            image = sharp(image[270:650,200:700])
            cv2.destroyAllWindows()
            camera.close()
            img_path="/home/pi/Desktop/FKP/test/test.jpg"
            #frame = frame[x:x+h,y:y+w]
            #cv2.imwrite(name2,image)
            img_crop,y0,origin =step_2(img_path)
            ie,canny_mask = step_3(img_crop,10)
            icd = step_4(ie)
            x0, conMag = step_5()
            #print(x0)
            IROI = step_6(origin,x0,y0)
            timeROI = time.time() - start_time
            print("--- ROI Extraction: %s seconds " % (np.round((time.time() - start_time),3)))
            cv2.imwrite("/home/pi/Desktop/FKP/test/IROI.jpg",IROI)
            start_time = time.time()
            #image = sharp(crop)
            image=IROI
            #image = cv2.cvtColor(IROI, cv2.COLOR_BGR2GRAY)
            gabor_res = feature_extraction(image)
            BOCV_res = BOCV_cal(gabor_res)
            timeFE = time.time() - start_time
            print("--- Feature Extraction: %s seconds " % (np.round((time.time() - start_time),3)))
            distance = []
            start_time = time.time()
            for i in range(0, len(pklfile["label"])):
                D = distance_cal(BOCV_res,pklfile['BOCV'][i])
                distance.append(D)
            #print('Matching Process Is Complete! ')
            timeMatch = time.time() - start_time
            idx = np.argmin(distance)
            print("Distance_Min: ",np.min(distance))
            if np.min(distance) > 0.135 : #0.12
                print("--- Matching 1v1: %s seconds " % (np.round((time.time() - start_time)/len(pklfile["label"]),3)))
                #print('--- Min Distance: '+str(np.round(np.min(distance),3))+' ---')
                print("------------------------------------------------")
                print('--- Unrecognizable. Please Try Again!')
                print("------------------------------------------------")
                im_show = im_show[270:650,200:700]
                im_show = cv2.putText(im_show, "unknown", (x0-60,y0-15-280-20),cv2.FONT_HERSHEY_SIMPLEX , 
                       1, (0,0,255), 2, cv2.LINE_AA)
                #im_show = cv2.rectangle(im_show,(x0+140,y0-15),(x0-140,y0-280-15),(255,0,160), thickness=2)
                im_show = cv2.rectangle(im_show, (x0-140,y0-15-280), (x0+140,y0-15), (0,0,255), thickness=2)
                #IROI = origin[y0-280-15:y0-15,x0-140:x0+140]
                cv2.imwrite("/home/pi/Desktop/FKP/test/test3.jpg",im_show)
                #im_show = cv2.putText(im_show, "Unknown", (470,340),cv2.FONT_HERSHEY_SIMPLEX , 
                       #1, (0,0,255), 2, cv2.LINE_AA)
                #im_show = cv2.rectangle(im_show,(x,y),(x+w,y+h),(0,255,0),2)
                
                #cv2.imwrite("/home/pi/Desktop/FKP/test/utsai.jpg",im_show[230:680,:])
            
                return'Không nhận diện được, vui lòng thử lại',0
                exit()
            elif pklfile["label"][idx] == 'nothing':
                print("--- Matching 1v1: %s seconds " % (np.round((time.time() - start_time)/len(pklfile["label"]),3)))
                print("------------------------------------------------")
                print('--- There Are Nothing In Camera. Please Try Again! ')
                print("------------------------------------------------")
                exit()
            else :
                print("--- Matching 1v1: %s seconds ---" % (np.round((time.time() - start_time)/len(pklfile["label"]),3)))
                #print('--- Min Distance: '+str(np.round(np.min(distance),3))+' ---')
                print("------------------------------------------------")
                print('WELCOME: ' +str(pklfile["label"][idx])+'!')
                
                im_show = im_show[270:650,200:700]
                im_show = cv2.putText(im_show, pklfile["label"][idx], (x0-15,y0-15-280-20),cv2.FONT_HERSHEY_SIMPLEX , 
                       1, (0,0,255), 2, cv2.LINE_AA)
                #im_show = cv2.rectangle(im_show,(x0+140,y0-15),(x0-140,y0-280-15),(255,0,160), thickness=2)
                im_show = cv2.rectangle(im_show, (x0-140,y0-15-280), (x0+140,y0-15), (0,0,255), thickness=2)
                #IROI = origin[y0-280-15:y0-15,x0-140:x0+140]
                cv2.imwrite("/home/pi/Desktop/FKP/test/test3.jpg",im_show)
                print("------------------------------------------------")
                
                return 'xin chào:  ' +str(pklfile["label"][idx])+'!',1
                exit()
        if cv2.waitKey(1) & 0xff == ord("e"):
            exit()
        rawCapture.truncate(0)

#a,b = check()

           