import time
import cv2
import pickle
from imutils import paths
import os
import numpy as np
import scipy.spatial
from scipy.signal import convolve2d

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
#imagePaths = list(paths.list_images('data'))
start_time = time.time()
imagePaths = list(paths.list_images('/home/pi/Desktop/FKP/dataset'))
BOCVs = []
labels = []
for i in range (0,len(imagePaths)) :
  print(imagePaths[i])
  print("[INFO] Processing Image {}/{}".format(i+1,len(imagePaths)))
  image = cv2.imread(imagePaths[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imgname = imagePaths[i].split(os.path.sep)[6]
  label = imgname.split('_')[0]
  gabor_res = feature_extraction(image)
  BOCV_res = BOCV_cal(gabor_res)
  BOCVs.append(BOCV_res)
  labels.append(label)
print("--- %s seconds ---" % (time.time() - start_time))
data = {"BOCV" : BOCVs, "label" : labels}
f = open('/home/pi/Desktop/FKP/pickle/BOCV_PI.pickle',"wb")
f.write(pickle.dumps(data))
f.close
#print("--- %s seconds ---" % (time.time() - start_time))





