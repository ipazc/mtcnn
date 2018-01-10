from time import sleep
from timeit import default_timer as timer

import cv2
from mtcnn.mtcnn import MTCNN

s = timer()
sleep(1)
second_time = timer() - s


s = timer()
detector = MTCNN()
load_time = timer() - s

def measure(filename):

    s = timer()
    img = cv2.imread(filename)
    result = detector.detect_faces(img)
    run_time = timer() - s
    print(filename.split("/")[-1], run_time, img.shape[0]*img.shape[1], len(result))

print("One second time:", second_time)
print("Load time:", load_time)

measure("474x224_10.jpg")
measure("564x226_10.jpg")
measure("736x348_10.jpg")
measure("2100x994_10.jpg")
