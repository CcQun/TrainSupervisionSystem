import cv2 as cv
import os
import numpy as np

num_class = 8

def dataset():
    data = []
    label = []
    for i in range(num_class):
        path = os.getcwd() + "\\Data\\mov" + str(i + 1) + "\\"
        img_names = os.listdir(path)
        num_record = int(len(img_names) / 15)
        for j in range(num_record):
            item = []
            for k in range(15):
                image = cv.imread(path + str(j) + '-' + str(k) + '.png', cv.IMREAD_GRAYSCALE)
                item.append(image)
            data.append(item)
            label.append(i)
    return np.array(data).transpose(0,2,3,1).reshape((-1,128,128,15,1)), np.array(label)
