import dlib
import cv2
import os
import numpy as np
import pandas as pd

predictor_path = os.getcwd() + '\\dat\\shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(predictor_path)  # 获取人脸检测器.


# 归一化函数
def maxminnorm(array):
    mx = max(array)
    mn = min(array)
    dis = mx - mn
    for i in range(len(array)):
        array[i] = (array[i] - mn) / dis
    return array


# 获得面部特征点的坐标
# 检测不到人脸则返回空列表
def getFaceCoordinate(img):
    coordinate = []

    b, g, r = cv2.split(img)  # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片

    dets = detector(img, 1)  # 使用detector进行人脸检测 dets为返回的结果

    if len(dets) == 0:
        return coordinate
    else:
        for index, face in enumerate(dets):
            shape = predictor(img, face)  # 寻找人脸的68个标定点
            # 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
            x = []
            y = []
            for index, pt in enumerate(shape.parts()):
                # print('Part {}: {}'.format(index, pt))
                x.append(pt.x)
                y.append(pt.y)

            x = maxminnorm(x)
            y = maxminnorm(y)
            for i in range(len(x)):
                coordinate.append(x[i])
                coordinate.append(y[i])
            break
        return coordinate


# output:输出文件名,record_num:要获取的记录数
def getData(output, record_num):
    capture = cv2.VideoCapture(0)
    video_num = 0
    data = []
    while video_num < record_num:
        ref, frame = capture.read()
        coordinate = getFaceCoordinate(frame)
        if len(coordinate) == 136:
            video_num += 1
            data.append(coordinate)
        if video_num % 10 == 0:
            print(video_num)
    df = pd.DataFrame(data)
    df.to_csv(output, index=False, header=True)
    return np.array(data)


normal = 'faceDataNormal.csv'
normal_test = 'faceDataNormalTest.csv'
sleepy = 'faceDataSleepy.csv'
yawn = 'faceDataYawn.csv'
# data = getData(normal,5000)
# data = getData(sleepy, 300)
# data = getData(yawn,300)
# data =  getData(normal_test,300)
