import os
import random

import cv2 as cv
import numpy as np
import pandas as pd

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

thr = 0.2
inWidth = 368
inHeight = 368

net = cv.dnn.readNetFromTensorflow("dat\\graph_opt.pb")

cap = cv.VideoCapture(0)


def get_points(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    return points


def show_points(frame, points):
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', frame)


# 获取前50帧有右手腕坐标的坐标并保存到csv文件中
def get_rwrist_scope_data(output):
    data50 = []
    points = []
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()

        if not hasFrame:
            cv.waitKey()
            break

        points = get_points(frame)

        assert (len(points) == 19)

        if points[BODY_PARTS["RWrist"]]:
            # if points[BODY_PARTS["Nose"]]:
            data50.append(points)

        show_points(frame, points)

        if len(data50) == 50:
            df = pd.DataFrame(data50)
            df.to_csv(output, index=False, header=True)
            break


# 得到右手的起止范围 返回上、下、左、右的边界
def get_rwrist_scope(csv):
    def get_tuple(string):
        return (int(string[1:-1].split(',')[0]), int(string[1:-1].split(',')[1]))

    def get_rwrist_points():
        df = pd.read_csv(csv)
        wrist_points = []
        for i in range(df.shape[0]):
            # wrist_points.append(get_tuple(df.iloc[i, 0]))
            wrist_points.append(get_tuple(df.iloc[i, 4]))
        return np.mean(wrist_points, axis=0)

    center = get_rwrist_points()
    return center[1] + 20, center[1] - 20, center[0] - 20, center[0] + 20


def getData(path, record_num):
    u, d, l, r = get_rwrist_scope('RWistScopo.csv')

    # return:0-右手腕坐标为空，1-右手腕在起止范围内，2-右手腕不在起止范围内
    def judge_in_bound(points):
        if points[BODY_PARTS["RWrist"]]:
            x = points[BODY_PARTS["RWrist"]][0]
            y = points[BODY_PARTS["RWrist"]][1]

            # if points[0]:
            #     x = points[0][0]
            #     y = points[0][1]

            if x >= l and x <= r and y >= d and y <= u:
                return 1
            else:
                return 2
        else:
            return 0

    n = 0
    frames = []
    collecting = False
    while cv.waitKey(1) < 0:
        print(collecting, len(frames))
        hasFrame, frame = cap.read()

        if not hasFrame:
            cv.waitKey()
            break

        points = get_points(frame)
        show_points(frame, points)

        judge = judge_in_bound(points)
        if not collecting and judge == 2:
            collecting = True
            frames.append(frame)
        if collecting and judge != 1:
            frames.append(frame)
        if collecting and judge == 1:
            if len(frames) < 15:
                frames = []
                collecting = False
            else:
                resultList = random.sample(range(0, len(frames)), 15)
                mov = []
                for i in range(len(resultList)):
                    cv.imwrite(path + str(n) + '-' + str(i) + '.png',
                               cv.resize(cv.cvtColor(frames[resultList[i]], cv.COLOR_BGR2GRAY), (128, 128)))
                n += 1
                frames = []
                collecting = False
                if n == record_num:
                    break

# get_rwrist_scope_data('RWistScopo.csv')

# getData(os.getcwd() + "\\Data\\mov1\\", 2)
# getData(os.getcwd() + "\\Data\\mov2\\", 2)
# getData(os.getcwd() + "\\Data\\mov3\\", 2)
# getData(os.getcwd() + "\\Data\\mov4\\", 2)
# getData(os.getcwd() + "\\Data\\mov5\\", 2)
# getData(os.getcwd() + "\\Data\\mov6\\", 2)
# getData(os.getcwd() + "\\Data\\mov7\\", 2)
# getData(os.getcwd() + "\\Data\\mov8\\", 2)