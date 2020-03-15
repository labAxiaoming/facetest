# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:59:36 2019

@author: xm
"""



#import torch
import cv2 as cv
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dataset = 'COCO'
if dataset == 'COCO':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

else:

    assert(dataset == 'MPI')
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }



    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

inWidth = 368
inHeight = 368
thr = 0.1


import torch
from src import util

torchmodelfile = 'model/body_pose_model.pth'
from src.model import bodypose_model
torchnet =  bodypose_model()
if torch.cuda.is_available():
    torchmodel = torchnet.cuda()
else:
    torchmodel = torchnet.cpu()
model_dict = util.transfer(torchmodel, torch.load(torchmodelfile))
torchmodel.load_state_dict(model_dict)
torchmodel.eval()
        



frame = cv.imread("images/demo.jpg")


frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
stride = 8
padValue = 128

#imageToTest = cv.resize(frame, (0, 0), fx=368, fy=368, interpolation=cv.INTER_CUBIC)
imageToTest = cv.resize(frame, (368,368) )

imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
im = np.ascontiguousarray(im)

data = torch.from_numpy(im).float()
if torch.cuda.is_available():
    data = data.cuda()
# data = data.permute([2, 0, 1]).unsqueeze(0).float()
with torch.no_grad():
    Mconv7_stage6_L1, Mconv7_stage6_L2 = torchmodel(data)
Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

heatmap2 = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
out = heatmap2
points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponging body's part.
    heatMap = out[ :, :,i]

    _, conf, _, point = cv.minMaxLoc(heatMap)

    x = (frameWidth * point[0]) / out.shape[1]

    y = (frameHeight * point[1]) / out.shape[0]



    # Add a point if it's confidence is higher than threshold.

    points.append((x, y) if conf > thr else None)



for pair in POSE_PAIRS:

    partFrom = pair[0]

    partTo = pair[1]

    assert(partFrom in BODY_PARTS)

    assert(partTo in BODY_PARTS)



    idFrom = BODY_PARTS[partFrom]

    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:

        x1, y1 = points[idFrom]

        x2, y2 = points[idTo]

        cv.line(frame, (np.int32(x1), np.int32(y1)), (np.int32(x2), np.int32(y2)), (0, 255, 0), 3)

        cv.ellipse(frame, (np.int32(x1), np.int32(y1)), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        cv.ellipse(frame, (np.int32(x2), np.int32(y2)), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)





cv.imshow('OpenPose using OpenCV', frame)
cv.waitKey(0)
cv.destroyAllWindows()