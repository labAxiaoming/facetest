# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:01:15 2019

@author: xm
"""

# thanks https://github.com/yinguobing/head-pose-estimation
# and thanks https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/


import cv2 
import numpy as np
import face_recognition  #pip install dlib  and pip install face_recognition

# Read Image
im = cv2.imread("headPose.jpg");

im = cv2.imread("timg.jpg");
im = cv2.imread("lj.jpg");


face_loc =  face_recognition.face_locations(im)
ss2 = face_recognition.face_landmarks(im,face_loc, model="large")

size = im.shape
for i in range(len(face_loc)): # how mush face
    image_points = []
    image_points.append(ss2[i]['nose_bridge'][3])
    image_points.append(ss2[i]['chin'][8])
    image_points.append(ss2[i]['left_eye'][0])
    image_points.append(ss2[i]['right_eye'][3])
    image_points.append(ss2[i]['top_lip'][0])
    image_points.append(ss2[i]['bottom_lip'][11])
    image_points = np.array(image_points,dtype="double")
    
    
         
    #2D image points. If you change the image, you need to change vector
    #image_points = np.array([
    #                            (359, 391),     # Nose tip鼻尖  #'nose_tip': [(352, 405), (364, 414), (379, 419), (400, 416), (422, 411)],
    #                            (399, 561),     # Chin下巴
    #                            (337, 297),     # Left eye left corner
    #                            (513, 301),     # Right eye right corne
    #                            (345, 465),     # Left Mouth corner
    #                            (453, 469)      # Right mouth corner
    #                        ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])/4.5
     
    #image_points= image_points2 
    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    print ("Camera Matrix :\n {0}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)
     
    print ("Rotation Vector:\n {0}".format(rotation_vector))
    print ("Translation Vector:\n {0}".format(translation_vector))
     
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
     
     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (255,255,255), -1)
     
     
    #p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
     
    #cv2.line(im, p1, p2, (255,255,0), 4)
    
    
    image = im
    point_3d = []
    rear_size = 75
    rear_depth = 0
    color=(205, 205, 255)
    line_width=2
    dist_coeefs = np.zeros((4, 1))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
    
 
# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()




