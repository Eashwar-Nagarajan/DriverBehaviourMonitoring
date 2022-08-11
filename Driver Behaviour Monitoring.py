# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:58:16 2022

@author: eashw
"""
import imutils
import cv2
import mediapipe as mp
import numpy as np
import time
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import datetime
import threading

# define necessary conditions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# define yawn function
def cal_yawn(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = dist.euclidean(top_mean,low_mean)
    return distance
# define EAR
def eye_aspect_ratio(eye):
 	A = dist.euclidean(eye[1], eye[5])
 	B = dist.euclidean(eye[2], eye[4])
 	C = dist.euclidean(eye[0], eye[3])
 	return (A + B) / (2.0 * C)

# start video capture
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)
# define various variables 
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)
EYE_AR_CONSEC_FRAMESS = 1
TOTAL = 0 
total_frames = 1
BR = 0

#define class "queue" for blink rate sliding window
class Queue:
    def __init__(self,maxSize):
        self.maxSize = maxSize
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def display(self):
        print(self.queue)

    def size(self):
        return len(self.queue)
    
    def get_first(self):
      if(len(self.queue) > 0):
        return self.queue[0]
      
    def get_last(self):
      if(len(self.queue) > 0):
        return self.queue[-1]

#define timer
from threading import Timer
import time
class RepeatTimer(Timer):
    def run(self):
          while not self.finished.wait(self.interval):
              self.function(*self.args, **self.kwargs)
q = Queue(60)

#Queue function for blink rate sliding window
def QueueFunc():
    global TOTAL
    global BR
    b = q.size()
    if b < 60:
        q.enqueue(TOTAL)
        b = q.size()
    else:
        q.dequeue()
        q.enqueue(TOTAL)
        c = q.get_first()
        d = q.get_last()
        BR = d-c  
#uninitialize timer
timer_uninit = 1

#define variables
EYE_AR_THRESH = 0.29
EYE_AR_CONSEC_FRAMES = 30
GAZE_CONSEC_FRAMES = 30
GAZE_COUNTER = 0 
COUNTER = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while cap.isOpened():
    success, image = cap.read()
    cv2.imshow("footage", image)
    start = time.time()
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 	# detect faces in the grayscale frame
    rects = face_model(image, 0)
    # loop over the face detections
    for rect in rects:
            # writeable is false for improving performance
            image.flags.writeable = False
            shape = landmark_model(img_gray, rect)
            shape = face_utils.shape_to_np(shape)
		    # initilize eye values for EAR calculation
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            AR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            img_gray.flags.writeable = True
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:				
                    cv2.putText(image, "DROWSINESS ALERT!", (10, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
            else:
                COUNTER = 0
            cv2.putText(image, "EAR: {:.2f}".format(ear), (550, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gray.flags.writeable = False
    faces = face_model(img_gray)
    yawn_thresh = 35
    ptime = 0
    for face in faces:
        #Detect Landmarks
        shapes = landmark_model(img_gray,face)
        shape = face_utils.shape_to_np(shapes)
  
        #Detecting/Marking the lower and upper lip
        # lip = shape[48:60]
        # img_gray.flags.writeable = True
        # cv2.drawContours(image,[lip],-1,(0, 165, 255),thickness=3)
  
        #Calculating the lip distance
        lip_dist = cal_yawn(shape)
        # print(lip_dist)
        
        if lip_dist > yawn_thresh : 
            cv2.putText(image,'User Yawning!',(20,350),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)  
  


    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Improves performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # Improves performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -20:
                text = "Looking Right"
            elif y > 13:
                text = "Looking Left"
            elif x < -15:
                text = "Looking Down"
            elif x > 15:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            # cv2.line(image, p1, p2, (255, 0, 0), 3)
            # canvas = np.zeros((432, 768, 3), dtype="uint8")
            # cv2.arrowedLine(canvas, p1, p2, (255, 0, 0), 3)
            # #cv2.circle(canvas,p2,3,(255,0,0),-1)
            # cv2.imshow("Canvas", canvas)
            
            car_1 = cv2.imread("car1.png", cv2.IMREAD_COLOR)
            car = cv2.resize(car_1, (768, 432)) 
            cv2.arrowedLine(car,p1,p2,(0,255,0),3)
            cv2.imshow("Car", car)
            if text != "Forward":
                GAZE_COUNTER += 1
                if GAZE_COUNTER >= GAZE_CONSEC_FRAMES:				
                    cv2.putText(image, "DISTRACTED!", (10,450),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)	
            else:
                GAZE_COUNTER = 0
            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (550, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            
    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    #print("FPS: ", fps)

    cv2.putText(image, f'FPS: {int(fps)}', (550,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)    


    #cv2.imshow('Driver Focus Detection', image)
    image, faces = detector.findFaceMesh(image, draw=False)
    if timer_uninit == 1:
        timer = RepeatTimer(1, QueueFunc)
        timer.start()
        timer_uninit = 0
    if faces:
        
        #start = time.time()
        total_frames = total_frames + 1
        face = faces[0]
        # for id in idList:
        #     cv2.circle(img, face[id], 5,color, cv2.FILLED)
 
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
 
        #cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        #cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
 
        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 36.5:
            counter += 1
        else:
            if counter >= EYE_AR_CONSEC_FRAMESS:
                TOTAL += 1
            counter = 0 
        if BR < 10 and BR > 0 or BR > 20:
            cv2.putText(image, "Abmornal Blink Rate", (250, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
        cv2.putText(image, "Blink Count:{}".format(TOTAL), (25, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        cv2.putText(image, "Blink Rate:{}".format(BR), (300, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (255, 0, 255), 2)
        
            
        imgPlot = plotY.update(ratioAvg, color)
        image = cv2.resize(image, (480, 360))
        imgStack = cvzone.stackImages([image, imgPlot], 2, 1)
    else:
        image = cv2.resize(image, (480, 360))
        imgStack = cvzone.stackImages([image, image], 2, 1)    
    cv2.imshow("Image", imgStack)

    
    
    #cv2.imshow('Webcam footage', footage)
    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
