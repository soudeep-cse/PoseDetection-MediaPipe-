import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(r"D:\Computer_Vision\Videos\1.mp4")

pTime = cTime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()
    img = cv2.resize(img,(700,600))
    imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = pose.process(imgRgb)

    # Detect landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        # Find the points
        for id, lms in enumerate(results.pose_landmarks.landmark):
            # print(id,lms)
            h,w,ch=img.shape

            cx = int(lms.x*w)
            cy = int(lms.y*h)

            cv2.circle(img,(cx,cy),10,(0,255,0),-1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,250), 2)

    cv2.imshow("Original",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()