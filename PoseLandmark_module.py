import cv2
import mediapipe as mp
import numpy as np
import time

class Poselandmark():
    def __init__(self, mode=False, upBody=False, smooth=True, dcon=0.5,trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.dcon = dcon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def DetectionPose(self,img,draw=True):
        imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRgb)
        
        # Detect landmarks
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self,img,draw=True):
        lmList = []
        for id, lms in enumerate(self.results.pose_landmarks.landmark):
            h,w,ch=img.shape

            cx = int(lms.x*w)
            cy = int(lms.y*h)

            lmList.append([id,cx,cy])
            
            if draw:
                cv2.circle(img,(cx,cy),10,(0,255,0),-1)
        
        return lmList

def main():
    pTime = cTime = 0
    
    cap = cv2.VideoCapture(r"D:\Computer_Vision\Videos\1.mp4")
    detection = Poselandmark()

    while True:
        success,img = cap.read()
        img = cv2.resize(img,(700,600))

        img = detection.DetectionPose(img)
        detection.findPosition(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,250), 2)

        cv2.imshow("Original",img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
