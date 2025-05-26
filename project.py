import cv2
import mediapipe as mp
import time
import PoseLandmark_module as plm

cap = cv2.VideoCapture(r"D:\Computer_Vision\Videos\1.mp4")
pTime = cTime = 0

detection = plm.Poselandmark()

while True:
    success,img = cap.read()
    img = cv2.resize(img,(700,600))

    detection.DetectionPose(img)
    detection.findPosition(img)

    cv2.imshow("Original",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()