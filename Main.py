import cv2
import numpy as np
import autopy
from cvzone.HandTrackingModule import HandDetector
import time

# Camera settings
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7  # Smoothing value

# Previous and current mouse positions
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Get screen size
wScr, hScr = autopy.screen.size()

while True:
    # Read frame from camera
    success, img = cap.read()
    
    # Find hands in the frame
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    # Check if hands are detected
    if len(lmList) != 0:
        # Get coordinates of index finger
        x1, y1 = lmList[8][0:]
        fingers = detector.fingersUp()
        
        # Draw rectangle around the frame
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # Moving mode (index finger extended)
        if fingers[1] == 1 and fingers[2] == 0:
            # Map hand coordinates to screen coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            
            # Smoothen the cursor movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            # Move mouse cursor
            autopy.mouse.move(wScr - clocX, clocY)
            
            # Draw circle at index finger tip
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            
            # Update previous mouse position
            plocX, plocY = clocX, clocY
        
        # Clicking mode (index and middle fingers extended)
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between index and middle fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            
            # Click mouse if fingers are close
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
        
        # Calculate and display frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
        
    # Display the frame
    cv2.imshow("Image", img)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
exit()
