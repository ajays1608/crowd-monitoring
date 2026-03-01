import cv2

print("Attempting to open Laptop Webcam (Source 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("FAILED! Windows is blocking the camera.")
else:
    print("SUCCESS! Camera opened.")
    ret, frame = cap.read()
    if ret:
        print("SUCCESS! Captured a frame.")
    else:
        print("FAILED! Camera opened, but frame is completely blank.")
        
cap.release()
