import cv2


# Set the desired frame size
frame_width = 1280
frame_height = 720

# Open the camera capture device and set the frame size
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://admin:mgasa1234!.@192.168.1.108/cam/realmonitor?channel=1&subtype=0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
