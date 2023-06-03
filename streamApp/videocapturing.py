# Open the camera capture device and set the frame size
import cv2
import threading

def capture_video():
    capture = cv2.VideoCapture("rtsp://admin:mgasa1234!.@192.168.1.108/cam/realmonitor?channel=1&subtype=0")

    # Adjust capture properties for optimization
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 330)  # Set desired frame width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set desired frame height
    capture.set(cv2.CAP_PROP_FPS, 30)  # Set desired frame rate

    # Create a threading event to signal when the video capture is complete
    capture_complete = threading.Event()

    ret, frame = None, None  # Initialize ret and frame variables

    def read_frame():
        nonlocal ret, frame
        ret, frame = capture.read()
        capture_complete.set()

    # Start the video capture in a separate thread
    capture_thread = threading.Thread(target=read_frame)
    capture_thread.start()

    # Wait for the video capture to complete or timeout after a certain duration
    capture_complete.wait(timeout=0.001)  # Adjust timeout duration as needed

    # Release the capture and join the thread
    capture.release()
    capture_thread.join()

    return ret, frame



# import asyncio
# import cv2

# async def capture_video():
#     capture = cv2.VideoCapture("rtsp://admin:mgasa1234!.@192.168.1.108/cam/realmonitor?channel=1&subtype=0")
#     ret, frame =capture.read()
    
#     capture.release()
#     return ret, frame

# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(capture_video())
