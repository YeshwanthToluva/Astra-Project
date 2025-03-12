import cv2
import mediapipe as mp
import subprocess
import asyncio
import time
import websockets

# Virtual camera device
video_device = "/dev/video10"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)
print(f"cap.isOpened(): {cap.isOpened()}")
# Explicitly set resolution
width = 640
height = 480
fps = 30

# ffmpeg command
ffmpeg_command = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', f'{fps}',
    '-i', '-',
    '-f', 'v4l2',
    '-pix_fmt', 'yuyv422',  # or nv12
    video_device
]

# Start ffmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    count = 0
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[finger_tips[0]].x:
        count += 1
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

async def finger_count_handler(websocket, path): # Added websocket parameters.
    global cap
    print("WebSocket connection established.")
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"cap.read() failed: {ret}")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            finger_count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    finger_count = count_fingers(hand_landmarks)

            try:
                await websocket.send(str(finger_count)) # Send finger count
            except Exception as e:
                print(f"WebSocket send error: {e}")
                break

            try:
                ffmpeg_process.stdin.write(image.tobytes())
                ffmpeg_process.stdin.flush()
            except Exception as e:
                print(f"ffmpeg write error: {e}, ffmpeg.stderr: {ffmpeg_process.stderr.read()}")
                break

            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = max(0, 1.0 / fps - elapsed_time)
            await asyncio.sleep(sleep_time)

    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        print("WebSocket connection closed.")

start_server = websockets.serve(finger_count_handler, "localhost", 8765) # Start websocket server.

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
