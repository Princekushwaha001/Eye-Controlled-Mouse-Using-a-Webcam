import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and Mediapipe face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Initialize variables for blink detection
blink_threshold = 0.004
blink_start_time = None
blink_duration_threshold = 0.1  # duration in seconds

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            left_eye_landmarks = [landmarks[145], landmarks[159]]
            for landmark in left_eye_landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Detect blink by comparing the y-coordinates of landmarks 145 and 159
            if abs(left_eye_landmarks[0].y - left_eye_landmarks[1].y) < blink_threshold:
                if blink_start_time is None:
                    blink_start_time = time.time()
                elif time.time() - blink_start_time > blink_duration_threshold:
                    pyautogui.click()
                    blink_start_time = None  # reset blink detection
            else:
                blink_start_time = None

        cv2.imshow('Eye Controlled Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
