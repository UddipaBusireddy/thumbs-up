import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def detect_thumbs_down():
    def is_thumb_down(landmarks, handedness):
        is_right_hand = handedness.classification[0].label == 'Right'
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        if is_right_hand:
            is_thumb_down = thumb_tip.x > thumb_ip.x > thumb_mcp.x and thumb_tip.y > index_mcp.y > pinky_mcp.y
        else:
            is_thumb_down = thumb_tip.x < thumb_ip.x < thumb_mcp.x and thumb_tip.y > index_mcp.y > pinky_mcp.y

        return is_thumb_down

    thumbs_down_detected = False
    thumbs_down_start_time = 0

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if is_thumb_down(hand_landmarks, handedness):
                        if not thumbs_down_detected:
                            thumbs_down_detected = True
                            thumbs_down_start_time = time.time()

            if thumbs_down_detected:
                if time.time() - thumbs_down_start_time < 3:
                    height, width, _ = frame.shape
                    text_size = cv2.getTextSize("end", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(frame, "end", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    thumbs_down_detected = False  # Reset detection

            cv2.imshow('Thumbs Down Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    return thumbs_down_detected

def detect_thumbs_up():
    def is_thumb_up(landmarks, handedness):
        is_right_hand = handedness.classification[0].label == 'Right'
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        if is_right_hand:
            is_thumb_up = thumb_tip.x > thumb_ip.x > thumb_mcp.x and thumb_tip.y < index_mcp.y < pinky_mcp.y
        else:
            is_thumb_up = thumb_tip.x < thumb_ip.x < thumb_mcp.x and thumb_tip.y < index_mcp.y < pinky_mcp.y

        return is_thumb_up

    thumbs_up_detected = False
    thumbs_up_start_time = 0

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if is_thumb_up(hand_landmarks, handedness):
                        if not thumbs_up_detected:
                            thumbs_up_detected = True
                            thumbs_up_start_time = time.time()

            if thumbs_up_detected:
                if time.time() - thumbs_up_start_time < 3:
                    height, width, _ = frame.shape
                    text_size = cv2.getTextSize("start", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(frame, "start", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
                else:
                    thumbs_up_detected = False  # Reset detection

            cv2.imshow('Thumbs Up Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    return thumbs_up_detected

# Main loop to run both detection functions
if detect_thumbs_up():
    print("Thumbs up detected!")
if detect_thumbs_down():
    print("Thumbs down detected!")


cap.release()
cv2.destroyAllWindows()
