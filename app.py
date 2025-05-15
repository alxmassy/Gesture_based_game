import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Configuration ---
# Key mappings (you can change these to arrow keys if your game uses them)
KEY_FORWARD = 'w'
KEY_BACKWARD = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'

# Gesture sensitivity (higher means less sensitive, needs more movement)
# These are example values, you'll need to TUNE them.
STEERING_THRESHOLD_X = 0.08 # Normalized X-coordinate difference for left/right
FINGER_CURL_THRESHOLD_Y_FACTOR = 0.8 # For checking if fingers are curled (tip_y > mcp_y * factor)

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,  # Control with one hand
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # 0 for default webcam
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# --- State Variables for Key Presses ---
# To ensure we only send keyDown/keyUp once per state change
keys_pressed = {
    KEY_FORWARD: False,
    KEY_BACKWARD: False,
    KEY_LEFT: False,
    KEY_RIGHT: False
}

def press_key(key_to_press):
    if not keys_pressed[key_to_press]:
        pyautogui.keyDown(key_to_press)
        keys_pressed[key_to_press] = True
        print(f"Pressed: {key_to_press}")

def release_key(key_to_release):
    if keys_pressed[key_to_release]:
        pyautogui.keyUp(key_to_release)
        keys_pressed[key_to_release] = False
        print(f"Released: {key_to_release}")

def release_all_movement_keys():
    for key in [KEY_FORWARD, KEY_BACKWARD, KEY_LEFT, KEY_RIGHT]:
        release_key(key)

# --- Main Loop ---
print("Starting gesture controller. Make sure your web game window is active.")
print("Press ESC in the OpenCV window to quit.")
time.sleep(2) # Give user time to switch to game window

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # And convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_action = "NEUTRAL" # Default state

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- Gesture Recognition Logic ---
                # Get specific landmarks (normalized 0.0-1.0 coordinates)
                # (Refer to MediaPipe hand landmark model for IDs: https://google.github.io/mediapipe/solutions/hands.html)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP] # Base of index
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                # --- Simple Gesture Definitions ---

                # FORWARD: Fist (all fingers curled, index and pinky tip Y > their MCP Y)
                # (Y decreases upwards, so tip_y > mcp_y means finger is curled down)
                is_index_curled = index_finger_tip.y > index_finger_mcp.y * FINGER_CURL_THRESHOLD_Y_FACTOR
                is_middle_curled = middle_finger_tip.y > middle_finger_mcp.y * FINGER_CURL_THRESHOLD_Y_FACTOR # Added middle
                is_pinky_curled = pinky_tip.y > pinky_mcp.y * FINGER_CURL_THRESHOLD_Y_FACTOR

                # BACKWARD: Open Palm (all fingers extended, index and pinky tip Y < their MCP Y)
                is_index_extended = index_finger_tip.y < index_finger_mcp.y
                is_middle_extended = middle_finger_tip.y < middle_finger_mcp.y # Added middle
                is_pinky_extended = pinky_tip.y < pinky_mcp.y

                # GESTURE 1: FORWARD = Fist (Index, Middle, Pinky curled)
                if is_index_curled and is_middle_curled and is_pinky_curled:
                    current_action = "FORWARD"

                # GESTURE 2: BACKWARD = Open Palm (Index, Middle, Pinky extended, thumb may vary)
                elif is_index_extended and is_middle_extended and is_pinky_extended and \
                     thumb_tip.y < wrist.y : # Thumb also somewhat up
                    current_action = "BACKWARD"

                # GESTURE 3 & 4: STEERING LEFT/RIGHT based on hand X position relative to a center
                # This is simpler than tilt for basic control
                # Use wrist X position relative to center of detected hand bounding box or image center.
                # For simplicity, let's use index finger tip X relative to wrist X
                # (after image flip, left on screen is smaller X, right is larger X)
                elif index_finger_tip.x < wrist.x - STEERING_THRESHOLD_X:
                    current_action = "LEFT"
                elif index_finger_tip.x > wrist.x + STEERING_THRESHOLD_X:
                    current_action = "RIGHT"


                # --- Alternative Steering: Hand Tilt ---
                # More intuitive steering for some
                # Calculate horizontal distance between wrist and middle finger MCP
                # If middle_finger_mcp.x < wrist.x - TILT_THRESHOLD: current_action = "LEFT"
                # elif middle_finger_mcp.x > wrist.x + TILT_THRESHOLD: current_action = "RIGHT"
                # Ensure this doesn't conflict with FORWARD/BACKWARD gestures.
                # You might need to make forward/backward more specific if you use tilt.


        # --- Apply Actions to Game ---
        if current_action == "FORWARD":
            press_key(KEY_FORWARD)
            release_key(KEY_BACKWARD) # Ensure not braking
            # Release steering if not actively steering
            if not (keys_pressed[KEY_LEFT] or keys_pressed[KEY_RIGHT]): # if no active steer gesture
                 release_key(KEY_LEFT)
                 release_key(KEY_RIGHT)
        elif current_action == "BACKWARD":
            press_key(KEY_BACKWARD)
            release_key(KEY_FORWARD) # Ensure not accelerating
            # Release steering
            if not (keys_pressed[KEY_LEFT] or keys_pressed[KEY_RIGHT]):
                 release_key(KEY_LEFT)
                 release_key(KEY_RIGHT)
        elif current_action == "LEFT":
            press_key(KEY_LEFT)
            release_key(KEY_RIGHT) # Ensure not turning right
            # Keep forward/backward if they were active, otherwise release
            if not keys_pressed[KEY_FORWARD] and not keys_pressed[KEY_BACKWARD]:
                release_key(KEY_FORWARD)
                release_key(KEY_BACKWARD)
        elif current_action == "RIGHT":
            press_key(KEY_RIGHT)
            release_key(KEY_LEFT) # Ensure not turning left
            if not keys_pressed[KEY_FORWARD] and not keys_pressed[KEY_BACKWARD]:
                release_key(KEY_FORWARD)
                release_key(KEY_BACKWARD)
        elif current_action == "NEUTRAL":
            release_all_movement_keys()

        # Display the current action on the OpenCV window
        cv2.putText(image, f"Action: {current_action}", (20, image_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Gesture Controller - Game Window MUST BE ACTIVE', image)

        if cv2.waitKey(5) & 0xFF == 27: # ESC key
            break
finally:
    # --- Cleanup ---
    print("Exiting and releasing keys...")
    release_all_movement_keys() # Ensure all keys are released
    if cap.isOpened():
        cap.release()
    if 'hands' in locals() and hands: # Check if hands was initialized
        hands.close()
    cv2.destroyAllWindows()