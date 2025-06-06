# Gesture-Controlled Racing 🏎️✋

This Python project allows you to control simple web-based racing games using hand gestures detected via your webcam. It translates gestures like a fist, open palm, and pointing into keyboard commands (W, A, S, D).

## Features
*   Real-time hand gesture detection.
*   Controls for:
    *   **Forward:** (Describe your gesture, e.g., "Making a fist")
    *   **Backward:** (e.g., "Open palm facing camera")
    *   **Turn Left:** (e.g., "Hand pointing left")
    *   **Turn Right:** (e.g., "Hand pointing right")
    *   **Neutral:** (e.g., "Relaxed hand")
*   Visual feedback of detected hand landmarks and current action in an OpenCV window.

## Technologies Used
*   Python 
*   OpenCV (`opencv-python`) - For webcam access and image processing.
*   MediaPipe (`mediapipe`) - For robust hand tracking and landmark detection.
*   PyAutoGUI (`pyautogui`) - For simulating keyboard inputs.
*   NumPy (if used directly)

## Click for Video Demo 
[![Watch the Demo](https://img.youtube.com/vi/hE5uxVegz0o/hqdefault.jpg)](https://youtu.be/hE5uxVegz0o)

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your webcam is connected and accessible.
2.  Navigate to the project directory in your terminal (if not already there).
3.  Run the main Python script:
    ```bash
    python app.py
    ```
    (Replace `app.py` with the actual name of your Python file).
4.  Once the script is running and the OpenCV window appears, **quickly make your web browser window (with the game) the active window.** The script sends keyboard inputs to the currently focused application.
5.  Perform the defined hand gestures to control the game.
6.  Press `ESC` in the OpenCV window to quit the application.
