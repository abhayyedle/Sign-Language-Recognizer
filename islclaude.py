import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image, ImageTk
import threading
import time
import os

# Check if model file exists
model_path = "model.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    model = None
else:
    try:
        # Load the saved model from file
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the alphabet and numbers
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Set dark theme colors
DARK_BG = "#121212"
PRIMARY_COLOR = "#6200EE"
SECONDARY_COLOR = "#03DAC6"
SURFACE_COLOR = "#1E1E1E"
TEXT_COLOR = "#FFFFFF"
TEXT_SECONDARY = "#B0B0B0"
ERROR_COLOR = "#CF6679"
SUCCESS_COLOR = "#4CAF50"

# Global variables for detection history
detection_history = []
MAX_HISTORY = 10
current_detection = ""
camera_running = False
video_frame = None
cap = None  # Global camera capture object

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) if list(map(abs, temp_landmark_list)) else 1

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def start_camera_thread():
    global camera_running
    camera_running = True
    camera_thread = threading.Thread(target=camera_processing)
    camera_thread.daemon = True
    camera_thread.start()

def stop_camera():
    global camera_running, cap
    camera_running = False
    # Make sure to release the camera
    if cap is not None:
        cap.release()
        cap = None

def update_ui():
    global video_frame, current_detection, detection_history
    
    # Check if the video frame exists and is valid before updating
    if video_frame is not None and isinstance(video_frame, np.ndarray):
        try:
            # Convert color format and update the video label
            img = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        except Exception as e:
            print(f"Error updating video feed: {e}")
    
    # Update current detection display
    if current_detection:
        result_var.set(current_detection)
        
    # Update history in the detection log
    detection_log.config(state=tk.NORMAL)
    detection_log.delete(1.0, tk.END)
    for i, item in enumerate(detection_history):
        detection_log.insert(tk.END, f"{item}\n")
    detection_log.config(state=tk.DISABLED)
    
    # Call this function again after 15ms if camera is running
    if camera_running:
        root.after(15, update_ui)

def camera_processing():
    global camera_running, video_frame, current_detection, detection_history, cap
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            messagebox_error("Camera Error", "Could not open camera. Please check your camera connection.")
            stop_detection()
            return
            
        # Set camera resolution to higher quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
            
            last_detection_time = time.time()
            
            while camera_running and cap.isOpened():
                success, image = cap.read()
                
                if not success:
                    print("Ignoring empty camera frame.")
                    time.sleep(0.1)  # Add short delay
                    continue

                # Flip the image horizontally for a selfie-view display
                image = cv2.flip(image, 1)
                
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                debug_image = copy.deepcopy(image)

                # Add stylish overlay for instructions and branding
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                # Add title and brand name
                cv2.putText(image, "Indian Sign Language Detector", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(image, "Show hand gestures clearly in camera view", (20, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Calculate bounding box for the hand
                        x_min, y_min = image.shape[1], image.shape[0]
                        x_max, y_max = 0, 0
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y
                        
                        # Add some padding
                        padding = 20
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(image.shape[1], x_max + padding)
                        y_max = min(image.shape[0], y_max + padding)
                        
                        # Draw tracking rectangle around hand
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                        
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        # Draw the landmarks with custom style
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                        # Predict only if we have valid landmark data and model is loaded
                        if len(pre_processed_landmark_list) > 0 and model is not None:
                            try:
                                df = pd.DataFrame(pre_processed_landmark_list).transpose()
                                
                                # If df is not empty
                                if not df.empty:
                                    # predict the sign language
                                    predictions = model.predict(df, verbose=0)
                                    # get the predicted class for each sample
                                    predicted_classes = np.argmax(predictions, axis=1)
                                    confidence = np.max(predictions) * 100
                                    
                                    if len(predicted_classes) > 0:
                                        label = alphabet[predicted_classes[0]]
                                        
                                        # Display result in a better UI
                                        result_bg = image.copy()
                                        cv2.rectangle(result_bg, (x_max + 10, y_min), (x_max + 130, y_min + 90), (0, 0, 0), -1)
                                        cv2.addWeighted(result_bg, 0.7, image, 0.3, 0, image)
                                        
                                        # Display the label with confidence
                                        cv2.putText(image, "Sign:", (x_max + 20, y_min + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                                        cv2.putText(image, label, (x_max + 25, y_min + 65), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 2)
                                        cv2.putText(image, f"{confidence:.1f}%", (x_max + 20, y_min + 85), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                                        
                                        # Update current detection and history with timestamps
                                        current_time = time.time()
                                        if current_time - last_detection_time > 1.0:  # Add to history only after a delay
                                            current_detection = label
                                            timestamp = time.strftime("%H:%M:%S")
                                            detection_history.insert(0, f"{timestamp} - Detected: {label} ({confidence:.1f}%)")
                                            if len(detection_history) > MAX_HISTORY:
                                                detection_history.pop()
                                            last_detection_time = current_time
                            except Exception as e:
                                print(f"Error during prediction: {e}")
                
                # Add FPS counter
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps < 1:  # If FPS is not properly read
                    fps = 30  # Use a default value
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(image, fps_text, (image.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        
                # Store the frame for UI update
                video_frame = image.copy()  # Create a copy to avoid reference issues
                
                # Add small delay to reduce CPU usage
                time.sleep(0.01)
                
    except Exception as e:
        print(f"Camera error: {e}")
        messagebox_error("Camera Error", f"An error occurred with the camera: {e}")
    finally:
        # Make sure to release the camera when done
        if cap is not None:
            cap.release()

def messagebox_error(title, message):
    """Display an error message dialog"""
    error_window = tk.Toplevel(root)
    error_window.title(title)
    error_window.configure(bg=DARK_BG)
    error_window.geometry("400x200")
    
    msg = tk.Label(error_window, text=message, bg=DARK_BG, fg=ERROR_COLOR, 
                   font=tkfont.Font(family="Segoe UI", size=12), wraplength=350)
    msg.pack(expand=True, fill="both", padx=20, pady=20)
    
    ok_button = tk.Button(error_window, text="OK", command=error_window.destroy,
                         bg=PRIMARY_COLOR, fg=TEXT_COLOR, padx=20, pady=5)
    ok_button.pack(pady=10)

def start_detection():
    global camera_running
    if not camera_running:
        start_button.config(text="Starting...", state=tk.DISABLED)
        root.update()  # Force update to show the disabled state
        
        if model is None:
            messagebox_error("Model Error", "The model could not be loaded. Please check that 'model.h5' exists in the application directory.")
            start_button.config(text="Start Detection", state=tk.NORMAL)
            return
            
        start_camera_thread()
        update_ui()
        start_button.config(text="Stop Detection", bg=ERROR_COLOR, state=tk.NORMAL)
        start_button.config(command=stop_detection)

def stop_detection():
    global camera_running
    if camera_running:
        stop_camera()
        start_button.config(text="Start Detection", bg=SUCCESS_COLOR)
        start_button.config(command=start_detection)

def show_about():
    about_window = tk.Toplevel(root)
    about_window.title("About Indian Sign Language Detector")
    about_window.geometry("600x500")
    about_window.configure(bg=DARK_BG)
    about_window.resizable(False, False)
    
    # Add title
    title_label = tk.Label(about_window, text="About Indian Sign Language Detector", 
                          font=tkfont.Font(family="Segoe UI", size=16, weight="bold"),
                          bg=DARK_BG, fg=TEXT_COLOR, pady=10)
    title_label.pack(fill="x")
    
    # Add content
    frame = tk.Frame(about_window, bg=DARK_BG, padx=20, pady=10)
    frame.pack(fill="both", expand=True)
    
    about_text = """
Indian Sign Language Detector uses computer vision and machine learning
to detect and interpret sign language gestures in real-time.

Features:
• Real-time detection of hand gestures
• Support for digits (1-9) and all uppercase letters
• High-accuracy neural network model
• Visual feedback with confidence scores
• Detection history logging

This application is built using:
• TensorFlow for machine learning
• MediaPipe for hand tracking
• OpenCV for computer vision
• Tkinter for the user interface

Version: 2.0.0
© 2025 Indian Sign Language Detection Project
    """
    
    info_text = tk.Text(frame, wrap="word", width=60, height=20, 
                        font=tkfont.Font(family="Segoe UI", size=11),
                        bg=SURFACE_COLOR, fg=TEXT_COLOR, padx=15, pady=15)
    info_text.insert("1.0", about_text)
    info_text.configure(state="disabled")
    info_text.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Add close button
    close_button = tk.Button(about_window, text="Close", command=about_window.destroy,
                            font=tkfont.Font(family="Segoe UI", size=11),
                            bg=PRIMARY_COLOR, fg=TEXT_COLOR, padx=20, pady=5)
    close_button.pack(pady=15)

def show_how_to_use():
    info_window = tk.Toplevel(root)
    info_window.title("How to Use")
    info_window.geometry("600x550")
    info_window.configure(bg=DARK_BG)
    info_window.resizable(False, False)
    
    # Add title
    title_label = tk.Label(info_window, text="How to Use Indian Sign Language Detector", 
                          font=tkfont.Font(family="Segoe UI", size=16, weight="bold"),
                          bg=DARK_BG, fg=TEXT_COLOR, pady=10)
    title_label.pack(fill="x")
    
    # Add steps with better formatting
    frame = tk.Frame(info_window, bg=DARK_BG, padx=20, pady=10)
    frame.pack(fill="both", expand=True)
    
    steps = [
        {"title": "1. Start the Detection", 
         "desc": "Click on the 'Start Detection' button to activate your camera and begin real-time detection."},
        {"title": "2. Position Your Hand", 
         "desc": "Hold your hand clearly in front of the camera, ensuring good lighting and a clean background."},
        {"title": "3. Make Clear Gestures", 
         "desc": "Form hand signs corresponding to Indian Sign Language alphabet or numbers."},
        {"title": "4. View Results", 
         "desc": "The detected sign will appear on the screen along with the confidence level."},
        {"title": "5. Check History", 
         "desc": "View your detection history in the log panel on the right side of the interface."},
        {"title": "6. Stop When Finished", 
         "desc": "Click the 'Stop Detection' button when you're done to release the camera."}
    ]
    
    for i, step in enumerate(steps):
        step_frame = tk.Frame(frame, bg=SURFACE_COLOR, padx=15, pady=10)
        step_frame.pack(fill="x", pady=5)
        
        title = tk.Label(step_frame, text=step["title"], 
                        font=tkfont.Font(family="Segoe UI", size=12, weight="bold"),
                        bg=SURFACE_COLOR, fg=SECONDARY_COLOR, anchor="w")
        title.pack(fill="x")
        
        desc = tk.Label(step_frame, text=step["desc"], 
                       font=tkfont.Font(family="Segoe UI", size=11),
                       bg=SURFACE_COLOR, fg=TEXT_COLOR, anchor="w", 
                       wraplength=500, justify="left")
        desc.pack(fill="x", pady=5)
    
    note_frame = tk.Frame(frame, bg=SURFACE_COLOR, padx=15, pady=10)
    note_frame.pack(fill="x", pady=10)
    
    note_title = tk.Label(note_frame, text="Note:", 
                         font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
                         bg=SURFACE_COLOR, fg=ERROR_COLOR, anchor="w")
    note_title.pack(fill="x")
    
    note_text = tk.Label(note_frame, 
                        text="This system currently detects digits 1-9 and uppercase letters A-Z. For best results, ensure good lighting and a plain background.",
                        font=tkfont.Font(family="Segoe UI", size=10),
                        bg=SURFACE_COLOR, fg=TEXT_SECONDARY, 
                        wraplength=500, justify="left")
    note_text.pack(fill="x", pady=5)
    
    # Add close button
    close_button = tk.Button(info_window, text="Close", command=info_window.destroy,
                            font=tkfont.Font(family="Segoe UI", size=11),
                            bg=PRIMARY_COLOR, fg=TEXT_COLOR, padx=20, pady=5)
    close_button.pack(pady=15)

# Create main application window
root = tk.Tk()
root.title("Indian Sign Language Detection System")
root.geometry("1280x720")
root.configure(bg=DARK_BG)
root.minsize(1000, 600)

# Import messagebox after root is created
from tkinter import messagebox

# Configure style
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11))
style.configure("TFrame", background=DARK_BG)

# Create the title bar
title_frame = tk.Frame(root, bg=PRIMARY_COLOR, pady=15)
title_frame.pack(fill="x")

title_label = tk.Label(title_frame, text="Indian Sign Language Detection System", 
                      font=tkfont.Font(family="Segoe UI", size=20, weight="bold"),
                      bg=PRIMARY_COLOR, fg=TEXT_COLOR)
title_label.pack()

subtitle_label = tk.Label(title_frame, text="Real-time vision-based sign language interpreter", 
                         font=tkfont.Font(family="Segoe UI", size=12),
                         bg=PRIMARY_COLOR, fg=TEXT_COLOR)
subtitle_label.pack()

# Create main content frame
content_frame = tk.Frame(root, bg=DARK_BG)
content_frame.pack(fill="both", expand=True, padx=20, pady=15)

# Configure grid layout
content_frame.columnconfigure(0, weight=7)  # Video feed
content_frame.columnconfigure(1, weight=3)  # Controls and history
content_frame.rowconfigure(0, weight=1)

# Create left frame for video
video_frame = tk.Frame(content_frame, bg=SURFACE_COLOR, padx=10, pady=10)
video_frame.grid(row=0, column=0, sticky="nsew", padx=5)

# Configure video frame layout
video_frame.columnconfigure(0, weight=1)
video_frame.rowconfigure(0, weight=9)
video_frame.rowconfigure(1, weight=1)

# Video display with initial blank image
initial_img = np.zeros((480, 640, 3), dtype=np.uint8)
initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB)
initial_img = Image.fromarray(initial_img)
initial_imgtk = ImageTk.PhotoImage(image=initial_img)

video_label = tk.Label(video_frame, bg="black", image=initial_imgtk)
video_label.image = initial_imgtk  # Keep a reference to prevent garbage collection
video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Current detection display
result_frame = tk.Frame(video_frame, bg=SURFACE_COLOR)
result_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

result_label = tk.Label(result_frame, text="Current Detection:", 
                       font=tkfont.Font(family="Segoe UI", size=12),
                       bg=SURFACE_COLOR, fg=TEXT_COLOR)
result_label.pack(side="left", padx=10)

result_var = tk.StringVar(value="None")
detection_result = tk.Label(result_frame, textvariable=result_var,
                           font=tkfont.Font(family="Segoe UI", size=20, weight="bold"),
                           bg=SURFACE_COLOR, fg=SECONDARY_COLOR)
detection_result.pack(side="left", padx=10)

# Create right frame for controls and history
right_frame = tk.Frame(content_frame, bg=DARK_BG)
right_frame.grid(row=0, column=1, sticky="nsew", padx=5)

# Configure right frame layout
right_frame.columnconfigure(0, weight=1)
right_frame.rowconfigure(0, weight=4)  # Controls
right_frame.rowconfigure(1, weight=6)  # History

# Controls section
controls_frame = tk.Frame(right_frame, bg=SURFACE_COLOR, padx=15, pady=15)
controls_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

controls_title = tk.Label(controls_frame, text="Controls", 
                         font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
                         bg=SURFACE_COLOR, fg=TEXT_COLOR)
controls_title.pack(anchor="w", pady=(0, 10))

# Add control buttons
start_button = tk.Button(controls_frame, text="Start Detection", command=start_detection,
                        font=tkfont.Font(family="Segoe UI", size=12),
                        bg=SUCCESS_COLOR, fg=TEXT_COLOR, padx=15, pady=8)
start_button.pack(fill="x", pady=10)

help_button = tk.Button(controls_frame, text="How to Use", command=show_how_to_use,
                       font=tkfont.Font(family="Segoe UI", size=12),
                       bg=SURFACE_COLOR, fg=TEXT_COLOR, padx=15, pady=8)
help_button.pack(fill="x", pady=10)

about_button = tk.Button(controls_frame, text="About", command=show_about,
                        font=tkfont.Font(family="Segoe UI", size=12),
                        bg=SURFACE_COLOR, fg=TEXT_COLOR, padx=15, pady=8)
about_button.pack(fill="x", pady=10)

exit_button = tk.Button(controls_frame, text="Exit Application", command=root.destroy,
                       font=tkfont.Font(family="Segoe UI", size=12),
                       bg=ERROR_COLOR, fg=TEXT_COLOR, padx=15, pady=8)
exit_button.pack(fill="x", pady=10)

# History section
history_frame = tk.Frame(right_frame, bg=SURFACE_COLOR, padx=15, pady=15)
history_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

history_title = tk.Label(history_frame, text="Detection History", 
                        font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
                        bg=SURFACE_COLOR, fg=TEXT_COLOR)
history_title.pack(anchor="w", pady=(0, 10))

# Create text area for detection history
detection_log = tk.Text(history_frame, wrap="word", width=30, height=12,
                       font=tkfont.Font(family="Segoe UI", size=10),
                       bg="#252525", fg=TEXT_COLOR)
detection_log.pack(fill="both", expand=True)
detection_log.config(state=tk.DISABLED)

# Add scrollbar to history
scrollbar = tk.Scrollbar(detection_log)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
detection_log.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=detection_log.yview)

# Add footer
footer_frame = tk.Frame(root, bg=SURFACE_COLOR, pady=8)
footer_frame.pack(fill="x", side="bottom")

footer_label = tk.Label(footer_frame, text="© 2025 Indian Sign Language Detection Project | Created for Computer Vision Competition", 
                      font=tkfont.Font(family="Segoe UI", size=9),
                      bg=SURFACE_COLOR, fg=TEXT_SECONDARY)
footer_label.pack()

# Clean up resources when closing the window
def on_closing():
    stop_camera()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
if __name__ == "__main__":
    root.mainloop()