import cv2
from ultralytics import YOLO
import math
from collections import deque
import numpy as np
import simpleaudio as sa
from pydub import AudioSegment


# Load the pose model
model = YOLO('weights/yolov8n-pose.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Define keypoint indices based on YOLO's keypoint ordering
KEYPOINTS = {
    'Nose': 0,
    'Left Eye': 1,
    'Right Eye': 2,
    'Left Ear': 3,
    'Right Ear': 4,
    'Left Shoulder': 5,
    'Right Shoulder': 6,
    'Left Elbow': 7,
    'Right Elbow': 8,
    'Left Wrist': 9,
    'Right Wrist': 10,
    'Left Hip': 11,
    'Right Hip': 12,
    'Left Knee': 13,
    'Right Knee': 14,
    'Left Ankle': 15,
    'Right Ankle': 16
}

# Define connections between keypoints for skeleton drawing
SKELETON_CONNECTIONS = [
    ('Nose', 'Left Eye'), ('Nose', 'Right Eye'),
    ('Left Eye', 'Left Ear'), ('Right Eye', 'Right Ear'),
    ('Left Shoulder', 'Right Shoulder'),
    ('Left Shoulder', 'Left Elbow'), ('Left Elbow', 'Left Wrist'),
    ('Right Shoulder', 'Right Elbow'), ('Right Elbow', 'Right Wrist'),
    ('Left Shoulder', 'Left Hip'), ('Right Shoulder', 'Right Hip'),
    ('Left Hip', 'Right Hip'),
    ('Left Hip', 'Left Knee'), ('Left Knee', 'Left Ankle'),
    ('Right Hip', 'Right Knee'), ('Right Knee', 'Right Ankle')
]

# Arm keypoints
RIGHT_ARM = ['Right Shoulder', 'Right Elbow', 'Right Wrist']
LEFT_ARM = ['Left Shoulder', 'Left Elbow', 'Left Wrist']

# Initialize counters and state variables
curl_right_counts = 0
curl_left_counts = 0
right_arm_bent = False
left_arm_bent = False

# Elbow stability parameters
BUFFER_SIZE = 30  # Number of frames to track
stability_threshold_x = 50  # Maximum deviation in x (pixels)
stability_threshold_y = 50  # Maximum deviation in y (pixels)

right_elbow_buffer = deque(maxlen=BUFFER_SIZE)
left_elbow_buffer = deque(maxlen=BUFFER_SIZE)

# Variables and constants 
is_right_stable = False
is_left_stable = False

# Sound effect
ding_sound = sa.WaveObject.from_wave_file("sound/ding.wav")
error_sound = sa.WaveObject.from_wave_file("sound/error.wav")

def calculate_angle(p1, p2, p3):
    """Calculate the angle at point p2 formed by p1-p2-p3."""
    try:
        vector1 = (p1[0] - p2[0], p1[1] - p2[1])
        vector2 = (p3[0] - p2[0], p3[1] - p2[1])
        angle = math.degrees(math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0]))
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle
    except:
        return 0

def detect_arms(keypoints):
    """Detect if both arms are present in the frame."""
    
    if len(keypoints) == 0:
        return False, False
    
    right_arm_detected = all(
        keypoints[KEYPOINTS[name]][0] != 0 and keypoints[KEYPOINTS[name]][1] != 0 
        for name in RIGHT_ARM
    )
    left_arm_detected = all(
        keypoints[KEYPOINTS[name]][0] != 0 and keypoints[KEYPOINTS[name]][1] != 0 
        for name in LEFT_ARM
    )
    return right_arm_detected, left_arm_detected

def display_progress_bar(frame, top_left, length, angle, color=(0, 255, 0)):
    """Display a horizontal progress bar indicating the curl progress based on the elbow angle."""
    # Map angle (35-160 degrees) to percentage (100%-0%)
    percentage = (160 - angle) / 125 * 100  # Progress from fully bent (35 degrees) to extended (160 degrees)
    percentage = np.clip(percentage, 0, 100)  # Ensure percentage stays within realistic range
    filled_length = int(length * (percentage / 100))
    
    bottom_right = (top_left[0] + length, top_left[1] + 20)
    filled_bottom_right = (top_left[0] + filled_length, bottom_right[1])

    # Draw background and filled part of the progress bar
    cv2.rectangle(frame, top_left, bottom_right, (50, 50, 50), -1)  # Dark gray background
    cv2.rectangle(frame, top_left, filled_bottom_right, color, -1)  # Filled part

    # Display percentage text
    cv2.putText(frame, f"{percentage:.1f}%", (top_left[0], top_left[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def is_elbow_stable(elbow_buffer):
    """Check if the elbow movement is within the stability thresholds."""
    if len(elbow_buffer) < elbow_buffer.maxlen:
        return False  # Not enough data to determine stability
    xs = [pt[0] for pt in elbow_buffer]
    ys = [pt[1] for pt in elbow_buffer]

    return not (max(xs) - min(xs) > stability_threshold_x) or (max(ys) - min(ys) > stability_threshold_y)

def draw_ui_background(frame):
    """Draw semi-transparent background panels for UI elements."""
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor

    # Define areas for right arm, left arm, counts, and instructions
    cv2.rectangle(overlay, (0, 0), (300, 250), (30, 30, 30), -1)      # Top-left panel
    cv2.rectangle(overlay, (0, 250), (300, 400), (30, 30, 30), -1)    # Middle-left panel
    cv2.rectangle(overlay, (0, 400), (300, 450), (30, 30, 30), -1)    # Counts panel
    cv2.rectangle(overlay, (0, 450), (300, 480), (30, 30, 30), -1)    # Instructions panel

    # Apply the overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_pose_annotations(frame, keypoints):
    """Draw keypoints and skeleton on the frame."""
    # Draw keypoints
    for name, idx in KEYPOINTS.items():
        x, y = keypoints[idx]
        if x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Cyan dots

    # Draw skeleton connections
    for connection in SKELETON_CONNECTIONS:
        pt1_name, pt2_name = connection
        pt1 = keypoints[KEYPOINTS[pt1_name]]
        pt2 = keypoints[KEYPOINTS[pt2_name]]
        if all(coord != 0 for coord in pt1) and all(coord != 0 for coord in pt2):
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)  # Green lines

def process_arm(annotated_frame, keypoints, arm_keypoints, buffer, buffer_name, curl_counts, arm_bent, bar_top_left, bar_color):
    """Process an arm (either right or left) for stability and curl detection."""
    shoulder = keypoints[KEYPOINTS[arm_keypoints['Shoulder']]]
    elbow = keypoints[KEYPOINTS[arm_keypoints['Elbow']]]
    wrist = keypoints[KEYPOINTS[arm_keypoints['Wrist']]]

    # Add elbow position to buffer
    buffer.append(elbow)
    stable = is_elbow_stable(buffer)

    # Calculate angle
    angle = calculate_angle(shoulder, elbow, wrist)
    
    # Clip the angle to reflect human limitations (max bend at 60 degrees, fully extended at 160 degrees)
    angle = np.clip(angle, 35, 160)

    # Display progress bar
    bar_length = 200
    display_progress_bar(annotated_frame, bar_top_left, bar_length, angle, color=bar_color)

    # Check for arm curl logic based on realistic bending thresholds
    if angle <= 70:  # Arm is realistically considered bent at <= 70 degrees
        if not arm_bent:
            arm_bent = True
    elif angle >= 150:  # Arm is considered extended at >= 150 degrees
        if arm_bent:
            arm_bent = False
            curl_counts += 1  # Count one curl
            print(f"{buffer_name} Arm Curl Count: {curl_counts}")

    # Display stability indicator
    stability_text = "Stable" if stable else "Unstable"
    stability_color = (0, 255, 0) if stable else (0, 0, 255)
    cv2.putText(annotated_frame, f"{buffer_name} Elbow: {stability_text}", 
                (bar_top_left[0], bar_top_left[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1, cv2.LINE_AA)

    return curl_counts, arm_bent, stable

def add_glowing_red_border(frame, border_thickness=10):
    """Add a fading red gradient border around the frame."""
    global should_play_error_sound

    if should_play_error_sound:
        # error_sound.play()
        should_play_error_sound = False 

    height, width, _ = frame.shape

    # Create a gradient mask for the border
    gradient_mask = np.zeros_like(frame, dtype=np.uint8)

    cv2.rectangle(gradient_mask, (0, 0), (width, height), (0, 0, 255), thickness=4)

    # Thickness of the gradient (the "glow" effect)
    for i in range(border_thickness):
        color_intensity = 255 - (i * (255 // border_thickness))  # Reduce red intensity towards center
        cv2.rectangle(gradient_mask, (i, i), (width - i, height - i), (0, 0, color_intensity), thickness=1)

    # Blend the gradient mask with the original frame
    alpha = 0.4
    blended_frame = cv2.addWeighted(frame, 1 - alpha, gradient_mask, alpha, 0) 

    return blended_frame

def main():
    global curl_right_counts, curl_left_counts, right_arm_bent, left_arm_bent
    global is_left_stable, is_right_stable
    global should_play_error_sound

    # Initialize the variable
    should_play_error_sound = False

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)  # Mirror the frame

        if not ret:
            print("Error: Could not read frame.")
            break

        # Predict pose keypoints
        results = model(frame, verbose=False)

        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy
            if keypoints is not None and keypoints.size(1) > 0:  # Ensure keypoints are detected
                keypoints = keypoints[0].tolist()  # List of [x, y] for each keypoint
                draw_pose_annotations(annotated_frame, keypoints)
            else:
                print("No keypoints detected.")
                keypoints = []
        else:
            print("No person detected or no keypoints available.")
            keypoints = []

            
        # Draw UI background
        draw_ui_background(annotated_frame)

        # Process each detected person (assuming single person for UI consistency)

        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy[0].tolist()  # List of [x, y] for each keypoint

            is_right_arm_detected, is_left_arm_detected = detect_arms(keypoints)

            # Process Right Arm
            if is_right_arm_detected:
                curl_right_counts, right_arm_bent, is_right_stable = process_arm(
                    annotated_frame, keypoints, 
                    arm_keypoints={'Shoulder': 'Right Shoulder', 'Elbow': 'Right Elbow', 'Wrist': 'Right Wrist'},
                    buffer=right_elbow_buffer, buffer_name="Right", 
                    curl_counts=curl_right_counts, arm_bent=right_arm_bent, 
                    bar_top_left=(20, 20), bar_color=(0, 255, 0)
                )
            else:
                # Clear buffer if arm is not detected
                right_elbow_buffer.clear()
                cv2.putText(annotated_frame, "Right Arm Not Detected", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Process Left Arm
            if is_left_arm_detected:
                curl_left_counts, left_arm_bent, is_left_stable = process_arm(
                    annotated_frame, keypoints, 
                    arm_keypoints={'Shoulder': 'Left Shoulder', 'Elbow': 'Left Elbow', 'Wrist': 'Left Wrist'},
                    buffer=left_elbow_buffer, buffer_name="Left", 
                    curl_counts=curl_left_counts, arm_bent=left_arm_bent, 
                    bar_top_left=(20, 100), bar_color=(255, 0, 0)
                )
            else:
                # Clear buffer if arm is not detected
                left_elbow_buffer.clear()
                cv2.putText(annotated_frame, "Left Arm Not Detected", 
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            if not (is_right_stable or is_left_stable):
                annotated_frame = add_glowing_red_border(annotated_frame, border_thickness=30)  # Adjust thickness as needed
            else :
                should_play_error_sound = True

        else:
            # No person detected
            cv2.putText(annotated_frame, "No Person Detected", 
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "No Person Detected", 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Display the counts on the UI panel
        cv2.putText(annotated_frame, f"Right Arm Curls: {curl_right_counts}", 
                    (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Left Arm Curls: {curl_left_counts}", 
                    (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Display instructions
        cv2.putText(annotated_frame, "Press 'q' to exit.", (20, 430), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("Pose Estimation - Arm Curls", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
