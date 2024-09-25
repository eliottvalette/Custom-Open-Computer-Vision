import cv2
from ultralytics import YOLO
import math

# Load the pose model
model = YOLO('weights/yolov8n-pose.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Keypoint names
keypoint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
right_arm_names = ['Right Shoulder', 'Right Elbow', 'Right Wrist']
left_arm_names = ['Left Shoulder', 'Left Elbow', 'Left Wrist']

# Initialize counters and state variables
curl_right_counts = 0
curl_left_counts = 0
right_arm_bent = False
left_arm_bent = False

def calculate_angle(point1, point2, point3):
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    angle = math.degrees(math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0]))
    angle = abs(angle)
    
    if angle > 180:
        angle = 360 - angle
    return angle

def detect_arms(keypoints_xy):
    right_arm_detected = sum(1 for name, pt in zip(keypoint_names, keypoints_xy) if name in right_arm_names and pt[0] != 0 and pt[1] != 0) == 3
    left_arm_detected = sum(1 for name, pt in zip(keypoint_names, keypoints_xy) if name in left_arm_names and pt[0] != 0 and pt[1] != 0) == 3
    return right_arm_detected, left_arm_detected

def display_progress_bar(frame, top_left, length, shoulder_point, elbow_point, hand_point, color=(0, 255, 0)):
    angle = calculate_angle(shoulder_point, elbow_point, hand_point)

    # Map angle (0-180 degrees) to percentage (100%-0%)
    percentage = (180 - angle) / 180 * 100

    filled_length = int(length * (percentage / 100))
    bottom_right = (top_left[0] + length, top_left[1] + 20)
    filled_bottom_right = (top_left[0] + filled_length, bottom_right[1])

    # Draw background and filled part of the progress bar
    cv2.rectangle(frame, top_left, bottom_right, (200, 200, 200), -1)
    cv2.rectangle(frame, top_left, filled_bottom_right, color, -1)

    # Display percentage text
    cv2.putText(frame, f"{percentage:.1f}%", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return angle

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Predict pose keypoints
    results = model(frame)
    
    for result in results:
        keypoints_xy = result.keypoints.xy[0]
        keypoints_xyn = result.keypoints.xyn[0]
        
        # Annotate the frame with the predicted keypoints
        annotated_frame = result.plot()

        is_right_arm_detected, is_left_arm_detected = detect_arms(keypoints_xy)

    # Check and count for right arm curls
    if is_right_arm_detected:
        hand_point = keypoints_xy[10]
        elbow_point = keypoints_xy[8]
        shoulder_point = keypoints_xy[6]

        bar_top_left = (50, 50)
        bar_length = 200

        # Display the progress bar and get the angle
        angle = display_progress_bar(annotated_frame, bar_top_left, bar_length, shoulder_point, elbow_point, hand_point, color=(0, 255, 0))

        # Check for arm curl logic
        if angle < 45:  # Arm is fully bent
            if not right_arm_bent:
                right_arm_bent = True
        elif angle > 160:  # Arm is fully extended
            if right_arm_bent:
                right_arm_bent = False
                curl_right_counts += 1  # Count one curl
                print(f"Right Arm Curl Count: {curl_right_counts}")

    # Check and count for left arm curls
    if is_left_arm_detected:
        hand_point = keypoints_xy[9]
        elbow_point = keypoints_xy[7]
        shoulder_point = keypoints_xy[5]

        bar_length = 200  
        bar_top_left = (50, 150)

        # Display the progress bar and get the angle
        angle = display_progress_bar(annotated_frame, bar_top_left, bar_length, shoulder_point, elbow_point, hand_point, color=(255, 0, 0))

        # Check for arm curl logic
        if angle < 45:  # Arm is fully bent
            if not left_arm_bent:
                left_arm_bent = True
        elif angle > 160:  # Arm is fully extended
            if left_arm_bent:
                left_arm_bent = False
                curl_left_counts += 1  # Count one curl
                print(f"Left Arm Curl Count: {curl_left_counts}")

    # Display the count on the frame
    cv2.putText(annotated_frame, f"Right Arm Curls: {curl_right_counts}", (10, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Left Arm Curls: {curl_left_counts}", (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the annotated frame
    cv2.imshow("Pose Estimation", annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
