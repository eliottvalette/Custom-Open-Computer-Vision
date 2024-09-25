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

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Predict pose keypoints
    results = model(frame)

    left_arm_detected_objects = 0
    right_arm_detected_objects = 0
    
    for result in results:
        keypoints_xy = result.keypoints.xy[0]
        keypoints_xyn = result.keypoints.xyn[0]
        
        # Annotate the frame with the predicted keypoints
        annotated_frame = result.plot()

        is_right_arm_detected, is_left_arm_detected = detect_arms(keypoints_xy)
    
    # Calculate angle and map it to percentage
    if is_right_arm_detected :
        hand_point = keypoints_xy[10]
        elbow_point = keypoints_xy[8]
        shoulder_point = keypoints_xy[6]

        bar_top_left = (50, 50)
        bar_length = 200

        # Display the progress bar
        display_progress_bar(annotated_frame, bar_top_left, bar_length, shoulder_point, elbow_point, hand_point, color=(0, 255, 0))
        
    if is_left_arm_detected :
        hand_point = keypoints_xy[9]
        elbow_point = keypoints_xy[7]
        shoulder_point = keypoints_xy[5]

        bar_length = 200  
        bar_top_left = (50, 150)

        # Display the progress bar
        display_progress_bar(annotated_frame, bar_top_left, bar_length, shoulder_point, elbow_point, hand_point, color=(255, 0, 0))

    

    # Display the annotated frame
    cv2.imshow("Pose Estimation", annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

