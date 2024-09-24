import cv2
from ultralytics import YOLO
import numpy as np
import warnings
import math

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # Calculate the angle between the two vectors
    angle = math.degrees(math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0]))
    angle = abs(angle)
    
    # Adjust the angle to be within 0-180 degrees
    if angle > 180:
        angle = 360 - angle
    return angle

# Optional: Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the custom YOLO model
model = YOLO('weights/best_new.pt')  # Ensure this is a YOLOv8 model

# Define colors for each class (adjust class names as needed)
class_colors = {
    'elbow': (0, 255, 0),  # Green
    'hand': (0, 0, 255),   # Red
    'shoulder': (255, 0, 0)  # Blue
}

# Capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to 640x640 for model input
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform inference on the resized frame
    results = model(resized_frame, verbose=False)

    # Dictionary to hold highest confidence detection for each class
    highest_conf_boxes = {}

    # Parse results
    for result in results:
        boxes = result.boxes  # Boxes object containing all detections in the frame
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract confidence and class
            conf = box.conf.item()
            cls = int(box.cls.item())
            
            # Check if this class is already in the dictionary
            if cls not in highest_conf_boxes or conf > highest_conf_boxes[cls]['conf']:
                middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                highest_conf_boxes[cls] = {'conf': conf, 'box': (x1, y1, x2, y2), 'middle_point': middle_point}
        
    # Draw only the highest confidence bounding box for each class
    for cls, box_info in highest_conf_boxes.items():
        x1, y1, x2, y2 = box_info['box']
        conf = box_info['conf']
        middle_point = box_info['middle_point']
        
        label = f"{model.names[cls]}: {conf:.2f}"
        
        # Get color for the current class
        color = class_colors.get(model.names[cls], (0, 255, 0))  # Default to green if class not in dict

        # Draw bounding box with the class-specific color on resized frame
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label above the bounding box
        cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)

        # Draw a circle at the middle point of the bounding box
        cv2.circle(resized_frame, middle_point, 5, color, -1)

    # Calculate angle and map it to percentage
    if 1 in highest_conf_boxes and 0 in highest_conf_boxes and 2 in highest_conf_boxes:
        hand_point = highest_conf_boxes[1]['middle_point']
        elbow_point = highest_conf_boxes[0]['middle_point']
        shoulder_point = highest_conf_boxes[2]['middle_point']

        # Calculate angle between the points
        angle = calculate_angle(shoulder_point, elbow_point, hand_point)

        # Map angle (0-180 degrees) to percentage (100%-0%)
        percentage = (180 - angle) / 180 * 100

        # Display the progress bar
        bar_length = 200  # Length of the progress bar in pixels
        filled_length = int(bar_length * (percentage / 100))
        
        # Define the position and size of the progress bar
        bar_top_left = (50, 50)
        bar_bottom_right = (bar_top_left[0] + bar_length, bar_top_left[1] + 20)

        # Draw the background of the progress bar (empty part)
        cv2.rectangle(resized_frame, bar_top_left, bar_bottom_right, (200, 200, 200), -1)

        # Draw the filled part of the progress bar
        filled_bottom_right = (bar_top_left[0] + filled_length, bar_bottom_right[1])
        cv2.rectangle(resized_frame, bar_top_left, filled_bottom_right, (0, 255, 0), -1)

        # Display the percentage text
        cv2.putText(resized_frame, f"{percentage:.1f}%", (bar_top_left[0], bar_top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw line between hand and elbow
        cv2.line(resized_frame, hand_point, elbow_point, (255, 255, 255), 2)  # White line

        # Draw line between elbow and shoulder
        cv2.line(resized_frame, elbow_point, shoulder_point, (255, 255, 255), 2)  # White line

    # Resize the frame back to the original size for display
    original_shape = cv2.resize(resized_frame, (1920, 1080))
    
    # Display the resized frame with detections
    cv2.imshow('YOLO Object Detection', original_shape)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
