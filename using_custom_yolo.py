import cv2
from ultralytics import YOLO
import numpy as np
import warnings

# Optional: Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load only the custom model
model = YOLO('weights/best3.pt')  # Ensure this is a YOLOv8 model

# Define colors for each class (adjust class names as needed)
class_colors = {
    'elbow': (0, 255, 0),  # Green
    'hand': (0, 0, 255),   # Red
    'shoulder': (255, 0, 0)  # Blue
}

# Object persistence parameters
object_persistence = {}  # Dictionary to hold detected objects and their properties
persistence_threshold = 5  # Number of frames an object is kept after disappearing
iou_threshold = 0.5  # Intersection over Union threshold for object matching

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate intersection
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

# Capture video from webcam
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

    # Perform inference
    results = model(resized_frame, verbose=False)

    # List to hold detected objects in the current frame
    current_frame_objects = []
    
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
            class_name = model.names[cls]
            
            # Store the current detected object
            current_frame_objects.append((x1, y1, x2, y2, conf, class_name))
            
            # Get color for the current class
            color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not in dict
            
            # Draw bounding box with the class-specific color
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label above the bounding box
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
            
    # Update the object persistence dictionary
    updated_persistence = {}

    # Loop through the current frame detections and update the persistence
    for (x1, y1, x2, y2, conf, class_name) in current_frame_objects:
        matched = False
        for key in list(object_persistence.keys()):
            old_box, old_conf, old_class_name, disappearance_counter = object_persistence[key]
            
            # Check if the object matches a previously detected object
            if old_class_name == class_name and calculate_iou((x1, y1, x2, y2), old_box) > iou_threshold:
                # Update the existing object with new box and confidence
                updated_persistence[key] = [(x1, y1, x2, y2), conf, class_name, 0]
                matched = True
                break
                
        if not matched:
            # Add a new object to the persistence dictionary
            updated_persistence[len(updated_persistence)] = [(x1, y1, x2, y2), conf, class_name, 0]

    # Check for objects that disappeared and update their disappearance counter
    for key in list(object_persistence.keys()):
        if key not in updated_persistence:
            old_box, old_conf, old_class_name, disappearance_counter = object_persistence[key]
            if disappearance_counter < persistence_threshold:
                # Keep the object for a few more frames
                updated_persistence[key] = [old_box, old_conf, old_class_name, disappearance_counter + 1]

                # Draw bounding box with the class-specific color
                color = class_colors.get(old_class_name, (0, 255, 0))
                x1, y1, x2, y2 = old_box
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label above the bounding box
                label = f"{old_class_name}: {old_conf:.2f} (Lost)"
                cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)

    # Update object persistence dictionary
    object_persistence = updated_persistence

    # Resize back to original resolution for display
    original_sized_frame = cv2.resize(resized_frame, (1920, 1080))

    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', original_sized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
