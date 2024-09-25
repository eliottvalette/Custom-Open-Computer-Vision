import cv2
from ultralytics import YOLO

# Load the YOLO pose estimation model
model = YOLO('weights/yolov8n-pose.pt')

# Open a connection to the webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture frames continuously
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Make a prediction on the frame
    results = model(frame)
    
    # Draw keypoints and bounding boxes on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("Pose Estimation", annotated_frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
