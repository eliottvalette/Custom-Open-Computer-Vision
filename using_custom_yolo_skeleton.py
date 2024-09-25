import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 Pose model (replace with your custom model path if needed)
model = YOLO('weights/best-pose.pt')  # Replace with 'your_custom_model.pt' if you have a custom model

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera. Change if needed.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Loop through the detections
    for result in results:
        # Extract keypoints using the 'xy' attribute
        keypoints = result.keypoints.xy if result.keypoints.has_visible else []
        
        if len(keypoints) > 0:
            # Get the first set of keypoints (assuming single person detection)
            keypoints = keypoints[0]

            # Draw keypoints and skeleton on the frame
            for keypoint in keypoints:  
                x, y = int(keypoint[0].item()), int(keypoint[1].item())  # Convert tensor values to int
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw keypoints

            # Draw lines between keypoints (example for shoulder, elbow, and hand)
            # Ensure that the required keypoints exist
            if len(keypoints) >= 3:  
                shoulder = (int(keypoints[0][0].item()), int(keypoints[0][1].item()))
                elbow = (int(keypoints[1][0].item()), int(keypoints[1][1].item()))
                hand = (int(keypoints[2][0].item()), int(keypoints[2][1].item()))

                # Draw skeleton
                cv2.line(frame, shoulder, elbow, (255, 0, 0), 2)
                cv2.line(frame, elbow, hand, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
