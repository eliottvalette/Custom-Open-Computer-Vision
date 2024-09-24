import cv2
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

print("Press 's' to save a frame, 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret: 
        break
    
    cv2.imshow('Webcam', frame)
    
    k = cv2.waitKey(10)  # Try increasing delay if needed
    
    if k % 256 == 27:
        # 'ESC' key to break the loop
        print("Escape hit, closing...")
        break
    elif k % 256 == ord('s'):
        # 's' key to save the frame
        img_name = os.path.join(output_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
