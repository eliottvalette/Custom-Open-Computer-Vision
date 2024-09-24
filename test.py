
import cv2
import os

# Create an output directory if it doesn't exist
output_dir = "output_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read an image from a file
image = cv2.imread("Curl-Biceps.v2i.yolov8/valid/images/frame_69_jpg.rf.c586c0b9d77caf2da0e8a35d3695a589.jpg")

# Get and print image shape (height, width, channels)
print(f"Training Shape: {image.shape}")

# Save the original image in the output folder
cv2.imwrite(os.path.join(output_dir, "Training_image.jpg"), image)

# Read an image from a file
image = cv2.imread("output_images/opencv_frame_0.png")

# Get and print image shape (height, width, channels)
print(f"Original Shape: {image.shape}")

# Save the original image in the output folder
cv2.imwrite(os.path.join(output_dir, "original_image.jpg"), image)

# Define new size (width, height)
new_size = (640, 640)  # Width, Height

# Resize image
resized_image = cv2.resize(image, new_size)

# Save the resized image in the output folder
cv2.imwrite(os.path.join(output_dir, "resized_image.jpg"), resized_image)

# Get and print resized image shape (height, width, channels)
print(f"Resized Shape: {resized_image.shape}")

# Optional: Save another image read from a different location
image = cv2.imread("output_images/opencv_frame_0.png")
if image is not None:
    print(f"Second Image Shape: {image.shape}")
    cv2.imwrite(os.path.join(output_dir, "second_image.png"), image)
else:
    print("Image not found: output_images/opencv_frame_0.png")

