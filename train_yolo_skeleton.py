from ultralytics import YOLO
import os

# Path to the pre-trained model
model_path = 'weights/yolov8n-pose.pt'

# Load YOLO model
model = YOLO(model_path)

# Path to the data.yaml file for your skeleton dataset
data_path = 'Curl-Biceps-Skeleton-Dataset/data.yaml'

print('Model Loaded, launching training...')

# Train the model with specified parameters
model.train(
    data=data_path,  # path to the data.yaml file
    epochs=25,       # number of training epochs
    batch=8,         # batch size
    imgsz=640,       # image size
    device='cpu',    # specify 'cpu' or 'cuda'
    name='yolov8_custom_model'  # name for the training run
)
