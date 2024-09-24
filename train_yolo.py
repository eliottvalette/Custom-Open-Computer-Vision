from ultralytics import YOLO
import os

# Path to the pre-trained weights
model_path = 'weights/yolov8n.pt'

# Load the model
model = YOLO(model_path)

# Path to the dataset configuration file
data_path = 'Curl-Biceps.v2i.yolov8/data.yaml'

print('Model Loaded, launching training...')
# Train the model on the custom dataset
model.train(
    data=data_path,  # Path to data.yaml file
    epochs=25,  # Number of epochs to train
    batch=8,  # Batch size
    imgsz=640,  # Image size (pixels)
    device='cpu',  # Use GPU (0 for first GPU, 'cpu' for CPU)
    name='yolov8_custom_model'  # Name of the training run
)
