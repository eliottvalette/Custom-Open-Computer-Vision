from ultralytics import YOLO
import os

model_path = 'weights/yolov8n.pt'

model = YOLO(model_path)

data_path = 'Curl-Biceps.v3i.yolov8/data.yaml'

print('Model Loaded, launching training...')

model.train(
    data=data_path,
    epochs=25,
    batch=8,
    imgsz=640,
    device='cpu',
    name='yolov8_custom_model'
)
