from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./eyeway/train/weights/best.pt')

# Define path to video file
source = 'adae.mov'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects