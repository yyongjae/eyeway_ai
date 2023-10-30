import sys
import os
sys.path.append('/home/elicer/eyeway_ai/ultralytics/ultralytics/')
from models import YOLO

model = YOLO("yolov8n.pt")
print(model)

# model.train(
#     data='subway_data/data.yaml',
#     epochs=
# )