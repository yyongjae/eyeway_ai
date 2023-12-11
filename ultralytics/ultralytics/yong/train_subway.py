import sys
import os
sys.path.append('/Users/yongcho/dev/yonggit/eyeway_ai/ultralytics/ultralytics/')
from models import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

model = YOLO("yolov8n.pt")

add_wandb_callback(model, enable_model_checkpointing=True)

model.train(
    project="eyeway",
    data='/Users/yongcho/dev/yonggit/eyeway_ai/ultralytics/ultralytics/yong/subway_data/data.yaml',
    epochs=30,
    device='mps'
)

metrics = model.val()