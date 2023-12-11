import yaml

data = {
    'train': '/home/elicer/eyeway_ai/ultralytics/ultralytics/yong/subway_data/train/images',
    'val': '/home/elicer/eyeway_ai/ultralytics/ultralytics/yong/subway_data/valid/images',
    'test': '/home/elicer/eyeway_ai/ultralytics/ultralytics/yong/subway_data/test/images',
    'names': ['escalator', 'gate', 'guide board', 'left arrow', 'platform', 'right arrow', 'stair', 'straight arrow', 'toilet door', 'toilet', 'u arrow', 'under arrow'],
    'nc': 12
}

with open('subway_data/data.yaml', 'w') as f:
    yaml.dump(data, f)