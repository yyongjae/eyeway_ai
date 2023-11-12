import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from dotenv import load_dotenv
import json
import time
import uuid
import requests


s = time.time()
ocr_key = "ekZGcmxtdU92dElVenFhaUZGQnB1bm9jV3VORmxzUW8="
ocr_url = "https://4l285jou16.apigw.ntruss.com/custom/v1/26032/df38e814ad16a031b7878fee9e515bc44f2d93a2fdf6ecef8c6673b308e83900/general"
        
request_json = {
    'images': [
        {
            'format': 'jpg',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}

payload = {'message': json.dumps(request_json).encode('UTF-8')}
files = [
('file', open('a.jpg','rb'))
]
headers = {
'X-OCR-SECRET': ocr_key
}

response = requests.request("POST", ocr_url, headers=headers, data = payload, files = files)

result = response.text.encode('utf8')
result_json = json.loads(result.decode('utf8').replace("'", '"'))
result_text = result_json['images'][0]['fields'][0]['inferText']

e = time.time()
print(e-s)
print(result_text)
