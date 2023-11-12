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

load_dotenv()

'''
서버로부터 이미지 하나와 이미지에 detection된 box들의 좌표와 클래스는 받아온다.
일단 정보가 넘어 왔다는 것은 -> 2: 'guide board' 에 대한 값은 확실히 있다는 사실.
처리를 해봐야하는 부분은 3, 5, 7, 10, 11 화살표에 대한 정보와 안내판 정보를 이용해서 길안내 제공.
'''
class BoxCalculation:
    def __init__(self, img, box_info):
        self.img = img
        self.box_info = box_info
        self.label = {0: 'escalator',
                        1: 'gate',
                        2: 'guide board',
                        3: 'left arrow',
                        4: 'platform',
                        5: 'right arrow',
                        6: 'stair',
                        7: 'straight arrow',
                        8: 'toilet door',
                        9: 'toilet',
                        10: 'u arrow',
                        11: 'under arrow'}
                        
        self.ocr_key = os.getenv("OCR_KEY")
        self.ocr_url = os.getenv("OCR_URL")
        
        self.guide_board = list()
        self.arrow = list()
        
        

    def check_closest(self, box1, box2):
        distance = math.sqrt((box2[0] - box1[0])**2 + (box2[1] - box1[1])**2)
        return distance
    
    def guide(self,):
        pass
    
    def ocr(self, img, ):
        
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
        ('file', open(img,'rb'))
        ]
        headers = {
        'X-OCR-SECRET': self.ocr_key
        }

        response = requests.request("POST", self.ocr_url, headers=headers, data = payload, files = files)
