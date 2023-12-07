import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import json
import time
import uuid
import requests
import io
import base64
from PIL import Image, ImageFile
from similarity import find_same_string, jaccard_similarity, Levenshtein_similarity
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()

class Guide:
    def __init__(self, img, box_info, situation='board'):
        self.img = img
        self.box_info = box_info
        self.situation = situation # 1.길찾기  2.화장실찾기  3.나가기
        
        self.ocr_key = os.getenv("OCR_KEY")
        self.ocr_url = os.getenv("OCR_URL")
        
        self.guide_board = list()
        self.arrow = list()
        self.text = list()
        self.label_onboarding = {0:'guide board',
                                 1:'왼쪽',
                                 2:'오른쪽',
                                 3:'직진',
                                 4:'아래 방향'}
        
        self.label_toilet = {0:'왼쪽',
                             1:'오른쪽',
                             2:'직진',
                             3:'화장실',
                             4:'화장실 입구',
                             5:'아래 방향'            
                             }
        
    def get_text_center(self,box,feat): 
        
        if feat == 'ocr':
            x_coords = [vertex['x'] for vertex in box]
            y_coords = [vertex['y'] for vertex in box]

            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
        ### --- xyxy xywh 확인 필요 --- ###
        elif feat == 'yolo':
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # x, y , w, h = box
            # center_x = x + w//2
            # center_y = y + h//2
        
        return center_x, center_y    
    
    def is_all_english(self, text):
        # 영어만 있으면 true    
        return all(char.isalpha() and ord(char) < 128 for char in text)

    def is_all_korean(self,text):
        # 한글만 있으면 True
        for char in text:
            if not ('\uAC00' <= char <= '\uD7A3'):
                return False
        return True

    def to_byte_img(self, img):
        # ocr 보내려면 바이트 형태의 이미지 값을 넘겨야 해서 변환.
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode('.jpg', opencv_image)[1].tobytes()

        return image_bytes
    
    def process_ocr_data(self, data):
        
        processed_data = []
        uid = data['images'][0]['uid']
        # json 형태가 좀 복잡함. 아래의 iteration은 한 박스에 대한 정보가 들어있음.
        for box in data['images'][0]['fields']:
            
            infer_text = box['inferText']

            # 한글 외에 다른 문자 있으면 거르기
            if not self.is_all_korean(infer_text):
                continue
            
            # 중심 좌표 계산
            bounding_poly = box['boundingPoly']['vertices']
            center_x, center_y = self.get_text_center(bounding_poly, 'ocr')

            processed_field = {'uid': uid, 'center': (center_x, center_y), 'text': infer_text}
            processed_data.append(processed_field)

        '''
        [{'uid': 'd83c37b54b~',
        'center': (532.5, 28.25),
        'text': '고속터미널'}, ...]
        '''
        
        return processed_data


    def ocr(self, img):
        
        start = time.time()
        
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
        
        byte_img = self.to_byte_img(img)
        files = [('file',byte_img)]
        headers = {
        'X-OCR-SECRET': self.ocr_key
        }

        response = requests.request("POST", self.ocr_url, headers=headers, data = payload, files = files)

        result = response.text.encode('utf8')
        result_json = json.loads(result.decode('utf8').replace("'", '"'))
        
        end = time.time()
        delay = end - start
        print(f"[OCR] 완료! 걸린 응답 시간: {delay}")
        print(result_json)
        processed_data = self.process_ocr_data(result_json)
        
        return processed_data
    
    def crop_boxes(self, img, gb_box, ar_box):
        # 안드로이드에서 base64로 인코딩된 이미지 정보를 가져오게됨.
        decoded_data = base64.b64decode(img)
        image = Image.open(io.BytesIO(decoded_data))
        g_x1, g_y1, g_x2, g_y2 = gb_box['box']
        cropped_image = image.crop((g_x1, g_y1, g_x2, g_y2))
        
        # crop한 이미지에 맞게 화살표의 박스 좌표도 변환을 해준다.
        for box in ar_box:
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = x1 - g_x1, y1 - g_y1, x2 - g_x1, y2 - g_y1
            box['box'] = [x1, y1, x2, y2]
            
        print(f'[CROP] 변환 완료')
        
        return cropped_image, ar_box
    
    def box_size(self, box):
        ### --- box 좌표의 형태가 xyxy인지 xywh인지 확인 후 수정 필요 --- ###
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        return width * height
    
    def distance(self, box1, box2):
        # 박스 중심 좌표 두개 사이의 거리.
        distance = ((box2[0] - box1[0])**2 + (box2[1] - box1[1])**2) ** 0.5
        return distance
    
    def filtering_object(self,box_info):
        
        guide_board = []
        arrows = []
        toilet = []
        outtlier = []
        
        ### --- 박스 정보에 따라 인덱싱 수정해야함 --- ###
        for box in box_info:
            
            if box['label'] == 0: # guide board
                guide_board.append(box)
            elif box['label'] in [1,2,3,4]: # L, R, S, under    
                arrows.append(box)
            elif box['label'] == 8:
                toilet.append(box)
            else:
                outtlier.append(box)
                print(f"[Label 분류]{box['label']} 발견")
        
        try:
            # 안내판이 여러개 검출되었다면 가장 큰 녀석만 골라서 사용할 계획. -> 잘린 안내판이 있을 수 있기에 수정 필요.
            board = guide_board[0]
            
            if len(guide_board) > 1:
                max_size = 0
                
                for gb in guide_board:
                    size = self.box_size(gb['box'])
                    if max_size < size:
                        max_size = size
                        board = gb
            
            # 화살표가 모두 안내판 안에 있는 화살표인지 확인.
            ### --- xyxy xywh 확인 --- ###
            filtered_arrow = []
            for arrow in arrows:
                arrow_cen_x, arrow_cen_y = self.get_text_center(arrow['box'], 'yolo')
                x1, y1, x2, y2 = board['box']
                if (x1 <= arrow_cen_x <= x2) and (y1 <= arrow_cen_y <= y2):
                    filtered_arrow.append(arrow)
                
            print(f"[Filtering] 총 {len(filtered_arrow)}개의 화살표 인식")
            return board, filtered_arrow, toilet
        
        except:
            print('[Filtering] 인덱싱 에러 발생.')
            return False, False, False
        
        
    
    def mapping_arrow(self, data, arrow_data,feat):
        result = {'arrows':[]}
        if feat == 'guide':
            
            for text in data:
                text_point = text['center']
                direction = ''
                min_distance = 9999
                for arrow in arrow_data:
                    arr_point = self.get_text_center(arrow['box'], 'yolo')
                    dist = self.distance(text_point, arr_point)
                    
                    if dist < min_distance:
                        min_distance = dist
                        direction = arrow['label']
                result['arrows'].append({"type": self.label_onboarding[direction],
                                        "text": text['text']})
        
        elif feat == 'toilet':
            
            direction = ''
            min_distance = 9999
            for arrow in arrow_data:
                arr_point = self.get_text_center(arrow['box'], 'yolo')
                toilet_point = self.get_text_center(data['box'], 'yolo')
                dist = self.distance(toilet_point, arr_point)
                
                if dist < min_distance:
                    min_distance = dist
                    direction = arrow['label']
            
            result = self.label_toilet[direction]
            
        return result
            
    def board(self,):
        
        box_info = self.box_info
        org_img = self.img
        # 화살표 있는지 체크하고 안내판, 화살표 리스트를 나눠서 받기.
        ## 화살표가 안내판 박스 안에 있는지 체크
        ### 해당 조건에 맞는 안내판 및 화살표 리스트 받아오기
        guide_board, arrow, _= self.filtering_object(box_info)
        
        if not guide_board:
            # crop한 안내판 이미지에 대해 OCR 진행
            crop_g_board, trans_arrow_box = self.crop_boxes(org_img, guide_board, arrow)        
            ocr_data = self.ocr(crop_g_board)
            for data in ocr_data:
                text = data['text']
                ## 텍스트 박스에 대해서 역 이름, 나가는 곳 등과 같은 정보만 필터링 및 단어 유사도 보완
                jc1, jc2 = jaccard_similarity(text)
                lv = Levenshtein_similarity(text)
                mod_text = find_same_string(jc1, jc2, lv)
                # --- 추가할 점 --- #
                ## 유사도 엄청 낮으면 아예 빼버려야함. ## 
                # 수정한 텍스트 반영
                data['text'] = mod_text
                if text != mod_text:
                    print(f'[텍스트 수정] 기존: {text} -> 수정: {mod_text}')
            print('글자 수정 완료')
            
            # 화살표 박스와 텍스트 박스의 유클리드 거리를 비교하여 맵핑
            result = self.mapping_arrow(ocr_data, trans_arrow_box,'guide')
        
        else:
            result=False
            
        return result
    
    def toilet(self,):
        
        box_info = self.box_info
        guide_board, arrow, toilet= self.filtering_object(box_info)
        result = self.mapping_arrow(toilet, arrow,'guide')

        return result
    
    def extract_number(self, input_string):
        current_num = ''
        for char in input_string:
            if char.isdigit():
                current_num += char
            elif char.isalpha():
                break

        if current_num:
            return int(current_num)
        else:
            return None
        
    def exit(self, exit_num):
        box_info = self.box_info
        org_img = self.img
            
        guide_board, arrow, _ = self.filtering_object(box_info)
        
        # crop한 안내판 이미지에 대해 OCR 진행
        crop_g_board, trans_arrow_box = self.crop_boxes(org_img, guide_board, arrow)        
        ocr_data = self.ocr(crop_g_board)
        check = False
        direction = ''
        num_lst = []
        for data in ocr_data:
            if data['text'] in ['나가는', '나가는곳', '나가', '곳', '나가는 곳', '출구', 'Exit']:
                check = True
            if any(char.isdigit() for char in data['text']):
                num_lst.append(data)
        
        num_check = False
        if check:
            for num in num_lst:
                if '~' in num['text']:
                    matches = re.findall(r'\d+|~', num['text'])
                    numbers = [int(match) if match.isdigit() else None for match in matches]
                    numbers = [num for num in numbers if num is not None]

                    if numbers[0] <= exit_num and numbers[1] >= exit_num:
                        num_check = True
                        break
                else:
                    pure_num = str(self.extract_number(num['text']))
                    if str(exit_num) in pure_num:
                        num_check = True
                        break
    
        if num_check:
            if len(trans_arrow_box) > 1:
                max_size = 0
            big_arrow = trans_arrow_box[0]
            for ab in trans_arrow_box:
                size = self.box_size(ab['box'])
                if max_size < size:
                    max_size = size
                    big_arrow = ab
                
            direction = big_arrow['label']
        
        return direction
            
            
    def start(self,):
        s = time.time()
        # 상황에 따라 세가지로 나눠서 처리
        ## 1. 길찾기
        ## 2. 화장실찾기
        ## 3. 출구찾기
        if self.situation == 'board':
            print('[Start] 승강장 찾기 시작.')
            res = self.board()
        elif self.situation == 'toilet':
            print('[Start] 화장실 찾기 시작.')
            self.toilet()
        elif self.situation == 'exit':
            print('[Start] 출구 찾기 시작.')
            self.exit()
        e = time.time()
        print(f'총 {e-s}초 소요')
        return res


if __name__ == "__main__":
    
    # image_path = './ultralytics/ultralytics/yong/subway_data/v2/images/images_864.jpg'
    # with open(image_path, 'rb') as image_file:
    #     image_data = image_file.read()
        
    box = [{'label':0, 'box':[257, 198, 576, 232]},
         {'label':1, 'box':[268, 197, 293, 219]},
         {'label':2, 'box':[546, 208, 568, 229]}]
    
    
    encoded_image = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAIQAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAABZAAAABRnWFlaAAABeAAAABRiWFlaAAABjAAAABRyVFJDAAABoAAAAChnVFJDAAABoAAAAChiVFJDAAABoAAAACh3dHB0AAAByAAAABRjcHJ0AAAB3AAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAFgAAAAcAHMAUgBHAEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFhZWiAAAAAAAABvogAAOPUAAAOQWFlaIAAAAAAAAGKZAAC3hQAAGNpYWVogAAAAAAAAJKAAAA+EAAC2z3BhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABYWVogAAAAAAAA9tYAAQAAAADTLW1sdWMAAAAAAAAAAQAAAAxlblVTAAAAIAAAABwARwBvAG8AZwBsAGUAIABJAG4AYwAuACAAMgAwADEANv/bAEMACgcHCAcGCggICAsKCgsOGBAODQ0OHRUWERgjHyUkIh8iISYrNy8mKTQpISIwQTE0OTs+Pj4lLkRJQzxINz0+O//bAEMBCgsLDg0OHBAQHDsoIig7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O//AABEIAoACgAMBIgACEQEDEQH/xAAcAAACAwEBAQEAAAAAAAAAAAAAAQIDBAUGBwj/xABQEAABAwIEAwUDCQUDCgQGAwABAAIDBBEFEiExBkFREyJhcYEUMpEVI0JScqGxwdEHM2KCkhZDRCQlNFNUY4OT4fA1wtLxF0VVc6KyJoTi/8QAGgEBAQEBAQEBAAAAAAAAAAAAAAECAwQFBv/EACgRAQEAAgEFAQADAAMAAwEAAAABAhESAxMhMVFBBCJhFCMyUnGBkf/aAAwDAQACEQMRAD8A5WR7DmJV7ZmZLF4Llz6meUEgmwWQS5Te5K+JcLX3OTuOfGxmYv8ARRE+ZhGawPJcuOpu75zborfaWfQAvyWeFjW5VxqnQy2A0XSiqC+O91wyc5u46noulQghliVz6mM9tYseIPJqByUYmNknY07FSxFoEoShcGSxuXon/l5srrJ6OPCGmBrozYpOhqYBa1wulRStfTtseSvdYjUXWOEXvX9effUzM3jKyVEz5hqwr0r4ozu0LO+CIfQCnCuuPXwn480InOFshUaiHsYS5+i9CWxg2sAs9VRU9QzKXi/mrjjq+WsuvL6eNLwoMdd91uxPDHUpLm7Lnxe+vTPTz9XPkskeWuCkHkhIuYHgO0vzWjsGyAFpsrdMYqRcouWqwxlhsQoubcaDVR0uPjZB91a14tuqmxO3IsqxIDJYHZTKM45eW+GUg76KNZIHBUxyNBFyp1AZl0cFzk1W8/S2lHcutNlmpAcgWoBYy9uRWQpJKKikVJIhVEUk7IIQRIUSFOyjZURISsppWVECkpkKJCqElZSskgSVk7FFiqiNklPKTySykckEUJkFIhAkHVOySoVkW0TKSBITSQCSdkWVEUKwROOzSrW0hIuTZTZplIQtPsov72iZgjaO88Jyi8ayJLSGReYQRCRoE3DjWZKy1Cna7wTNIOTldxNVjSWo0T9wQVS6GRu7VZqs6VFRUyCNwoqoihNCqFZaIBsqFpgClajQFIBIBSCy0adkBNRRZNATUAgBMBNo1QSA0R5qdkwFk0zua2oZvdYZoZIjtopzNmpn3Zt0SZiLCLStWpuO11Wd02lraobMGnY3WztaKTW4CYOHtF3SBXlCSowTHaOMuJ5ldWB5jju4gErmHEqaMZYI8x6nZJj3zOzvcbrjljt2xXV7szwb3VU37oEFVzjvDvXU5b+z6LtJ4ebL/wBOlhmLvpow2U6dV6CnxWGdos8Lw0F5BZxWpkDmaskIPgs7k9u2XQ35xe1NQwi+YKiapY0auC8i+orIxYTFZpJqqQd6Ureo49rKPVuJqAezcuS6gr46wSMeS2+oXKgramkddsp8ito4lqGtsWAnqpqnGx0sTGWiJl3svLR/vDZXVuJVNc7vus3oFRFo8LpJqMZCr90FVwVckPuu06FWVn7sLDddJjLHHlZXS+Ui4d5g9FdTVrHSAFm64+YqTZC3UGxS4NTOvS1UTpKf5v7lwnNfG8h2hWqixl0No5+8zrzC3zQQVsfaRuBv0WJ48Vr/AFx85TEji4AlSmppKd1iCQq2Ou4JZFtrtUZ7gWtZaL3FqXlvtpEhFlKyRCio2SspJFVESEtVJIoIpFSSVEUlKyRCCJSspEJWVRGyVlLVRJsqqL5RE25WZ9e6xs1KqkvosbiRzW5jKzbpe2veX66LW2pJA1XEeTm3WumLjuVvLpyJjlt0nT2HJWR5JW35rnTPszQqVHUEmy53GyNSytjmFpUbHopPdfVREjhsFDQynoUxG52wKfbOOltVqp85AJClulkZ20sjvoqYoZCt2cMGp1Vb6rTRZ51rjGb2JwOpVjYGs5XKi6oJ5qIlIV3SSL8xAtYBQdru5UGUnmoOmA5qcabWvIA3Wd3eKi6W6iJNV0k0i5rARZaI4mNFyFTG7mUpam2gWfNVofI1osqHyt5GyxS1VtSVUJzJstTBm5NpnI+kVJsrjre6wAkuW1hs0JcZFl2tzxu0c1QdTMfqw2UXEKpzyNik2lkKSncwqoiykZnE6klSa+5tl1W9/WOPxWN1rhGiXs+mbZWxtsLJaSaWBNIKVlhTCYSUgEBZMICkAoospMGqWVWRhSh2TsnZSAWVc2apbLu0rDMxjthZWDGsNdvHIPgpjE8Ld9OQebV2mNxL1ccnOdDqhsIuumKvC3f4gDzaph2GvOlVF6hauVZlxZYIoxut4EIZooCGld7tTCf5lP2Rp92SM+T1yyltejHq4yMs4AsVMm9MrnUDztr5FP2SUMyZHWW445Xd25bagRk6qfyj4q52FG5JDtfBVOwtwOht5hOON9vROrlPSt1fcc1WazwKtOHPHNpSNA/oFqTFOedUGqB5KBnB5K80Lx9AqJpHj6B+C1OLN51V2w6K2F+Z4UDCRu0/BTiblcNErjlv9SrP3V1zrrpVX7lcxdMPTz5e0rq2IttY7qlNpIN1bFxvlc6IuNwNFOCpno35mHTmORSZId03uDlj37d7jPcdiCsgrmZXjK/mCstRSNikDmnRYGEtdmG4VoqJJHBpJssa0xXZo/dWtY6MdxbF5b7dAhCLqKRSJTSRCQUWRZUR0RZMhRQIhKykUlREpJlCIiouCmghWDn1TQASua9xB0XZqo7sNlx5G6kL0YMZKS7VbKexAssLwQVfTS8rrplPDGN8tkoBaraOFo7yzyuJatFG7urll6dZ7a7BSaBtZQzAapMcXO02XJtrayJupAJUn1AtZugWdzrBV5vFZ1tdrXSk7lUuluovfyuq7rcxRZmJ5pOfyuoF1gqy+61IWrDIQN1U5/ile6rcVrTO17dRcptHeUInXCkXBtyVFSmqmRDLexWV89xe6x1TnTVGVmpV7aSYR3cFuYSOXO1TLIXbKVO43VMrSw2KvpLO3W8vETH21Rgk6BagbBQaQ0aKMkmi462670jLOGjdViTMs0j7utdaqamc8XOgWrJGd2pxMc86BbY4QwXO6nDG2NtgpuNxZcrdukmkA4E2KlkyqgktcrY5+TlPMSpgJgI0OoTsqyLKQCAFIBRQBqpAIAUgFArK1g0ULa2V7Ros1SsmAnZSDVkfM09bJpL7L5wuepTD3D6RSuhNG0+1k+smKiUbOVaV1OM+LyrQ2tqW7SEeRKuZi1YzaZ/9RWFCzenjfxeeX11G8QV7f79/xVzeJq4DV9/NoXFQs9rBru5O+3imf6UcTvNitHFDD79LEfiF5tCnYxXu5PUN4io3e9SAfZerm45hrt4ZGHweCvIoWexPq96vYtxPC3jV8rfQFT9pwt+oqfixeL2Ug5w2J+Kl6F+td/69pkoJm2FXCfBwIUfkyik919O7yfZeO7WT65+KkKiUbPKdrOeqd3G+49ccCgd7rGn7Mig7h9vJso8rFeXFbO36asZitUzaRw8nFTh1F54PRHAsumd/q1VuwVw2lt5tsuQzHq5m00n9aubxNWgayOPmAVnj1HTuYuiMMe0e8w+qr+TqjtLhrSPtKhnFVSNHNa7zYFY3in61PCf5SFnjnPwueNdaljdG2zhZarhcRvE1OffpWejiFc3iHDz70Dx9mRcL08vjXOOrcdULnDG8Nd/r2+oKmMUw521RI3zjWeFi8o22SIVDa2hdtXNH2mEKxstO/wB2upz5vsmqu4lZFkw0n3ZYXeUgU+wmOzQfJwKgqIUdVa6CYbxO9Aoljxuxw8wqK7pKWiNOoQQRZSyX2I+KfZu6IIWSUzG7oVEsd0KqKpW3YVxqmNzXErulptqFiq4ARddenkmU24jwSPFVtdleCtUgAJ0WV4sV6p5jz3xXQY7NGrqY2Nlgp5bd0rZE4B64ZT8d8a2nVTYQ0KntNEnTCy56roufJcqDngBUGXndVGbNstTFNrjJcoz6KltyVZld0V0mye+wVXaaqUjHHkothceSrNoMqiXZld7L9Y2TbDGzdwTcPKqN5CpqpiG2BW8RwnYqipoe0BLCrjraZctMVFrIXE7LstJc0X1XMpaZ8TjmC6DX2Fjus5+1w9MWIwgd4LNSvylbqqKSRp6LFHE5jtQuk84sesm0P01Vcr+6kHWColeXGwUkXKkzvyLrQmzAAuU35tlyV0aR2ZgKzm1g2Mcpl2iqaRdWbri6qn6qDWlxstQYLaqLmhuwTYnC22iuLCFjzO+iVdC+Q6ONwp5SxcApAJhhsnZEACkAgBSAWQNF3K62ihGLlW2UtUgFIBACkBdQeSl4MroZYopJoI3zAlgfIBe3/us+I8K4jhlNJUTtYGRmzrOBXUq+OYKiWOduFZnx3yNmkDmsv0sFnqOKI6/h+sp6px9rmcC0BvdtroF97jPr5W68ymASbDUqKshcGSNcRexXNuNHyVXmISeyS5HbOyGxVD6adhIdG4EbghfWsKxSCbBKXsGsndazmtqG3b6FOOSKpnngbURU7i+4feNx8i1y3xZ2+QZHX2KC0jcEL6ZTQUdNjUnyi+nDp22idLC1pJ6gC7SuZx1RYbBTt9nfS+1xutIInBhI+wpxNvCoTV9HFHPUsjkdla42J6LLTOhe8reDMPihzwR4jMwsBEsTWPYfvurMP/Z7DV4fFO6eoa+QXNohlZ563V41Hz9C9tH+zmWYG1dG15eWNaY3WNvHkvPY5gM+CPg7Vweydmdjm80spuOUhCFAIQhA0JIUDR6JJoCyEk0AhCEBZGo5oQpqLumHvGznfFPtpfrlRQpxi8qsFRKPpfcptrZ27PVCFOGPxeeX1sZi1YzaZ/o8q9nEGIs2qJf67rmIWb0sb+L3MnZbxTiI96ZzvtAFWDiuq+kyJ3nEFwkKdnBe7k9E3io/TpYHfyEK1vE9Mfeoo/5XuC8wms9jFrvV6tnEeHu96nlb9mUfmrm49hrvpVDPUOXjbIss3+Piver27cYw13+LmH2orqfyhh8g/wBOh8nxkfkvC28UwXDZzvip/wAfX6vf/wAe2MeGz6iWid6lqicIopB3RAb/AFZwvGZ3/XKYlkH0h8Fe1nPVO5jfcevPDkRuWRyfyvBQcBezbth5suvJtqp27PPxVrMTrGHuzPHk8qXp5rOpjHpnYW8aZ3erCs0uF1BPdlZ63C5TOIMTZtVTf1q5nFGJN3lLvtNBU7eca7mNajhlXa3cPk5WQ4dM3V7Cso4rrPpsif8AaiH5KxvFZ+nR05/lI/NS45/FmeLZ2JYf3LvgnmtvGR6LO3iinPv0YH2ZSFY3iOgd71PKPKUH8li45fGueJumYN22VElawaMAWoY5hbxZwnH8rSpe3YPLvNb7cP6J/wDcN/HEmrHF26qNUTuV38mCS/31P6tLUvkvCpPdlpj5T2W+WLF5OA2qcH3ubLoQVgcBqt39nqSTWN2/1ZQUf2dDNWOl+4/gplcb6axuX6oFQPBJ07RqQFe7BZLe+/1YVnlwefYTM9bhZ8fWt0Mq2uNiBZWF0UgsGi6piwmpYdXRnyctIo5WDRt/Ip69L79sE8ZbtssL5hGdV2pKabL+6d6C65NZh81yRG7+ldsNX249Tc8xjlqjIbDZdWgmHZAE6rimGRju80jzC1QSFoW+phNeGOnl58u6x9zutDXLlU9Tci5XRjcHBeXKaeqXbQHXTNiqwbJ3WNNIuaQdE2uLeaZN1S9+U3Cutm26KbqVpADhcLkMqLFbaepabC9lizQ1gKVk22cLgp2WUSjCsshgs1O2qyoA1UwEgFMBTY+W9g/wSMTxyWtC+/p8pj7N/wBUoyuG7SttkKaNsNrciEee63WSLAdwEGRsjmODmuLSNiDspTzy1MplnkdJId3ONyfVaOzZ9UJdkz6qeRkVsE8lPI2WJ2V7TcG11aYI+n3pezs6lNVduyzjjFxAYHsopYib5X0zbX9LLp0H7RpKWlZBJhjCG3uYpcoPoQfxXkvZh9YpezfxK7qeHtcM/aFS0k1S+eiqCJn5m5JM2XwsV5zinGafHMWFVTRvjiETWBrgARa/Rcz2d31gkad/gpbaam1SFZ2EnRLsnj6JWVQQpZH/AFT8ErHoUCQnqkgEIQoGhJCBoSQgaEkIGhCEAhCEAhCEAhCEDSQhAJpJoBCEKKLppIQNCSEErhGiSEDsEWR6J20QLKEZR0CRNkZkDyBIttzPxRmRdAd4fTd8Uw54+mVG6LpqLyqfayj6amysqGe7I4eRIVN0lOGPxeeX1tbjFezapl/5hVzeIsSaLe0yEeJB/FcxCz28fi9zJ2G8TVw95zX/AGmBWDiiY+9DD/y7LhoUvRwXu5PQN4nb9KlYfJxCubxLSn3oHt+zJf8AFeZSWexiveyesbj1A/3mSjzylMYlhEnvffACvJWCdlOz8q93/HrRLgzzo6EebC1XMOHH3KiD+srxlyNifimHvH0ipehfrU60+PbiGFw7srD5SBBpM3uvPpYrxOd/1vuU21Mzdn28ljsZfWp149iaKQj3nf0ql9A8/wB4PUFeYbiFWzaV/wDWVc3GsQaNKiT+pO1nF72Nds4bNfuvYfWyYo6ph0DT5PC5DcfrxvIT9oAqxvEdUN8h/wCGFm9PP4s6uP13YXVcRAMZI8Ddb4pc9szS0+K8w3iWT6UMZ+IVjeJm/SgZ6OK5XpZfG+5i9e21tx8UwvKDiSA7wvHk9WN4gpTuZW+Viud6WXxeeL1QCkAvMtx6m5Tyt8wrW45Adq63mCp28l5R5W4PNNc/OepQJHDZxX3Nvluihc/tn/WKkKiQfSTatyFiFTIOd0xVv6BEbELJ7YebQn7YObUGq6LrP7WzoUxVR87j0QX3RdUioj+sn20f1wqLboUBIw/SCeZp5j4oJISBHVF0DQhCAsOiWUfVCaamjaBjZzaEuxYR7qsRZNQ2p7BnQpezs6lXoU1DbOaYcnFL2Y8nLTZCaNsvs7vrBLsH9QtdkWTRtj7GToPijsZPqrZZMBTS7Yuyf9UpZHfVPwW+yLJo2wWPQpLoWSyjoE0bYELd2bT9EJGFh+iFNG2NC1+zx/VSNNGeo9U0u2VC0+yt6lL2UfWKaptnQtHsh5OCj7K/qFNU2pQrTTSDkPil7PL9VNCtCs7GT6pUTG8fRPwTQihPKRyKSKd0ZikhAkIQiBCEIBCEIBCEIBCEIBCaFQkJr13DmEUUtHSyzQxTzVkzo2duXBkYba50800PIIX0DEOHqGN5pqjDYqaR4cWPglebWtqQ4WtqvBOADiBsClhtFC24ThkmL4jFQxSRxvlNgZDYLRj+COwLEDSOqI5iPpMP/dk0OUhOy9HRcDYvX4bFXwiAQytzNL5g029U0PNoXbxLhLF8KpTVVVOGwg2zteHD7lxbKaCQptjc82a0k9AE5IZIjaRjmn+IWRVaE0kAiwQhAIuRzPxQhNRWfmhJNaYCEIQCEkIGhJNAJpIQNCSaBoukmgdz1TD3fWPxUUXVRZ2r/rFPtpPrFVoTYuFRJ1+5WxTPe8DRZQrYH5JATyXTpzeUlS+nYZRyOjzWFlz3ymN5aW6heho5mTUtmnkuRV0UhqHOa3Qr1/yOhMZLi49LqcrqutgvDb8ZoTUtqGxWcW5SLrVJwTWNvlqInehC9PwVwrV1PDrJ2Cm77ibSN1+K7UnCeJxtJbBA8fwSOC+X/wBtuo9dmD5lNwtXxc4yPArHJhFZGdYwfIr6FXYbNS92qp5IgfpCRxC41TSNDjZ8g6d66x1Or1Olf7RrHp45eq8e6iqG7xFRNPMN4nfBemfSi/vu9bKo038X3LnP5f8AjXY/15zspBvG74IykbtI9F6MQOB3B9Fb2D7e60+iv/LnxOxfry6F2aqoZTyZZaa3QkDVSYKSohzCDI/+JoAK6d+fsY7dcRC7zaKlcNY2ehVjcJpXD938CrOtidvJ51C9GcBpyLgP9CoHh6Lk5481ru4s8K4CF23cPdJnf0qDuH5OUw9WrXcx+nGuOhdQ4FUDaRhVbsFq27ZD6q88fqarAhbDhVYP7sHyKrNBVN3hctbiaZ1IDVWGlqBvC8eij2Ug3Y4eiqvp3DvC2AS8P0j8QoZJKidpfnbKW6aLynGvDMXD+IxinkLoKhuZjXHvN81uw/8AaLU0OHwUU+B0VUynZka+S4dbzsVx+KeJpeJ66KpfSMpBEzI2ON5cPwCrP6sZwdUyuayOoikeHNbM1oJ7K/X4KmPhKpmmijZNTEytzNBk72XqQoU3E2IRVMMs8pnZEb5NGZj4kDX1WjDuKpqSVr56aJ5ZHkbJG3K8DlrzUVy5sDqIzPaNr2U7sj3tOl1Z/ZXEHGUMoy/sjZ5aQbFdCLihzZXwupIRRzSZ5G5buPje+6vHFzGVNQBSF8EzyTZwYSPEZTdTUV5iqwt9JN2U8Lo3kXyndVOocri1zXAjcHkvTVWK0NZj7q6eqlMcco7JnZkjJ18F1peIuH5X1dRld2tSzK67OXQJqJt4A0gtcE2UDS9Hfcve1OJYWykqCKmglpjGBTUzYh2jXeOl1i4gkweGidJRCGWausQGgfMNG/kTf7k1F28aaY/WCRp3dQtSSzo2y9g8clExPH0StiSaNsZY4bgo16LYhNG2JO612HQJFjT9EJo2yr0OE8TxUNFHS1FJLIIi4xyQTBjgHWuDcEcguP2bPqhLsmdEHoazi6CSlyU0NY6URGKN1TK1wjb4WAJXl7q7sW+KXYDqUHT4WxDD8Lx+mrcREvZwvDg6PWxHUcwquIsRpsVxypraRkjYpnlw7S1/gFg7Do5LsXdQiog2IK+qV+FVGJUVDSUZgq8MiiBdBHUtjc5/iei+W9k/wS7J42HwKs8F8voPEsdZhnAdPQYgWNnE1mMa4H5sbDxXz1NzZDbNmNtrm6iWuHIqW7I+kfswfhgpKt0jGGtjOYuc29mcrXUMfxOfG4ZaeWKFkZF2EQg5fHPuvnjJpo2OYx72tf7wa4gO81Z7fV9j2PtEnZ/VzaLGfK64vT0Muhjvuy34qcMriL3stWHUQrqjI6ZkEbRmfK/Zo/VYrq2nqpqSTtIXBrrWN2hwI8QdCtPM7TuHszmzwyNdQ6B03bx3+BIXNxKgdh1WYDI2RpaHMe0ghzTsdEhi9UXkydlK1wAMb4m5LDbQWsqKmskq5RJLluGhrQ1tg0DYBXwExhe8MaC5zjYAc113YFDBAZKmomGU5XvhpjJHGehdcLjRTvglbLGcr2G7T0K7v9oqd9K+Bxr4oZSXS0sL2iN5O/ePet4WKTRt5hCEIgQhCAQhNAk0IQJCaEAhCEDQEk0DQkmqhoQhFMJhRTTaNtFWvpnixu3ovQ0WI0tQ5okaGkryQOq6+EMZM+x3C93S/lZSas245dGX0+48KYrh1JgMML6lsbhyIK7jMZw957tdD8SF5LAOF+2wWmmbXiMvbfKW3t966f8AZaYN7tXC7zBC8/8AXK726XcdHEhhuJQlklZACRoc4XzTGYzh1S5l2yRg6FpuvYScKYhmu2SmcOmYj8lzcT4erWwnNCw2+q4K9Tp4dTpWW7XDLLHJ4uSpaRcbKr2sDcK+qw+phkc10RCxubbuvBBX5/jp9HbQKlu60wztdZcwFre6/QHYq2N+U73WbFdOSGnqWBs0TXjxCk+jp5YeyyNy20CqgnDrArbG0ON7BSdTKX2lxaMB4bwHEGGCqD4qpuoyv0ePBdvC+C8CmqZo/apWhlsoL7XXBEDg9sjHlj2m4cNwt8OJ1MMmZ8cMpO5dcE/Ar6fT63Tznn28uWGcvh6xnA2DgDLJP6SBSPA2Gn3ZqgeoK4kfEbAAH4YL9WVDgrm8SU4/wdQ37NR/0XX/AK78Y/u6LuBaM+7VSjzaCqH8Axn3K74xqDOJqYDavb5SNKtHFNMP76uHmxp/NOOFTlkyycAz/QqoXebbLO/gOu+i6B3k6y7DeKaYj/TpW+D6b9Fa3iaA7YhAftwvH5J28F55PNScDYm3aBjvJ6yycG4qzehcfJwK9q3iKE/46gPm8t/FXR47G/3JqN/2agKdrE5188fwziTN6Cb0as78CrG+9RTj+RfU2Yk920LT9mZpVxq3CNrzC91+TSCQp2Yvcr4+/Cnt96nePNiodhbOcHxYvshroj79NN6x3UTPQu96EfzQq9q/lOc+PjBwindvC34KJwWkd/dtHkV9mcMJf70EHrFZVmgwOTenpvhZXhlP05Y/Hxo8P0p+i70cq3cOQ/Re8eq+yuwPAZP8ND6OUDwvgj9qcej01n9N4/Hxd/DjQNJX/AKh3D8g2l+IX2mTg3CH7CVvk9Z38CYa73Z52/Ap/wBh/R8Zdgc42e0ql2EVDb+6fVfXK79n8QhLqeueHcg5q8TjGCV+EShktQxxdsLXUuec9wmON/XlHYbUj+7v6qs0VSP7py9OyJ7mDPlJ8EGDwC5/8j/Gu28oaaYbxu+CiYnjdh+C9W6n8FA03gtd6J268qQeiWq9O6kB+iqn0UfONvwV72KduvOpLtyUMQBJYAAqG01NIbNaT5Arc6mNThXLQu+3AM7cwilF/BJ3Drv4x6K8onGuChdl3D8g2cfgq3YFONnD4K8omq5SS6TsEqQNMpUDg9WPoA+qu4arAha3YbVN/uiqjR1A/unfBBSkrDBKN2OHooljhuD8FRFGnRBFklAi0dAllb0CZKRQRLGnkomNvippIm0DEOqRhvzViEGJCEKKEIQgaEIQCEIQCSaSBoQhAKSWqAgaaSaAQhCATSQgYV9PO+CQPY6xCoUm7hLbPS4zy/QfDD3HhuhLtzGLrqFy5OAdzAqNvSILoF+i+Jepl9evjEy8jms9RK4tIzH4pvk8VmlfoVe/1JNbThPjmVcQfe4B9FxqmiiedYmH+Vd2Y3usErQVxvUy+uskc+Hh+mxCmlaW5Hj3XBeWqaebDql1NUNIIPdd1X0nBou5KbLFj+DRYjTkOaBIPdcu2GW55S3VeGhmLDuurTVGYDVcaWnlpZTDKLOb96uppiw2KZ4Lt6WJ+YaqbrFYqWbMBqtRKzjEqQd8VLMeqqOmqebRdtsLQfFTCpa7VWNKsRNF0kLW0GZRdlI1A+CZ2USryv1NRAFlvcbfyTErmnukt8iQqybPIUc132Wu5l9XjGttbUt92pqG+Uzh+atbjGItFm11SP8AiE/isIUgr3c/qcMXSZjuKjbEJj9oNP5LQ3iHExvNG/7UTSuQPJWDZanVy+s8MXWHEdfzjpXecP8A1UxxHMPfoaV3lmauNdRJK13sk4R328TsvZ2HW+xMVMcUUoGtNVtP8MwK80DdxN0iVe/kduO7V8SwysytNZ5OcLfcV5TEpKivq+1e45RtmN1qcqnBZy61s0uOElZspA3SN/BXOVZGi4Ois5ug+KVnfVUykSequxCx+qVW82GrT8FaSetkt+aDmuz1tQKeFrnXOwF7r6Bw5hGG0WV1RSyB4aDmey/e+C8e2NrX52jK4bOboV0IMXxKAAMxCpA6dovT0+phjPLlljb6fSPaqEj96webUi/D3bugPm0LwTOI8Wb/AI6R32gD+Sl/aXFDvOw+cLSu3cwc+GT3Jp8Pf9CmPoFW7DMNf/h6c+gXixxLW/SZSv8AOED8FNvEk596jpHeQcPwK1zwOOT1jsCwx/8AhIvRUv4Ywt3+Ft5OXnRxGB71Cz+SVwUxxJEN6SYfZqP1Cu8D+zsP4Sws/wBy4eTlQ/gzDXbGRvqsY4mp7e5WsPhI0/kpM4ngP9/Wt82NP5p/VPKUnA9GR3ZnjzCxzfs9p5P8U71augOI6f8A+oSeToP+qkOI4P8A6jF/NC4LUkN1xHfs7a33Joz9pq5mM8DvocMqKvNEREwuNl7D5fjIuK6jd55h+S4HGvEI/s3NGyakeZjkIjkubHwTim3yh8jmuIuo9s/qoE3KSgs7Z3QJ9t/Cqkk2Lu2/hR2w6FUoTYihCFAIQhAJpIQCaSaBJoQgEIQgaAEBNAJpIQNCEIBCE0ApM1cPNRU4v3jQeqzl6bx9vvuGPyYXTNvtGFpMum65tFKz2GANe02jGxVxeSvg3b1rnzLPJKoOeVS551WGhI/fVZnG5U3G6ruix1sEHdkV9RHe4VnDFNFPHMXyZCCuhW4cGRmSOZjwOV9V650c+EunG5zlp4LHMLE7SQLOGzgvMGlfmLS20jd/FfRKyDOwrg1FCCc1iCOgUmX43txqJr22vcLqNDrKvs2B1wXtPO7FfG9trZx6tKhSseiibt0WgHxBUJC1rC51rBajKsE3V8eyyw1lNIbZx5roxMjcNHBdJLWbULIstbYW9QmYAVvizthcoFb3U1+SrdShTjV25s1w4OVUT88rtb2XSlo7sK4+B0pfJVPJJ+dICvH9XbddSAV3sqYpSFNU2g1NWinICDCVrVRSSkTorHRu6Kp8brKKiz3b80nFPK4DQKt+ZAnFVuKDdVOLlFDioEoLioF56KKZUHGwQ55G4UASXXKIlbRCL3QtILouhIoHmKWcqKSqJZjdSL7DQ6qgOvIQFYFRYHHfmjtCoIuqJGQpdoVEqJK1tEu1PVRdMepVbnKt7wE3RY6crhcQz3jjZcakkrpucXbaBefxpxNU1vINXXp22sZTw5qSlZC7uaNkJoQJCaFBWhCFUCEIQCEIQCaE0CTQiyAQmhAITRZAkJ2QgEIQgE0kBA1JmrhqoqyCMyysY3dxsFL6We3rKWOXs2kVzSLbGIaLa0VVvm66Pycwj8CuSMBxCJoLYyRb6DkmsxCn97tWgdRdeHLpX49UyjvRuxX6NVAf+I8fmtEMuNB4zuheznlmIP3rzzcQqI9HtafSy1Q4y5pGZr2+Wq5XCfuLW/8AXromyPLQZpAXaWcRYeq3yYJirIu0ET3C3IA/gvJw46AQGyNd4HQr1OBcWmC0Uwc6LpzaufHD9N5Rmw7H6vBq0xV1OWRu3IFnN8V6ttcKmASRPZJG4aGwUK3D8Px2kzMyuuNHNOoXlXUuI8N1F4wZacnVo2P6FbnUsnFnjMvLvTtzXI08FjnjzlugFgrqaugroRJC7Xm07hEjTdcsqrFNh7JG57OBH1eawmGEHR7x5tC9BEwlq83jctTBiLmU7GOG+psmO6WrBEz/AFp9WqL4wRbO0g+C5ZxHEGHXD3vHVpCfynUWu+ikb4G11uY2Jt3cFrm4c59PNSwyMcbxyOjvlPiV7WjxqinhGeOlD7agSNH4r5lBiXaSBnZFpP1tFviqCCGlpF+q92PW1NWOOWD6Q2ooJf8ADwu8iw/mn2eHu3oWHyY0/mvAMnB3Cn2reg+C6d3C/jPGvdOosLdvQ28oz+SqdhmDOOtO5v8AUF4xtS5vuuI8jZTFdONp5R5SH9U54VOOT1jsEwV+neb/ADkLJS8M4E1sns9S8ZnHNZw3+C8+cUrGMcRVTaC/7wlYsGxyvfR5zWS3c4ncFXeC6yez/szh592sf8WpHhWA+7WH1aF59uN1w3qXHza0/krW4/XAW7Rh84wp/wBZ/Z1zwmT7tY0+bFW7hOf6NTEfMELnjiCs5iE+cf8A1Vg4jqhvHCfQj8010zeS93CdZykhPqf0VEvCmI8mRHyen/aedv8AcsPk9wR/auUGxgt5TO/RS49NeWTJLwxijdqXN5PBXDxCCqoCfaKKZo65dF6g8XuA1hlv4SA/iFwsVxepxLuOe8R3vlcbrnlhhJ4rWOWXxyg/O0GxF+qi4eCkRbRRJXmdlZAUS0KZKgVBW5oO40Q3snGwcL+ahVdoYiI2kk6acl6bhTD8Gp42yVUdJNJbXPKLg+RXbDp3JjLKRwm04PJT9l02X0ZpwR4sKWlPkW/qpdjgzv8ACQ+lv1XXsX6x3I+amkd0UDSnovpnsGDu/wAG30B/VRdg+Du/wxHlmTsVO5HzM0ptsVW+nIBX0t2AYM76D2/zH9Fnm4YwV7DeZ7B4vCs6NXuR8tw2B0navJJu7S66AgK9xT8G4QyECKtfY66OapO4Noz7tc74AqXpZbXnHhTCVAxkL3DuCWn3K74sVDuCJ792sjPm0qdvI5x4tzHKtzHL2UnBVaPdngd6kLO/gzEQNDC7/iKdvL4co8i5juSrcwjU6r1E3CWLtBy07XfZeFxa3CsUpHZZMOnJ/gbm/BONn4bjmO0XmsRfnrZPA2XoTKXOc1zHNI3BFrLzNQ7PO89SunTZzUoTSXZzCSaSAQhCCtCEKoEIQgEIQgLppJhA00k0AhCEAmkhUSSQhAJpJogQhNAluwdmfFqZhG8gCxLpcPC+O0Y/3oSrH1V1IALAbLRhtOw1rGzRNkY7QtcLhbezudlZCzs5GvABLTdYhtxMQwSmdVSMNMzKDpYKNLwVQ1sMmd74AwXaQLr0Mw7ed0lst+SvpyI4nMsbuTUXlXzqv4JrWi8MjZQNr6FZcNpMRocRZTuiLyfoOuvpr47rnV1N842QRuc9uoLBdw+C8/UwljthnfTnU8mI0d3wRPhPPKbj4FdiirjisfYzVTRPbWOWG2by6rlVE0jnAAVLSTqTE9KWWlAAe5+Yag5HAj15LxZdO/jrv6vrcGqoHuno7CSPUuaDY+YVVNij53CKaPsphu2+/iFpouJRTtcx+etNtMpDXgeN7XWOrqafEJQ4YbVU7r6Ps05fgU43KefZvy1Y3iE2GYLJWQBvaN2DhcIj4XwbE4I6yqos00zQ5xbI5uvoVx+Iq5zeHp6aoBB0ySAaPXscKb/mum2v2YXu/jdOSbscOrXFdwRgAt83PGTtaqcPzTPAuGH3KmvZ5VF/xVAhxZvErvbn0biT/kpma/s7eFtAfNduWoxqljMr6KjqGt1c2CZzXkeAcLfevXwx+OO65R4Epf7vFMQafFzT+S8hicNdQYlLTtxOV7WHQvjaSvqdFVRV1JHUw3yPFxfcLxkWDtxjiypbKfmo3XeOq49TD1xizKuDTTYzLpHUCX/+vf8ABXPq8YgPzhh/niLV9KML6CBrMOoY3gaFokEf3rNDVNxSeagrsNydmLuzPa8fEJ2V5188+VMUA9yjPmXBNuL4oN6Wkd9mRwXS4lwVuE1YdCSYJNW35eC5lG5rauIubmaHC4te48l5eVmXGxqVN2L172OaaCPUWu2b/oqcPrqqhp2QvoC4N3c2Uar2PteCSU0wPYRucRZrowxzfQgKyvfhAgaWspHnMNBlC9Hbqc3lflxw3w+o/lLSpDiGIe9Q1g/4YP5r0VfR4bDTulMdMy5ADmd/7gQuXjLMOgpmxRtjFTveNjgCPiQsZ48ZurM6opcZp6uURMhqGvI2fHl/NWurmN96nqm+Jp32/BcmBhlnjZrq4L6nGHNiY0EiwHNXpYzqTZlnY+enEabcukb9qNw/JVDEqS5vUNHmCF9IyPdyJ9FS9sJk7N4jLyL5SBddf+PPqdyvnhxGjO1VF6vAUTW0x2qof+YF9AfRUbjZ9LTknkY26ql+D4Y/38NpD5wN/RZv8efVnV/x4MzxO2mjPk8JXB2cD6r27uHcGfvhNIfKEBVP4TwR2+DwDyaQsf8AGv1rvT48YVGx3Xrn8IYCQf8ANzW/ZkcPzVDuDcG5U8zPKd4/NWfxL9S9aPM3I5FSDj0v5r0R4Nwo+6+sb5VBUTwXQ/RrK5v/ABr/AJLX/Gyn6nexrhB/8Lfggy25D0C2Y1w23DaDt4MSqs17APyledjgxCZwYytc9x5FjVzuGWN1s7mLq+1Pbs8jyJR7dOP76Qfzn9Vz/YMYEgjDwXkaDsv0VRpMZLjYxutp+7Kms/q88HYbi1Wz3amUW/jKqrsfxCOkcW1s4J099c3sMUDe9G0u6ZDZZ6iGvmblfCwBpuSCVqc/pvB6amx7EGQsaKuQgDnY/krxxBiH+0A+cYK8uyrrGi3ssZ8pFP26pA1ox6SBN5r/AEen/tJXDd8R/wCEEDimtB/dwu8S234FeW9umJ1o3nycCkcQcN6ScegV5dQ1g9Z/a+qbvDF6OcPzUv7ZzjenHpK5eYibWVMXaQ4bWPYfpNjuFCRlW33sPrG/8Aq8up8OOD0dRxpWEAQxlh53fm/JcfEOJMTrWOj7cwxu3bGbX8yuXI97fegqW+cLv0WV9Q0aZZB5sIUuWdOOJVTuzp3uvqBdeXcbkldvEpz7M4AbrhrfTmomV8hJCRXRgyUkkIGkhF0FaaSEQ0JIQNCEIBNJNAJ3RZCoE0kIGhCEQ0JJoBMJJoGhCSoa63DLc3EFGLf3gXJXd4Obn4mpARfvIPswbqrWs52Ugy6sY1TSbTqWAljgBqOQSYzTbVTsXAC97K2NlldJtEQXF3KbKWBvzz6iOC2gc46eqtAubWJ8lJ1p5bPhbJGW2LHC4ss5THXlcbdlHkkI7OvpJPszD9VJ1PPJIezbHIP/ALgv+C5dVgOCUsElVFhNLE+MZszb6KVO6CrpGPdDHI0j6TdVx49Ouu8nRdQzn36Jjx/KfyVbqBv0sMJ8o2n81AYFhbTdgqoyeTKlw/NS+SKb6Fficf2aon8VOPT+nLJwuLsKppeG6vJhrhIG3aezOh+K6GEt/wA101/9WE8WwCeuwuelp8drmukbYCZwc0+B5rzk/FruHHtw2twirlfA0NMlOQ5jvK9l2w4zxKxlbXri0OFiAbdRdKYfMSfZK8e39p2Ej95huKM/4TT/AOZWt/adw6dJG1sd989Pf8Cuu2Hc4aB+Qaa/RcHCsSio+LquGctYyZ1g48ir4v2j8JtAa2skiaNh7K8AfALweKY9h1Xic88VTdj3ktJYRdcOrnw1Y1Jt9hrqh1LRvljY6R9u41ovcqrCqR1NSXl1nlOaQ9SvmuG8f1dBGImV8U0Y2bOCbeu62VH7Q6yqjLG1dHADuYt/vK138NbTjXZ43rYpJIqNjg57NXW5LzVHE6SqiY12Qlw7wNrLGK2CV5c6pje5xuSZASVtopacVDHSPuwHXs3gO9NV4Lnz6m3STUewNJUNAa7FAdbDtKdpJ8tFhxQTYdGHPdRz9pcZTTBjvNQbilI8gmvxFuU3ALmuWPEqv2wsDaiWVjBoJWtBHwXfqXGY+Ejk9mLkhoF+gTDFoyJdn4FeHzW1uExB+LUresgX0y3JfP8AAIhJjNOLe666+gmy+n/GmsHPP28tWMjdi9S6o7ENDhl7aOo28CzRbWGaTHv8mELoo4G5s2a9jf3fhzXczH6x+KzNpWMq5anMS+UAEHYW/wDdelhya6Sd9dFUU8NTG+Jpb36Qvab26O8Fpoa2WU9nUseJDsRTPY31vdWVFFVmt9qpaqKM5cpbLCXj7iFphFSIrVEkb39Y2lo+BJQU1NTDStBmz2dp3I3O/ALzdTJQSGVsAqqaQSDLITPqOtiLL1UhmEZ7BzRJyL72+5c+ChxCmqzUCvE/bEdsyQFoH2LbeqDQyWOeBr4n52O0BsRf4rM2mkhLnRssSSSAb3+K3TgvjLQdTsSsMdFNG1g7YEg3NiRddIxQz2zZ4PgQwffroiH2kZGPad+9pf71cYpDJmElhyFzb4Jtjk7YFz+4OQKVI4HGcuSjii+sV5fDw0zZiIHFuoEx0Xa43nBrYoQfdbdeXuvm9XPWbpHom1NXDIS32O1vc7X8Oil7bWNAeWQFn8MmoXm9OilE1jpGh7sjSdT0Sdapp6MVNa5pAphrs5kt/wAVTitU72LKad8OY94vI73wQzB8PmYOwle5zr5SHggedlzcTpYqAxxCV0klrvBOgPgtZ55SeYSDDqCTEqpsEWn1ndAvXR4NhuFwhxpH1L9i7JnPwWDgrszDUEe/caL0NXVMo4DK+5OzWjdx6L0dHpzjuplfLlF2EVkopX0L2SP2DoMh+K87juDHD5x2RJjf7t17CippAXVVVrUSf/gOgXK4gkZUV1JQtN3ufcha6mE1sxvl1MJp/Z8MgYBY5QStTnAGxeAemZVYi4wUBaxxY5xDGkbi6pZg2HNiDXUMEjubpG5nE+JW5PDRPrmCuFHJmY9wvGSdH+SrxGRsFBPM4CzWE6hUM4do/bhVywQsMZvFHC3KB4k8ys/F9R7PgcjRvIcqqfr5hi7yYA4m/aOJXEXUxkkPjj5Bt1y1556dqRSumUlUCEIKBIQmiK0JJqoEIQgEwkmgAmEk0U09EroQCaSaIEIQgaEIQMISTVAgIQEDXpOBGZ+K6Uef5LzgXqf2eNzcXU32T+So+ytCta1RYrmBVg2NViQCkgbHZLkDUjQ9Ep5/ZWAZHZhq63MJXI1CxYnLIymfK25c0Xva6zl6We2Z3EdHPnhqIXxsf3SDobeRVNDidJHVexUcUjmRi4c7utPhdeSfxxUPc5klPBMAdnRE/mscvFMcxt7AIz1hzD7iV45ZZrT060+qsqM+7HN+8fEKXa+K+aUHFlRTyBxp5nt+qTZdWPjjObDDZjbez/8AovH1OnZf6uke37XxR2+lsy8X/bunHv4fUt8bj/omOO8OdvDUN/oP/mWNZxdR7Bzone/HG7zYCss8NA5pz0VM7zhb+i863jPDZBtUNHjGD+BSdxRhsn99KPOJ36Jc+rF4RqrsNwiQG+GUvpEB+C4rOHcIqKxjDh8NnHkCPzW9tdFWn/J3PkvsMhU6X5muj7UOj1+m0j8VnHrdXl5q3DHThVnCuEsqnMFE0DkA4qr+ymDW71I4HwkK9PU07Zaok7ErnVmEPjcXsrqhgP0eQXsnWy/XG4YvPVPC+Cse1jYpDI86M7YAn4hdGl/Za2riE0dJK1p/3zCrBFMJBeqLwD9JoWuGV8Z0e4DwJC7YdSX2xlh8Zj+yUj6FU37OU/mqJP2XVUZ+akrwPsfoV6OhqZZH2NRI0fbKsqMSq4HkR1kot0euu8fjGq8g/wDZ3ijPcqqseYf+qrPA2MRjTEakebZF635exMWtWzf1K1vEuKN/xbj5tBT+nxdV46LhviKhlEtNjE0cjdjldcfELUXcdRbY/I77TW/+leq/tViYH79h84wq38XYk07wO8DEFqZzHxEuO3ljinHcIucVjf8AaYz9FA8Tccx/4iif5wt/VeodxnWlvfpKN3nGvPV04rat1S+KNj3co22CmXW16WYbUDjDjdnvR0D/APgf/wClIcc8YM97DaJ/lGR+arKg46qTrVe3Gpv7QOJx72CUr/IuCmP2i48Pf4eh9JX/AKKrC56KkqXS11LJVtOzBLkA9F3BjnDp3wOUeVQuk6m2Li5H/wASsSv85w630md+imP2mSD3+H5h9mX9Qut8scLO97Cqpv2ZbpGu4Sk/wde3ycD+a33GeEc0ftOgHv4FWDykYps/afh9+/hNe31YVvEvCTzbLXs8w0qXs/Cbx/plSz7UY/RO5tODxGN8VU2KYi+obT1LGHQBwFwuf8s0nNszfNl19EdQ8JO/+aPHnCP0WSowvg8jXH4Y/txALz5dLHK7XTw/yxRfWlHnEVbT47QwvLiQ/TaSM2XSxej4apwfZMcpql/1GxG5+Gi5DIqZwuY47+Sx28ca1Mdt44kwu1zT0+bqC5qy1GL0U8mZssUY6BxP4qIpKZ39yw+SlHh1PNM2GOkdJI86NaEuMy8bXgvw3iBuG1Inp6mO/NpOjgvVRcd4NOGPqYS2RuxBDgPJZYv2ZySRte6OAEi9u22Sf+y9/Knjd5TBd8LlhNMXGVfXcewFhbh8QzH6crxp6BYeF5nYhxCKipmD3AZi5zhuh/7Man6NIfSRqzv/AGa1rNqWoH2XA/gpblld2rMden0KopxVOjOfusdmsNbq0sd9U/BfNf8A4f4kwaMrm+RP6qP9jMZYfm5sRbbpn/VduUTjX0lzXD6J+C83xDROxeshoRVwUwYM5MzrXXmXcO8RwDu1+Itt9pUVNFVw2krnTSyAW7Sa5P3qZZzSzF57imkFDjMtMKiOcx6F0ey4yvrHl9U9x6qhYaBSTSUCTsmhArIsmhBQhCFpk0IQgE0k0AmkhA00k+SBoSCaAQhCBoSQqGhCEQ0wkmFQwvWfs5F+LYPBp/JeTC9h+zRt+KWHowoPsbQrmhVM3VzVplMISCaIR2VErc8L2OcQ1w1A5rQRdU1DB2DweYUqvMjB4mS2EUWS99GBXw4dSMk78DS3oGhXYNF2mHNtuHHmtjacnvH3fNefTptibRUl/wDRob/YCbKKiMrgaKLbfJZdNlohZjbeig6Zsb++619rrNxXk5UuBYdKxxEEQ/hzEH4LlVHCmGvackFndDderdGyTW1j1CrLC05u8X3uHgpo5V4R3CNI6UAiVg6tcl/Y1xgcYK2Vr72a03GnVe3EIkeSbZtyvMcWYziGBVcIo5GBrxctfGHBamEv4vO/U8G4Vx2heJBKKmNwuQx3eavROjxKWHsqzD6mdoFgWxgu+8rJw3xJUYhhE8kscMcrACTGbE+hXYZiLoWC9TFbf50m49dl5epjjLvTrLa5lJQ1rXgS0tSyO+hfEczVi4graiCEtY2RzgbAPpZGD+q1l6iPEmyWMc8crT9KN17KisxI9mWFxIPIm65XqdKT01Jlt86E9c28s8ADRrdjs33LWypMsTXQtzk8icpXSlhilrmlsTQTvYWWXJPK8D2dsYY64LdcynTvL01lNNuFtllcWup2uI+j27Gn71dU0dUXG1O0eHbx3/FUUxdHVsfI90eXXuvDM3hcrdU12Hvf8/RVLz19oafyXvkmnnt8ueaCutpSvP2XNP5puw7EGRh5o5yD9VoJ+4rSJcFdvR1Q/mYfyQDg3IV0f2Q0pxhtgdSVoFzRVP8Ayis8sFS0G9LOPOJ36Lsh2EBv+nYk3yZ+hVb5cM+hjeIs82P/APUpcF5PPSOMY77XN+0wj8lWXG3L4LfitRd4jixCWsj5mTMLehK5xdouN8VuEXeAUA4PNmgOPRuqsjZ20zYy5jQ7TM91mjzXeoaWtomZKLHMOib0bOwfi1bxx2lunnSx/OF/wKWUj6H3r1hn4gvZuN4e/wA6iI/kpNfxIf7/AA2b0gcunBjk8cdD7v3ptBcQAw3Xs+z4ldr7Nhr/APgwn81ZDS48+RvbYRQuaTq4QM0+BTgcnkoYHl4BYQnjWfDImukY5oI003XrailxkV7IYcAwySAnvzvDgG+gKWJwyF+SfAIqtjR3fm35fxW+2zcnzDtK/EXWiHYx/XddaBQUtHRvlmb28xIAc8bL2YxXhyihecQw2mgnGgihDi77zovMYzW1GPwtjwXAZIqdh/eMa6QuPidl16WMxy3XPO2zUeXq8rpbRsa0k8hZbGwuDQung2ByU1YJsawmsqIgNI4gY9fE2XqGy8Ls9/AK+P8A4v8A0U6+s8v6tdPeM8vBOMkZOVsjiNwCqo8Vc1/eEoH2l9FE3CLv8DXxj7QKm1vBzj7lWD4xNK48HTk8rQVMk8Pah0jW8tbLaKudvuzSDycQvQmLhQjSorIx4xBVvouFXNJ+VJmfai/RZuOf4u8XDbilaDYVlQPKV36qwYziDNq6p/5pUK9mHwO/zfVGobfnEW/eue6YgbLnc8p423xldUcQ4o3bEJh63Uv7WYqzT5Rf6tb+i4T6ggbKk1QI1ZYp3Mvpwj0J41xdu1aXebB+i5uJcXYxUU781WQLe7lFlyn1AP0bLFXzD2Z1tyt4521LhJHElkdLI57yS4m5JULpndKy7uQTSTQCEIUUJIQqKUITVYJNCEAmkmgEIQgaaSEDTSQgE0k0AhCFQ00kwqgTCSaBhe0/Zi3NxIT0jXiwvZ/s1pBVY1MDLLHljvmjdYoPsTVaFyW4ZUN/c4tVs+3lePvC0Npa8NsMTzHq6Bv6rTLohOy55jxYMsyqpS76zoiPuCgx3EEZ70eHTD+FzmFQ06gChUD5o+SxGuxWP3sFMnjFUsP4qubG2tYW1WH11Meros4+LUFXD7Q/Cyf945bnNtyWPhkh2FOI2MpI5LoSBc9KpOyoJvKVc7S6pA76yq5moUw26radFexXQGwgG4AXh+PqGWtxehpoW3fILD7l79jbBfPv2lyPhxCikjNnMFwVuTSLsDw6ehqJ6eISQBoFnzx3v4jwVtZheMlxfT4qxoOuUssFm4LrZKuqqZG5o26Etc7MAfDoF62SJsvKx6hcs5t1mWnjHS8Vwdz26BwH+7A/JXUI4oq61kcsUckZPec0gGy9S2jBOoV87ZaGhfNSxMknaO4x7srT5leXL+PjZvTpOr5c84XVQOa9tJO9431aQpSUT4mdq9haObTuPisMnFGPRd1+Cwk/wVBP5K2CrxfFI3umwp8DW/x5rrjLenPEdLOXtn+VIKLEs8zJHU7o7d2PNr8FjZX0VVXdi10zg/3Q5oYB6kLXOx0lM6OSKWnkj3a8gZvUXXPc/wBkcA6ZwduLyC69OHUl8VyuOnYZh8UUVnQ3PX2lo/JHslPzgd6Std+S1UOI0UsIBqMRDhv7zvvC0PqaIC5rawD+Jsh/JejjHPk5j8Nhf7sFQB/C0H81mmwxjAXFtQwAa5oRb/8AZdZ01Cd8Rc37cb/zasNVUUbnNiirY6gvNizs27erQnCHKvPOMTyQ2WM66EPA+5D42EgM7x6Zh+Soq3ULKqRvssVg62jAtmFxULZW1LXw072nuluS49CQuHHd8Oss/Q2hjdoZ2Md0Mcn/AKFF+FM5VcBv4PH4tXf9ve7/AOaMd9qMH8HqL5nyixxCm8+xcD/+63Onpnk8xNhOZ+RtVSMJG5fY/eFppsDkcwhstPLYbtmZ+q7sftJls6roHttfK2Bwd8SSFqijijwmSsrKmOJ7SSHghjAPG4XSYMWvKzYXPJQP9kdA+YG1hMCB6gp8N8O1EeIGrxC05ibcNz3aD8VfivFrZ4W0mDUb6x0ejpWRksv6BcmowfFMSaHVeKwBp17ICRoHh7q1JpnbsTcZ0GEyODc1XOCfm4zZrT4n9FzJ8Q4sxhnbwU08MEuo7IWuPO90qbB5aOLs4Y8HlG95y5zviQFF9JK113UvDvrJY/8A7Bb2acp/D2ImTNUUlS487tK1/K2M4TEIIa+tpo26Bglc0D0XQoY+2xOGjdRYWx0lznheX2t5OXHr8ZgZUSQuwyGTK4jOJni/pquV39ammhvGfEDTb5ZqvV91a3jbiDnisjvtNafyXmnTdo8ubG1oPK5KWdzdQGBc+db4x6+PjPHSO/Vxv+1BGfyU38VYhKwtlZSSA/WpmfovHiqltpkKi6qnc62UfFZuWS8Y7dVW5yCYmA9I2BqzvqRa2vxXJdLO42Bt5KTBM59i4lcq6SOhJUkN7gHqVnNRLfVrbeDkmxPcbBrnHwF1NtPKTZsTyfBpWW9KjK4/R+9LO8rR7JNzicPMKyGkMj7WefstuisdnlY8SBbCL8yvQmjji99kzR1IC4OPGISsbE8uFtbjZdOn/wCnPP05CSChet5yTQkgaSL2USUVK6LqF0eqJtBCEKshNJNAIQhA0JJoBNJNAJpJqgTSTQCEIRDQhNUCaSYQML337KW3xirPSILwQX0D9k4/zrWH/dD80H1NqsaoNU2rTKYzZha1uasaoNHNWBQSFlGW+Q2NrKQSeLsI8ERx8Nq4YWTsnmbG7tCe9otRraR21VCf+IF5fHLnhfFO+4FslwQdRuvmNLXTQTskknc+MHVhnLbpMNra+5GendtPEfJ4Ubsz3D22t1C+PVGLUs0EmR0kb9MobUu0+CrfiMZoohDPN7SPed2zj96dunJ9qYQR3XA+RVzAvjEWKujhaTVTufzAe4W/FRkx7GKd2elxOUNvozKLj1ITt03H3Fo0Xgf2gulZjNCYjGH5TYS+6dt7rhcLcTY5X8RUlHV180kMjrOaTa49F6vi3NBidJQU0ULxUA3dUgy222uSs28ZbVk3fDPwpPNNJM6eKJjwBrGQQfgvUMFyvL4FNHQST+0CnjANrwx5dfEWC9FS4hTVBIhMkhG+SMmyxPPmNV0ImDQlFUA6HKeai2pjaO8JG+cZUJamGV7WMkDndLFb0ypjo2k6jRY8VZJI5goKstAHeyO09bLshpbE4hpOmwCxU+V8LnZ3Os3UO5LF6csamennDX4XBAfaayaqlk0LooS/Lbl1VlLTUeP1rKUC/wA05+bssr9LLfwxgNK7CflMGVrpHF3ZusWj814niarwylxRvsuIiGoDz2vZykenguPZ43brz34fQ6HhlmFuZJBO8OJsWvbf710nUJlOVzITzvZ115zDqVk2C3inq/8AKAHCV1Y5zm+RFlnmwiqjZeLG8TafGsefzXbGzTnXpH0TnECOKm/mLv1XJrOHamorWVLp6eNsQ91rt/ivM1bcepnWh4gr9OTpiVx6jH+K6Z9hjdSRfnlP4hXcR35OAMQqJ3OjrqfvG+97fBc2q/Z3VOfklr6e48Cbrn0/F/FxkyDGpj4dnH/6VZJxHxHG7NJir9f9zH+i83W7012ZP/11w4X/AN1Y79nE1jlrYTb+A/qqXfs6xAC8dRCfQj80hxfjgvfEL+dNGoSca480ECphNxa7qdoP3FcZn/N/+OP/APW7Oj9rmQXwmomjZitCZCMhD3vGX7l0p66gqMMpaWqxClmdC36L9L+N915p9P7RM6WapAc43JyEo+T6e3+mM/5L/wBF9OPNtprHvbUWpKuBrB/q5w0fioNqMU+jXu9K0fqsxw+HlVR/8t4/JUvoAPckY70I/FZqumKziBp+brX28Kpp/NUVLscqwBUyPqANs8jXLKMHqjSGqDGdi12UuzDdQOFVohbN2DuycSGv5FQdzAIq6ikqap8BYWREN0Gp8FxXtrHPLn08upvfKVV7FUj+6d8VERzsue8Mu+uylm2pW+ONxsCCPNXina5tnSNF+qwsbUOYHGR9j0K3YVTPklBku7qHarl261zlSbAxtwHtJ81dDTzxvP8Ak0UgOnzrwLfeuzBh0L6mNjqUSa95jQASPArtH2Chvbh2oc3o57PyCzZr3Wt79PHdkJpjm9mp8vIPuCiNh9pc2JrJj1BsF6STHKGCTN/Z1luQkeSsbeIhLUvfFgtCwO+iINvVZ4y/q7sZImVMTs5oQ63JszQtcL6ypaXR4FJIBzMjiFsGNB8fdwfJIPpxPLUQ4liWocK1gPKOTdWYYs3OsRnxOI5Y8JbEejGXP3lZ6ibFS4Nmpqhlj7uUNW6eoxkyHsBUhp2LpNfuWCdmMSOJkZMTzJN1eOK7rJUe1Zcz6eTL0vf815uvkz1JOUttyK79UytaxzpDI2w1uF5qYkyOJN9VvGSM21WhCF0ZCSEIBRKkla6CKLKVkkRWhCFUNCEIBCEIBNJNAJpJqgTskmgE0IRAhNCoE0k0AmEJhAwvof7Jx/l9Y6392PzXzwL6P+yhnztY/wAAPxQfTWqwKtqsaqysbsphQCkEEwgnT0SCHbIjxWKd/AcdbfRhuPvXzM02JVNGx7aRzmAWEgaNvivpteScL4iZbZv6r5lTyFlO055SzKbtYwXB87Lph6SufJHJC/JIwscORXQp8QhjonQOi7x5loNlzXF5cS+5dzzbqbf3bluzaE4NJNhorYGZHgyU7nN3vkKjFH2jw0vawcySu3DOImhrMRZa1rZHEfiqLeEJGP41ojHE2Juf3Wkn8V6r9qMz4q2idG9zHNBs5psRsvLcLn/+cUfzok7/ALwv+a9h+0Gpp6bG6CWqjzxtabtLQ6+3JebqWTe3XHfjQ4GpqfEcNkdV3qXXBIlN7FewpcPpKRxdTwNjLt7LzvA80FTTVE1PAImFwtaPJdesbssY60mU8ptJHNVVAEha1/eHQlWhVyayt8ltHMxhkcFDnaDG4O95rnD81ChMhwmSV00hOUkEm+nqjiRxGHgAkEu3AXieJONfk/B24NhxEtTI0tmlIvkHQeK8mGWV/kWb8aeiydqVe39peHYLgTcOoaWWpqmkh7pDkjHlzK8FTuwyvqnmudPDJI4nOwhzCT15hV0GB4jiUhENO/L9KRwNgqKuiqsNqOyqYXRu5Zha69f45R9owCD5M4fgp/aO1Abma48h0CnJPJ2dxVPaSOQC+acN8W1lNIygqZe0pnd1ubdnkV7J9X3Rqdlxu5Q6t0jy7/KPUtC4NUxxN3Pza7ZQFumqA65F1z6iS6yOdCzNVm/0TpZa5GZnXJv6LHSuJmefFb2HurtIzkzvY0j3dfJZnRAnYfBbnusLWWa+q1pjdZxT5nhulz/CqZ4xHVvhble1g1dltqurhjQ6s7Rx0jBd6rk531Ess9nHtHkk2SxSEbL+6FcylaRmsPIpQ6uAyhacLgxPFMU9nhYOyae9maAAPVax6Vyls/F5PccLcOUVbggFbRMkY52YDZczjengoGU2HU0RihZdzWjbkvcYfLTUtM2F0sbMosBdeR/aC0TVFNURWkja0hzmm4B0XNdvDiJoNzfTxXIe4di4kXL33XXqjkpZH+C4u7omE7JWsfLZDMxoaw20XYwgMkkc86BcylLQ+5XosBibKXvI0voqkej4cgjlxEyXB7Nq79TBHIDoLrDgUbWMlkIFybbLfKW/VHwXmz9uscGtw2N1y6MH0U8OwenZFfs9T4LfLHG8EFjdfBaYYYWxgCNo9Fzka2pjw+Frf3f3K32SIH3B8FeI4/qN+CeSP6jfgusibZH00d/cWOoporE5F0nRxm92N+Cw1MUWQ9xvwRl43iMRw0krgLX0Xzp5u4r3vGPZx02VrWg2Oy8EV0kRFCaS0BCLIsgEIsiyBIQhBUhCFWQmkmgE0kIGhJNAJoQqBNACEDQhCATCdrphqBJ2KkAE0EcqYCaAFQAL6f8Asqj/AMlrJB9YD8V8xBX1P9lIHyVWO/3o/ND8e/arGlVNVjSqwtGykoBSBQSCDsgJOOig8lVMdJFj8TWFxdHoANTuvm8ENRBTNa9s8bQCHgwOuF9iZRyx1ktTTyRtdILOD2uP4EK8urgNW0z/AOZw/G61M9Fm35/mbM+Rz5GvufrNISbYRuBIHmV9+c6oN81DTnx7Uf8ApVRa1zwX4ZCfIMP5LXcTi+DshfKCWNDgN7OH6rpUtH2UJHtdORJ70bmkEeq+zOpqV2+DROP/ANpiiKGjcO9gEZ8Ozj/VO4vF8r4bJHHdHmdG7vjWMWC95xvwpiPENVDJROgDY22PaSZV2I6CghmE0fDwjkadHsjjBH/5Lc6smA0w6qP9A/8AMudu6unC4Qwmp4foJKXEJIs5ILezcXC3wXoBV043mb8Hfoo0kr5S9zoZIddnkX+4rTmP1j8VItU+2U3+ub8Hfoo9qyeX5mVpsNe6StGY/WPxVYN5nanZEUTUxqGZZmwyD+JpXyziT9neLCuqK2gEMlMbv/e5S30K+tuOixYjf5KqbH+6dzUmMl3J5XldafnkyTxvIE0jS027ryFBz3yOzPe57uriSVKb9/Jf6xULLQbA8yN7MEvvoGi5X0CnnnNLH2zJA/KL3abryHD8NRNjVO2mBLy4aDovrcOCNzlz2uIA5FYym2nkX1DQO9mHm0rDU10EZsXOudrNJXqcYwsRubkDx5lcGoo3gG+ZZkRzaT3XHqt7XWaFCKmy2a0b8lsdhtawXNO4BdIzWJzXSPDQQC42XTj4Uraht46mneOdn7fFc8xSRSAuaWka6rbS4hLexOpWk0DgdZSQzMexw7QZQ9rL2+C20DKWCljpYg6NrBZ5c03d56LczHJI2xhga5w0Jfr8FrdW0NUwGpjcLkgOGu3/AHzV8UkcqqwimxENYBAyMkd4Pc1/4Kw4K2muYp+xt3WmOQnKPjqVfPhNJIBLBUvBIuAO9bzKxOgqIHhzKkO6Z2nT70viaXTb8m4k19ocVlBt7rrEN8SefkuLxD8pRU0UT6llY5xJJlZlb6W/NdSOrdG0Nke1wP7xzhqVojxPC3ubC+WnzHZh1PwWNK8FmlqInQuoaR9tSGSOaSsLoIXSNd7JlLtrVBsPiF9Rkhw+YBr4W2OxaCLfBYKjh7B5DZsLxm1swk3PVXiTw+fNgL/m44H5gdu0bcr1mESshApzBJE4N2cQ4n4KyThanz545JhJyJhIsn8lVrZI3xDtXNP0YnAn4poj09ASykFiW31sQpve8/T+5Uxx1zYWn5PqDpyATdDW2uaKo/oK8mUtrtNIOc8vAzj+la2PfYd5p9LLCIaztLmhqQBzMavEjmnvxStt1jIUksWtYkk/3f3pds8DVoPk5ZhUMOgeL9LoMummq2ytfO8NPzf/AOQXOq6g5SMjvQKdVVZBYDM47BcOrrmRZpckk0hH0To30WsZtLdPJcX1z3VZp8lgNSTuvL3XWx6oFRVZsrgbknNzuuSulmmQhCEUk0IUAhCLoBJCEFKEIWmQmkmgEIQgE0k0DQEIQNNIJoCykAkApIoTUU1Q7p3SATQNF0kKoYX1b9lYtg1Weso/NfKhovqX7NaykpsFnE9TDE50gNnyAFP0vp75pUmvf2mXs+5b3s35LEMUw+1/b6b/AJrf1Uhi2GjfEKb/AJoVZdC5t3bHzUwdFz24zhh2xCnPk9MY5hV7fKMF/tf9ER0Ah2y5juJMIYbe2Zz/AARPd+AQOIcPkHcNQ7ypn/oorW3QlWX0WannM93iKWNvLtBlJ9Fo5LIRS5plLmipKQSCaBjncDwTNrKN0E6Kgg3f5q1c+TE6Khk7OpnEbnC9iCVW7ibBWi7sQjaPFrv0QdMlc2uxaDCpC+p7VzX+6I481rKl/FeAtFziUdvsu/ROKtocYYZqV4mjabZi0jVBldxjhhIAirCTp+4K52N4Lh7sNq610L8xjL9SRqu8KaK+jGqyWlbW076R7rNmaWEjW10lH55dYuJAtqlZdbHMHGGcQ1GFwPfL2cmRpc2xPorMe4bn4ffCyoqoJXytzZY73b5haHT/AGbsz8UsGXN3D6L7M2AMitkuSvnX7KsCmbJLi8sRbG4ZIyea+mu3sAo1XAxaAPeO4BYLgz0rQT82CvVYkHGQacui40rDf3Xf0lTSOWylaNezsU5Ir+8XLpiI23PwUhELd7fyVHnKmlhsb2aTzK47HZZ8txofivaT0scjCDlPovNYnhzmEuijOnNo1RNMks3ea4HQXPwV0FURKWPkytZ3Oq5xfKLCZjgNr5bKylPaSNcb6yFx8tEV6COrytLWyEgaBJ82bW5WCKSxtyVj5GgE5itMq55s3cMT5G/SYw6kLj4GI5MbEmXI1t3anYKdfWRDO1+d1wbZHWsVjwsuzSljspLct/Ncs/8A1NLHp8TxeKqiFPRTB5PvPjOy24TSlkIlkzEkaXcbri4NQRRPbE3UDVzuZXqA5rRZugC2ITEDckAeJXBl4mq2CSnp6plJG92USu3HkeS6GKVXZ0srr8tF4iueWMjANiRdEt1NutIRIS44vRSO5ufVOBP3qsdu391ilB6V1vzXn3TSZbCQ/BV55fr39FNJ3HpJZ8TjYXx4rGbco6+5/FY6nFMVbkDauokdbvEzu/VcoAyPa11jc9FVUET4iGN2zWCzWuT10Jw+ppWPqcUqIJiO8DUE2PqqXU2Hg/N40ZSfrSltvXVXR3ZG1pDTYcwFFzYHjvws9BZYmGX1e7j8Y5+yYHdnWyuI5tma4H7gVSKyWoEVG2W73mziR+ey0VUVMGNa0Bube7QVyZpzR3lhs17diAuuMsZuc3pmx8MjxF8LDcRdz4LlKyaaSeV0sjsznG5KrWa2EIQihCEKA5pJpIBCEIKUIQtMhNJNAIQmgEIQgaEIugY1UgEgFIaIposhCATukhVDui6SEDTuopqiQX1n9nNJTv4dEkkEb3OedXMBXyZe44U4pOF4ZHTPrKCONpJySB3afdokS+n1BtJSf7LD/wAsKxtLSjamh/5YXixx6XktpKR9WR/q4yB8V1cDx7F8SxLsavD4aSENJLS671azK9KyCEbQxDyYFa2Nz7fkoOw17dRV0bvsz/8ARHOSqYP3uY/RF1DCmGbFGu3ANyrTCWU8vfjLnd0EPFlZg8TqeV8j2k6WfpxrnloPkq3tYATZbzGBrcKmaNjo9NDzV5T6arh1bWvPukeq8Xijga6SxuLr31TTZGOeToBcr51Um9Q/zTcppUhJF1FNIoSQCEIQIoTSQKyLJosiKk0k1UCEIQCaSaATSQgd00kIJJqIKaBppIVDQkmgClrWk6XDIMoLnyOt0NlvhpKNkl/Zs9ubnmyugprgEtc0eJ1WpkLb6Anw6rOxmigiD80cUYP8LQtzo2O10+zvopjUX/NSJaEFXZHoUZCDqVMuAFzfzQHZvJUIMG9vuWijjAzy2PQaKkuyt3K2xN7NjWd4AalBO2wN9PBJ9izKT75A2TvcE3PRRLgZo2knQFxHQqi0gam4+CWUaaj4KN+5vunfUa/cgyEanW+v62g06XQN9hsCAed0mvtsNuihqAbsuTyJURdu7beqCxpBOjRv1VjSeQHxVLnOsQAEMvrbUc0Glrh1cd2BzrDXQG6quGsbGwEBo2Ui/uNZYi2puEE4wWBzxYgC17oYHR5n3BsNTdRDrRNbYgk3Nwq6mQto"
    guide = Guide(encoded_image, box, situation='board')
    res = guide.start()
    print(res)