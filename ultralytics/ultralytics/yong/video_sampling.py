import cv2
import os
path = './subway_data/video_data'
# Video file path
video_name = '어대시작'
video_path = f'{path}/{video_name}.MOV'
save_path = f'{path}/sampling'

# os.mkdir(save_path)
# Set the number of frame
frame = 10 #@param {type:"slider", min:50, max:400, step:10}
vidcap = cv2.VideoCapture(video_path)

cnt, num = 0, 1 # cnt -> Input frame #, num -> output Frame #.

total_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
cycle = int(total_length / frame) # calculate cycle

while vidcap.isOpened():
    ret,image = vidcap.read()
    if num > frame:
        break
    if ret and cnt % cycle == 0:  
        
        try:
            cv2.imwrite(f"{save_path}/{video_name}_image{num}.jpg", image)
            num+=1
        except:
            print("fail")
            
    cnt += 1
    
vidcap.release()