import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import rmtree
# for i in range(1,18):
#     for root,dirs,files in os.walk(f"../Data_fall2/HAR_UP/Subject{i}"):
#         for dir_name in dirs:
#             if "camera1" == dir_name or "camera2" == dir_name:
#                 print(os.path.join(root,dir_name))
#                 img_folder = os.path.join(root,dir_name)
#                 img_names = os.listdir(img_folder)
#                 img_names = sorted(img_names)    
#                 #print(img_names)

#                 fps = 30
#                 frame_size = (320, 240)

#                 # 動画ファイルを作成する
#                 out_file = f'../Data_fall2/HAR_UP/Videos30_{i}/{img_folder.split("/")[-4]}_{img_folder.split("/")[-3]}_{img_folder.split("/")[-2]}_{img_folder.split("/")[-1]}.avi'
#                 out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

#                 # 画像を1枚ずつ読み込んで動画に追加する
#                 for img_name in img_names:
#                     if not img_name.endswith(".png"):
#                         # rmtree(os.path.join(img_folder,img_name))
#                         continue
#                     img_path = os.path.join(img_folder, img_name)
#                     print(img_path)
#                     img = cv2.imread(img_path)
#                     #print(img)
#                     # print(frame_size)
#                     img = cv2.resize(img, frame_size)
#                     out.write(img)

#                 # 動画を終了する
#                 out.release()

video_to_play = 0  # Choose video to play.

img_folder = '../Data_fall2/HAR_UP/Subject6/Activity10/Trial2/camera1'
img_names = os.listdir(img_folder)
img_names = sorted(img_names)    
#print(img_names)

fps = 30
frame_size = (320, 240)

# 動画ファイルを作成する
out_file = '../Data_fall2/Subject6_Activity10_Trial2_camera1.mp4'
out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

# 画像を1枚ずつ読み込んで動画に追加する
for img_name in img_names:
    img_path = os.path.join(img_folder, img_name)
    #print(img_path)
    img = cv2.imread(img_path)
    #print(img)
    # エラーチェック
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue
    img = cv2.resize(img, frame_size)
    out.write(img)

# 動画を終了する
out.release()