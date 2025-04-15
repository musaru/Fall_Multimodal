"""
This script to extract skeleton joints position and score.

- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the
    bounding box.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import math
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(".."))

from PoseEstimateLoader import SPPE_FastPose
from DetectorLoader import TinyYOLOv3_onecls
#from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast

save_path = '../Data_fall2/har30_6_new-pose+score.csv'

annot_file = '../Data_fall2/har30_6_new.csv'  # from create_dataset_1.py

video_folder = '../Data_fall2/HAR_UP/Videos30_6'
annot_folder = '../Data_fall2/HAR_UP/Annot6' 


# DETECTION MODEL.
detector = TinyYOLOv3_onecls()

# POSE MODEL.
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose('resnet101',inp_h,inp_w)

# with score.
columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


annot = pd.read_csv(annot_file)
vid_list = annot['video'].unique()
count = 0
for vid in tqdm(vid_list):
    #annot = pd.read_csv(annot_file)
    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # Pose Labels.
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    #frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(frames_count)
    # Bounding Boxs Labels.
    #annot_file = os.path.join(annot_folder, vid.split('.')[0]+'.txt')
    annot_file = os.path.join(annot_folder, vid.split('.')[0]+'.csv')
    print(annot_file)
    #annot_file = os.path.join(annot_folder)
    annot1 = None
    if os.path.exists(annot_file):
        header = pd.read_csv(annot_file, nrows=0).columns.tolist()
        # print("Header names:", header)
        annot1 = pd.read_csv(annot_file, usecols=['Activity'])
        # annot1 = pd.read_csv(annot_file,header=None,
        #                           names=['frame_idx', 'class'],skiprows=1)
        annot1 = annot1.dropna().reset_index(drop=True)#データのインデックスを直す
        # annot1 = annot1.dropna(axis=0)
        # print('err')
        print(annot1)
        print(len(annot1))
        print(frames_count)
        assert frames_count == len(annot1), 'frame count not equal! {} and {}'.format(frames_count, len(annot1))
        
    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cls_idx = int(frames_label[frames_label['frame'] == i]['label'])
            # cls_idx = 0
#             # cls_idx = int(frames_label[frames_label['frame'] == i]['Activity'])
#             # cls_idx = None  # デフォルト値としてNoneを設定

#             if annot1 is not None and not annot1.empty:
#             # 'Activity'列の最初の値を取得して整数に変換
#                 cls_idx = int(annot1['Activity'].iloc[0])
            
            cls_idx = int(annot1['Activity'].iloc[0])
            '''
            if(annot1[annot1['frame_idx'] == i]['class'].values==1):
                cls_idx=1
            else:
                cls_idx=0
            '''
            annot1__temp=annot1
            annot1=None
            # print(f"inf loop now {annot1} {i}")   
            if annot1 is not None:
            # if ret:
                bb = np.array(annot1.iloc[i-1, 2:].astype(int))
                #print(bb)
            else:
                # print(detector.detect(frame))
                return_data = detector.detect(frame)
                if return_data == None:
                    bb = np.zeros(4).astype(int)
                else:
                    bb = detector.detect(frame)[0, :4].numpy().astype(int)
                #print(bb)
            bb[:2] = np.maximum(0, bb[:2] - 5)
            bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]
            annot1= annot1__temp
            result = []
            if bb.any() != 0:
                result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                                torch.tensor([[1.0]]))
            #count frame
            
            if len(result) > 0:
                pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                     frame_size[0], frame_size[1])
                pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

                #idx = result[0]['kp_score'] <= 0.05
                #pt_norm[idx.squeeze()] = np.nan
                row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                scr = result[0]['kp_score'].mean()
            else:
                row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
                scr = 0.0

            df.loc[cur_row] = row
            cur_row += 1
            i += 1
            '''
            # VISUALIZE.
            frame = vis_frame_fast(frame, result)
            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = frame[:, :, ::-1]
            fps_time = time.time()
            i += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''
        else:
            print("brake now")
            break

    #cap.release()
    #cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)
print('Fall: ',count)