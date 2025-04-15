"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import os
import pickle
import numpy as np
import pandas as pd

class_names = ['Bending','Lying','Empty','Sitting','Standing','Crawling']
main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5

csv_pose_file = '../Data_fall2/har30_6_new-pose+score.csv'
save_path = '../Data_fall2/har30_6_sensor_new-set(labelXscrw).pkl'
# Params.
smooth_labels_step = 8
n_frames = 30
skip_frame = 1

annot = pd.read_csv(csv_pose_file)

# Remove NaN.
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)
# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label'])
print(np.unique(annot['label']))
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels

video_name_set = []
feature_set = np.empty((0, n_frames, 14, 3))
labels_set = np.empty((0, len(cols))) #labels_set=0
sensor_set = np.empty((0, n_frames, 15))
# print(labels_set)
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    add_df = pd.read_csv("Sensor_HAR_UP/sensor_har_6.csv")
    _annot = pd.concat([annot,add_df.drop(columns=["TimeStamps","Subject","Activity"])],axis=1)
    _data = _annot[_annot['video'] == vid].reset_index(drop=True).drop(columns='video')
    # print(_data.shape)
    sensor_col_names = ["AnkleAccelerometer_x-axis (g)", "AnkleAccelerometer_y-axis (g)","AnkleAccelerometer_z-axis (g)",
                        "RightPocketAccelerometer_x-axis (g)","RightPocketAccelerometer_y-axis (g)", "RightPocketAccelerometer_z-axis (g)",
                        "BeltAccelerometer_x-axis (g)","BeltAccelerometer_y-axis (g)","BeltAccelerometer_z-axis (g)","NeckAccelerometer_x-axis (g)",
                        "NeckAccelerometer_y-axis (g)","NeckAccelerometer_z-axis (g)","WristAccelerometer_x-axis (g)","WristAccelerometer_y-axis (g)",
                        "WristAccelerometer_z-axis (g)"]
    # print(len(sensor_col_names))
    data = _data[[col_name for col_name in _data.columns if col_name not in sensor_col_names]]
    # print(data.shape)
    sensor_data = _data[sensor_col_names]
    print(sensor_data.shape)
    # Label Smoothing.
    esp = 0.1
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

    # Separate continuous frames.
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        _sensor = sensor_data.iloc[fs].values
        # print(f"DEBUG {_sensor.shape}")
        
        
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3)
        print(f"DEBUG {xys.shape} {_sensor.shape}")
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point.
        xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

        # Weighting main parts score.
        scr = xys[:, :, -1].copy()
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score.
        scr = scr.mean(1)

        # Targets.
        lb = data.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels.
        lb = lb * scr[:, None]

        for i in range(xys.shape[0] - n_frames):
            # print(f"DEBUG {_sensor[i:i+n_frames][None, ...].shape}")
            video_name_set.append(vid)
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)
            sensor_set = np.append(sensor_set, _sensor[i:i+n_frames][None, ...], axis=0)
            #print(lb[i:i+n_frames].mean(0)[None, ...])            

with open(save_path, 'wb') as f:
    pickle.dump((video_name_set,feature_set, sensor_set, labels_set), f)
