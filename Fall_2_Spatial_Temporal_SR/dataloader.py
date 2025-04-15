import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import (
    train_test_split,
)

from dataset import (
    Skeleton_Sensor_Dataset
)

def build_dataloader(config):
    dataset_name = config.DATA.DATASET

    config.defrost()
    if dataset_name == "harup":
        dataloaders, config.DATA.NUM_CLASSES = _build_harup_dataloader(config.DATA.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY, config.SEED)
    elif dataset_name == "harup_egawa":
        dataloaders, config.DATA.NUM_CLASSES = _build_harup_dataloader_egawa(config.DATA.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY, config.SEED)
    elif dataset_name == "urfall":
        dataloaders, config.DATA.NUM_CLASSES = _build_urfall_dataloader(config.DATA.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY, config.SEED)
        # dataloaders, config.DATA.NUM_CLASSES = _build_urfall_dataloader_egawa(config.DATA.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY, config.SEED)
        
                            
    else:
        raise RuntimeError(f"Dataset [{dataset_name}] is not implemented.")
        
    config.freeze()
    return dataloaders


def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def _build_urfall_dataloader(batch_size:int=16, num_workers:int=8, pin_memory:bool=True, random_seed:int=42):
    data_files = [
        '../datasets/urfall/urfall_30.pkl'
        ]

    class_names = ['unfall', 'fall']

    #######################################################
    videos = []
    features,sensors, labels = [], [], []
    for fil in tqdm(data_files):
        with open(fil, 'rb') as f:
            vid, fts, sr, lbs = pickle.load(f)
            videos += vid
            features.append(fts)
            sensors.append(sr)
            labels.append(lbs)
        del fts, lbs, sr
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensors = np.concatenate(sensors, axis=0)
    labels = labels.astype(np.float32) # dtype : object -> float32
    ######################################################
    
    unique_video_names = np.unique(videos)
    train_video_names, other_video_names = train_test_split(unique_video_names, test_size=0.4, shuffle=True, random_state=random_seed)
    valid_video_names, test_video_names = train_test_split(other_video_names, test_size=0.5, shuffle=True, random_state=random_seed)

    train_samples, valid_samples, test_samples = [],[],[]
    train_label, valid_label, test_label = [],[],[]
    for video,feature,sensor,label in tqdm(zip(videos,features,sensors,labels),total=len(videos)):
        if video in train_video_names:
            train_samples += [(feature,sensor)]
            train_label += [label]
        elif video in valid_video_names:
            valid_samples += [(feature,sensor)]
            valid_label += [label]
        else:
            test_samples += [(feature, sensor)]
            test_label += [label]

    datasets = {
        "train":Skeleton_Sensor_Dataset(train_samples,train_label),
        "valid":Skeleton_Sensor_Dataset(valid_samples,valid_label),
        "test":Skeleton_Sensor_Dataset(test_samples,test_label),
    }
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    dataloaders = dict()
    for key in datasets.keys():
        dataloaders[key] = DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=True if key=="train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if key=="train" else False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    print("="*50)
    for key in dataloaders.keys():
        print(f"{key} : {len(dataloaders[key].dataset)}")
    print("="*50)
    return dataloaders, labels.shape[-1]


def _build_urfall_dataloader_egawa(batch_size:int=16, num_workers:int=8, pin_memory:bool=True, random_seed:int=42):
    data_files = [
        '../datasets/urfall/urfall_30.pkl'
        ]

    class_names = ['unfall', 'fall']

    #######################################################
    videos = []
    features,sensors, labels = [], [], []
    for fil in tqdm(data_files):
        with open(fil, 'rb') as f:
            vid, fts, sr, lbs = pickle.load(f)
            videos += vid
            features.append(fts)
            sensors.append(sr)
            labels.append(lbs)
        del fts, lbs, sr
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensors = np.concatenate(sensors, axis=0)
    labels = labels.astype(np.float32) # dtype : object -> float32
    ######################################################
    
    unique_video_names = np.unique(videos)
    train_video_names, other_video_names = train_test_split(unique_video_names, test_size=0.4, shuffle=True, random_state=random_seed)
    # valid_video_names, test_video_names = train_test_split(other_video_names, test_size=0.5, shuffle=True, random_state=random_seed)

    train_samples, valid_samples, test_samples = [],[],[]
    train_label, valid_label, test_label = [],[],[]
    for video,feature,sensor,label in tqdm(zip(videos,features,sensors,labels),total=len(videos)):
        if video in train_video_names:
            train_samples += [(feature,sensor)]
            train_label += [label]
        else:
            valid_samples += [(feature,sensor)]
            valid_label += [label]

    datasets = {
        "train":Skeleton_Sensor_Dataset(train_samples,train_label),
        "valid":Skeleton_Sensor_Dataset(valid_samples,valid_label),
        "test":Skeleton_Sensor_Dataset(valid_samples,valid_label),
    }
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    dataloaders = dict()
    for key in datasets.keys():
        dataloaders[key] = DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=True if key=="train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if key=="train" else False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    print("="*50)
    for key in dataloaders.keys():
        print(f"{key} : {len(dataloaders[key].dataset)}")
    print("="*50)
    return dataloaders, labels.shape[-1]

def _build_harup_dataloader(batch_size:int=16, num_workers:int=8, pin_memory:bool=True, random_seed:int=42):
    data_files = [
        '../datasets/harup/har30_1_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_2_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_3_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_4_sensor_new-set(labelXscrw).pkl',
        # '../datasets/harup/har30_5_sensor_new-set(labelXscrw).pkl',]
        '../datasets/harup/har30_6_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_7_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_8_sensor_new-set(labelXscrw).pkl',
        # '../datasets/harup/har30_9_sensor_new-set(labelXscrw).pkl',]
        '../datasets/harup/har30_10_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_11_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_12_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_13_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_14_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_15_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_16_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_17_sensor_new-set(labelXscrw).pkl'
        ]

    class_names = ['Falling_forwards_hands','Falling_forwards_knees','Falling_backwards','Falling_sidewards','Falling_sitting','Walking','Standing','Sitting','Picking','Jumping','Laying']

    #######################################################
    videos = []
    features,sensors, labels = [], [], []
    for fil in tqdm(data_files):
        with open(fil, 'rb') as f:
            vid, fts, sr, lbs = pickle.load(f)
            videos += vid
            features.append(fts)
            sensors.append(sr)
            labels.append(lbs)
        del fts, lbs, sr
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensors = np.concatenate(sensors, axis=0)
    labels = labels.astype(np.float32) # dtype : object -> float32
    ######################################################
    
    unique_video_names = np.unique(videos)
    train_video_names, other_video_names = train_test_split(unique_video_names, test_size=0.4, shuffle=True, random_state=random_seed)
    valid_video_names, test_video_names = train_test_split(other_video_names, test_size=0.5, shuffle=True, random_state=random_seed)

    train_samples, valid_samples, test_samples = [],[],[]
    train_label, valid_label, test_label = [],[],[]
    for video,feature,sensor,label in tqdm(zip(videos,features,sensors,labels),total=len(videos)):
        if video in train_video_names:
            train_samples += [(feature,sensor)]
            train_label += [label]
        elif video in valid_video_names:
            valid_samples += [(feature,sensor)]
            valid_label += [label]
        else:
            test_samples += [(feature, sensor)]
            test_label += [label]

    datasets = {
        "train":Skeleton_Sensor_Dataset(train_samples,train_label),
        "valid":Skeleton_Sensor_Dataset(valid_samples,valid_label),
        "test":Skeleton_Sensor_Dataset(test_samples,test_label),
    }
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    dataloaders = dict()
    for key in datasets.keys():
        dataloaders[key] = DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=True if key=="train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if key=="train" else False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    print("="*50)
    for key in dataloaders.keys():
        print(f"{key} : {len(dataloaders[key].dataset)}")
    print("="*50)
    return dataloaders, labels.shape[-1]


def _build_harup_dataloader_egawa(batch_size:int=16, num_workers:int=8, pin_memory:bool=True, random_seed:int=42):
    data_files = [
        '../datasets/harup/har30_1_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_2_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_3_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_4_sensor_new-set(labelXscrw).pkl',
        # '../datasets/harup/har30_5_sensor_new-set(labelXscrw).pkl',]
        '../datasets/harup/har30_6_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_7_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_8_sensor_new-set(labelXscrw).pkl',
        # '../datasets/harup/har30_9_sensor_new-set(labelXscrw).pkl',]
        '../datasets/harup/har30_10_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_11_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_12_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_13_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_14_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_15_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_16_sensor_new-set(labelXscrw).pkl',
        '../datasets/harup/har30_17_sensor_new-set(labelXscrw).pkl'
        ]

    class_names = ['Falling_forwards_hands','Falling_forwards_knees','Falling_backwards','Falling_sidewards','Falling_sitting','Walking','Standing','Sitting','Picking','Jumping','Laying']

    #######################################################
    videos = []
    features,sensors, labels = [], [], []
    for fil in tqdm(data_files):
        with open(fil, 'rb') as f:
            vid, fts, sr, lbs = pickle.load(f)
            videos+= vid
            features.append(fts)
            sensors.append(sr)
            labels.append(lbs)
        del fts, lbs, sr
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensors = np.concatenate(sensors, axis=0)
    labels = labels.astype(np.float32) # dtype : object -> float32
    ######################################################
    
    unique_video_names = np.unique(videos)
    train_video_names, other_video_names = train_test_split(unique_video_names, test_size=0.1, shuffle=True, random_state=random_seed)
    # valid_video_names, test_video_names = train_test_split(other_video_names, test_size=0.5, shuffle=True, random_state=random_seed)

    train_samples, valid_samples, test_samples = [],[],[]
    train_label, valid_label, test_label = [],[],[]
    for video,feature,sensor,label in tqdm(zip(videos,features,sensors,labels),total=len(videos)):
        if video in train_video_names:
            train_samples += [(feature,sensor)]
            train_label += [label]
        # elif video in valid_video_names:
        #     valid_samples += [(feature,sensor)]
        #     valid_label += [label]
        else:
            test_samples += [(feature, sensor)]
            test_label += [label]

    datasets = {
        "train":Skeleton_Sensor_Dataset(train_samples,train_label),
        "valid":Skeleton_Sensor_Dataset(test_samples,test_label),
        "test":Skeleton_Sensor_Dataset(test_samples,test_label),
    }
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    dataloaders = dict()
    for key in datasets.keys():
        dataloaders[key] = DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=True if key=="train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if key=="train" else False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
    print("="*50)
    for key in dataloaders.keys():
        print(f"{key} : {len(dataloaders[key].dataset)}")
    print("="*50)
    return dataloaders, labels.shape[-1]

if __name__=="__main__":
    dataloaders, num_classes = _build_urfall_dataloader()
    print(num_classes)
    for skele, sensor, label in dataloaders["train"]:
        print(skele.size(),sensor.size(),label.size())
    
