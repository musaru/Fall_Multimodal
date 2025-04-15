import os
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
import random
import multiprocessing

from sklearn.model_selection import (
    train_test_split,
    KFold,
)
import torch
from torch.utils.data import DataLoader
from dataset import (
    GeneralDataset,
    Fall2Dataset,
)


def build_urfall_dataloader(split_ratio:Dict[str,float]={'train':0.6,'valid':0.2,'test':0.2}, batch_size:int=32, seq_len:int=30, random_seed:int=42, num_workers:int=4, pin_memory:bool=True):
    dataset_dir_path = "../datasets/urfall"
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    

    csv_path_list = [os.path.join(root,file) for root,dirs,files in os.walk(dataset_dir_path) for file in files if file.endswith(".csv")]

    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_path_list],axis=0)
    label_df = pd.get_dummies(df["label"])

    df = pd.concat([df.drop(["label"],axis=1),label_df],axis=1)

    samples = []

    unique_video_name = np.unique(df["video"])
    for video_name in tqdm(unique_video_name):
        video_df = df[df["video"]==video_name]

        row, _ = video_df.shape

        skeleton = video_df.drop(["video","frame"]+label_df.columns.values.tolist(),axis=1)
        label = video_df[label_df.columns].values

        for i in range(0, row):
            _skeleton = skeleton.iloc[i:i+seq_len]
            _label = label[i:i+seq_len]

            if _skeleton.isnull().values.sum() != 0 or _skeleton.shape[0] < seq_len:
                continue
            _skeleton = _skeleton.values.reshape((seq_len,-1,3))
            _label = np.mean(_label,axis=0)
            
            samples.append((_skeleton,_label))
            

    train_samples, other = train_test_split(samples,train_size=split_ratio["train"],random_state=random_seed, shuffle=True)
    if split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]) < 1.0:
        valid_samples, test_samples = train_test_split(other, train_size=split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]),random_state=random_seed, shuffle=True)
    else:
        valid_samples, test_samples = other, []

    samples = {"train":train_samples, "valid":valid_samples, "test":test_samples}
    dataloaders = dict()
    for key in samples.keys():
        dataset = Fall2Dataset(samples[key])
        
        dataloaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

    if split_ratio["test"]==0:
        dataloaders["test"] = dataloaders["valid"]

    return dataloaders

def build_imvia_dataloader(split_ratio:Dict[str,float]={'train':0.6,'valid':0.2,'test':0.2}, batch_size:int=32, seq_len:int=30, random_seed:int=42, num_workers:int=4, pin_memory:bool=True):
    dataset_dir_path = "../datasets/imvia"
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    

    csv_path_list = [os.path.join(root,file) for root,dirs,files in os.walk(dataset_dir_path) for file in files if file.endswith(".csv")]

    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_path_list],axis=0)
    label_df = pd.get_dummies(df["label"])

    df = pd.concat([df.drop(["label"],axis=1),label_df],axis=1)

    samples = []

    unique_video_name = np.unique(df["video"])
    for video_name in tqdm(unique_video_name):
        video_df = df[df["video"]==video_name]

        row, _ = video_df.shape

        skeleton = video_df.drop(["video","frame"]+label_df.columns.values.tolist(),axis=1)
        label = video_df[label_df.columns].values

        for i in range(0, row):
            _skeleton = skeleton.iloc[i:i+seq_len]
            _label = label[i:i+seq_len]

            if _skeleton.isnull().values.sum() != 0 or _skeleton.shape[0] < seq_len:
                continue
            _skeleton = _skeleton.values.reshape((seq_len,-1,3))
            _label = np.mean(_label,axis=0)
            
            samples.append((_skeleton,_label))
            
    print(samples[0])
    train_samples, other = train_test_split(samples,train_size=split_ratio["train"],random_state=random_seed, shuffle=True)
    if split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]) < 1.0:
        valid_samples, test_samples = train_test_split(other, train_size=split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]),random_state=random_seed, shuffle=True)
    else:
        valid_samples, test_samples = other, []
    
    
    samples = {"train":train_samples, "valid":valid_samples, "test":test_samples}
    dataloaders = dict()
    for key in samples.keys():
        dataset = Fall2Dataset(samples[key])
        
        dataloaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

    if split_ratio["test"]==0:
        dataloaders["test"] = dataloaders["valid"]

    return dataloaders

def build_hurup_dataloader(split_ratio:Dict[str,float]={'train':0.6,'valid':0.2,'test':0.2}, batch_size:int=32, seq_len:int=30, random_seed:int=42, num_workers:int=4, pin_memory:bool=True):
    dataset_dir_path = "../datasets/harup"
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    

    csv_path_list = [os.path.join(root,file) for root,dirs,files in os.walk(dataset_dir_path) for file in files if file.endswith(".csv")]

    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_path_list],axis=0)
    label_df = pd.get_dummies(df["label"])

    df = pd.concat([df.drop(["label"],axis=1),label_df],axis=1)

    samples = []

    unique_video_name = np.unique(df["video"])
    for video_name in tqdm(unique_video_name):
        video_df = df[df["video"]==video_name]

        row, _ = video_df.shape

        skeleton = video_df.drop(["video","frame"]+label_df.columns.values.tolist(),axis=1)
        label = video_df[label_df.columns].values

        for i in range(0, row):
            _skeleton = skeleton.iloc[i:i+seq_len]
            _label = label[i:i+seq_len]

            if _skeleton.isnull().values.sum() != 0 or _skeleton.shape[0] < seq_len:
                continue
            _skeleton = _skeleton.values.reshape((seq_len,-1,3))
            _label = np.mean(_label,axis=0)
            
            samples.append((_skeleton,_label))
            

    train_samples, other = train_test_split(samples,train_size=split_ratio["train"],random_state=random_seed, shuffle=True)
    if split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]) < 1.0:
        valid_samples, test_samples = train_test_split(other, train_size=split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]),random_state=random_seed, shuffle=True)
    else:
        valid_samples, test_samples = other, []

    samples = {"train":train_samples, "valid":valid_samples, "test":test_samples}
    dataloaders = dict()
    for key in samples.keys():
        dataset = Fall2Dataset(samples[key])
        
        dataloaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

    if split_ratio["test"]==0:
        dataloaders["test"] = dataloaders["valid"]

    return dataloaders


def build_fukinect_dataloader(split_ratio:Dict[str,float]={'train':0.6,'valid':0.2,'test':0.2}, batch_size:int=32, seq_len:int=30, random_seed:int=42, num_workers:int=4, pin_memory:bool=True):
    dataset_dir_path = "../datasets/fukinect"
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    

    csv_path_list = [os.path.join(root,file) for root,dirs,files in os.walk(dataset_dir_path) for file in files if file.endswith(".csv")]

    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_path_list],axis=0)
    label_df = pd.get_dummies(df["label"])

    df = pd.concat([df.drop(["label"],axis=1),label_df],axis=1)

    samples = []

    unique_video_name = np.unique(df["video"])
    for video_name in tqdm(unique_video_name):
        video_df = df[df["video"]==video_name]

        row, _ = video_df.shape

        skeleton = video_df.drop(["video","frame"]+label_df.columns.values.tolist(),axis=1)
        label = video_df[label_df.columns].values

        for i in range(0, row):
            _skeleton = skeleton.iloc[i:i+seq_len]
            _label = label[i:i+seq_len]

            if _skeleton.isnull().values.sum() != 0 or _skeleton.shape[0] < seq_len:
                continue
            _skeleton = _skeleton.values.reshape((seq_len,-1,3))
            _label = np.mean(_label,axis=0)
            
            samples.append((_skeleton,_label))
            

    train_samples, other = train_test_split(samples,train_size=split_ratio["train"],random_state=random_seed, shuffle=True)
    if split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]) < 1.0:
        valid_samples, test_samples = train_test_split(other, train_size=split_ratio["valid"]/(split_ratio["valid"]+split_ratio["test"]),random_state=random_seed, shuffle=True)
    else:
        valid_samples, test_samples = other, []

    samples = {"train":train_samples, "valid":valid_samples, "test":test_samples}
    dataloaders = dict()
    for key in samples.keys():
        dataset = Fall2Dataset(samples[key])
        
        dataloaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

    if split_ratio["test"]==0:
        dataloaders["test"] = dataloaders["valid"]

    return dataloaders


if __name__=="__main__":
    # dataloaders = build_urfall_dataloader()
    # for data,label in dataloaders["train"]:
    #     print(data.size())
    # dataloaders = build_imvia_dataloader()
    # for data,label in dataloaders["train"]:
    #     print(data.size(),label.size())
    # dataloaders = build_hurup_dataloader()
    # for data,label in dataloaders["train"]:
    #     print(data.size(),label.size())
    dataloaders = build_fukinect_dataloader()
    for data,label in tqdm(dataloaders["train"]):
        # print(label.size())
        pass
