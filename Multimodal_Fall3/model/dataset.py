import torch
from torch.utils.data import Dataset

class Skeleton_Sensor_Dataset(Dataset):
    def __init__(self,skeleton_sensor_data,label):
        self.skeleton_sensor_data = skeleton_sensor_data
        self.label = label
        
    def __len__(self):
        return len(self.skeleton_sensor_data)
    
    def __getitem__(self,idx):
        skeleton,sensor = self.skeleton_sensor_data[idx]
        label = self.label[idx]
        
        # print(f"DEBUG {type(skeleton.dtype)}, {type(sensor.dtype)} {type(label.dtype)} ")
        
        if not isinstance(skeleton,torch.Tensor):
            skeleton = torch.tensor(skeleton,dtype=torch.float32)
            
        if not isinstance(sensor,torch.Tensor):
            sensor = torch.tensor(sensor,dtype=torch.float32)
        
        if not isinstance(label,torch.Tensor):
            label = torch.tensor(label,dtype=torch.float32)
            
        skeleton = skeleton.permute(2,0,1)
        return skeleton,sensor,label
    
class Skeleton_Sensor_Dataset_v2(Dataset):
    def __init__(self,skeleton_sensor_data,label):
        self.skeleton_sensor_data = skeleton_sensor_data
        self.label = label
        
    def __len__(self):
        return len(self.skeleton_sensor_data)
    
    def __getitem__(self,idx):
        skeleton,sensor = self.skeleton_sensor_data[idx]
        label = self.label[idx]
        
        # print(f"DEBUG {type(skeleton.dtype)}, {type(sensor.dtype)} {type(label.dtype)} ")
        
        if not isinstance(skeleton,torch.Tensor):
            skeleton = torch.tensor(skeleton,dtype=torch.float32)
            
        if not isinstance(sensor,torch.Tensor):
            sensor = torch.tensor(sensor,dtype=torch.float32)
        
        if not isinstance(label,torch.Tensor):
            label = torch.tensor(label,dtype=torch.float32)
            
        skeleton = skeleton.permute(2,0,1) # skeleton [num_sample, (time, vertex, xyz)] -> [num_sample, (xyz, time, vertex)]
        return skeleton,sensor,label
    