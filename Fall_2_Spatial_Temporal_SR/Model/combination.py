import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_gcn.stgcan import STGCAN
from .st_gcn.graph import Graph
from .bilstm import BiLSTM

class TwoStreamSTGCAN(nn.Module):
    def __init__(self,in_channels, graph_args, num_class):
        super().__init__()

        self.stgcan_1 = STGCAN(3, graph_args, num_class=None) #output (B, 256)
        self.stgcan_2 = STGCAN(2, graph_args, num_class=None) #output (B, 256)

        self.fc = nn.Linear(256*2 , num_class)

    def forward(self,skel,sensor):
        mot = skel[:,:2,1:] - skel[:,:2,:-1]

        pts = self.stgcan_1(skel)
        mot = self.stgcan_2(mot)

        x = torch.cat((pts,mot),dim=-1)
        return self.fc(x)

class TwoStreamSTGCAN_BiLSTM(nn.Module):
    def __init__(self,in_channels, graph_args, num_class, bilstm_input_size=15):
        super().__init__()

        self.stgcan_1 = STGCAN(3, graph_args, num_class=None) #output (B, 256)
        self.stgcan_2 = STGCAN(2, graph_args, num_class=None) #output (B, 256)
        self.lstm = BiLSTM(input_size=bilstm_input_size,hidden_size=64,num_layers=1,dropout_prob=0.3,num_classes=num_class,feature="mean")

        self.fc = nn.Linear(256*2+num_class , num_class)

    def forward(self,skel,sensor):
        # print(f"DEBUG {skel.size(),sensor.size()}")
        mot = skel[:,:2,1:] - skel[:,:2,:-1]

        pts = self.stgcan_1(skel,None)
        mot = self.stgcan_2(mot,None)
        sensor = self.lstm(None,sensor)

        x = torch.cat((pts,mot,sensor),dim=-1)
        return self.fc(x)