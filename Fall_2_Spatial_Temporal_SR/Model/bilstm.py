import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, input_size, reduce_rate = 1/8):
        super(ChannelAttention,self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_size, int(input_size*reduce_rate)),
            nn.ReLU(),
            nn.Linear(int(input_size*reduce_rate), input_size),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        weight = self.attention(x)
        x = torch.einsum('bc,bc -> bc',(x,weight))
        return x
        
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, num_classes = 1, feature = "last"):
        super(BiLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # Bidirectional LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,dropout=dropout_prob)
        
        self.batchnorm = nn.BatchNorm1d(hidden_size*2)
        self.channelattention = ChannelAttention(hidden_size*2)
        
        self.feature = feature
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*2,num_classes)
        )
        
    def forward(self, skel, sensor):
        # x = self.attention(x)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, sensor.size(0), self.hidden_size).to(sensor.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, sensor.size(0), self.hidden_size).to(sensor.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm1(sensor, (h0, c0))
        
        
        # out = self.meanovertime(out)
        if self.feature=="last":
            out = out[:, -1, :]
        else:
            out = torch.mean(out,axis=1,keepdim=False)
        out = self.batchnorm(out)
        out = self.channelattention(out)
        out = self.fc(out) #はずしてみる
        return out