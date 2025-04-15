# _*_ coding: utf-8 _*_
# @time     :2021/12/30 12:08
# @Author   :jc
# @File     :TA.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

# device=torch.device('cuda')
# device=torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using", device, "device")

class Transform(nn.Module):
    def __init__(self, outfea, d, seq_len=30):
        super(Transform, self).__init__()
        self.vff = nn.Linear(outfea, outfea)
        self.conv1=nn.Conv2d(seq_len,seq_len,(1,3),bias=True)
        self.conv2=nn.Conv2d(seq_len,seq_len,(1,3),bias=True)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x,score_his=None):# x : b t n hidden
        b, t, n, c = x.shape
        query=self.conv1(x)
        # print(query.shape)
        key=self.conv2(x)
        value=self.vff(x)
        query = query.permute(0, 2, 1, 3)
        # print(query.shape)
        key = key.permute(0, 2, 3, 1)
        # print(key.shape)
        value = value.permute(0, 2, 1, 3)




        A = torch.matmul(query, key)
        # print("A:",A.shape)
        A /= (c ** 0.5)

        # print(score_his.shape)

        A = torch.softmax(A, -1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=30):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe,requires_grad=False)
        return x

class transformer_layer(nn.Module):
    def __init__(self,dim_in,dim_out,num_layer,d=2,att_his=False):
        super(transformer_layer,self).__init__()
        # self.linear1=nn.Linear(dim_in,dim_out)
        self.trans_layers=nn.ModuleList(Transform(dim_out,d) for l in range(num_layer))
        self.PE=PositionalEncoding(dim_out)
        self.num_layer=num_layer
        self.att_his=att_his
        self.score_his = torch.zeros((64, 170, 12, 12), requires_grad=False).to(device)
    def forward(self, x):
        x=self.PE(x)
        for l in range(self.num_layer):
            if  self.att_his:
                x=self.trans_layers[l](x,self.score_his)
            else:
                x=self.trans_layers[l](x)
        return x






if __name__=="__main__":
    x = torch.randn(32, 12, 170, 64)
    dim_in=64
    dim_out=64
    num_layer=2
    d=64
    transformer=transformer_layer(dim_in,dim_out,num_layer,d)
    res=transformer(x)
    print(res.shape)