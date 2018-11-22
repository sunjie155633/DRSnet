import numpy as np
import torch
import torch.nn as nn

class 

self.rnn_type = 'GRU'

        
        self.rnn_x_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
top = []
for i in range (4)
    middle = []
    for j in range (4)
        low = GRU()        #first layer into RNN
        middle.append(low)
    middle = GRU (middle)   #second layer into RNN
    top.append(middle)
output = GRU (top)         #top layer into RNN
