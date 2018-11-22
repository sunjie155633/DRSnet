#slice sequences into many subsequences
#对于每一层，先分层，然后搭一个SRNN，接口同GRU,直接调用

import numpy as np
print(1)

x_split=[]
for i in range(x.shape[0]):
    split1=np.split(x[i],8) #取前8个split1
    a=[] 
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_split.append(a)

x_val_padded_seqs_split=[]
for i in range(x_val_padded_seqs.shape[0]):
    split1=np.split(x_val_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_val_padded_seqs_split.append(a)
   
x_train_padded_seqs_split=[]
for i in range(x_train_padded_seqs.shape[0]):
    split1=np.split(x_train_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_train_padded_seqs_split.append(a)



#build model
print "Build Model"
input1 = Input(shape=(MAX_LEN/64,), dtype='int32')
gru1 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(input1)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8,MAX_LEN/64,), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed2)
Encoder2 = Model(input2,gru2)

input3 = Input(shape=(8,8,MAX_LEN/64), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed3)
preds = Dense(5, activation='softmax')(gru3)
model = Model(input3, preds


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y