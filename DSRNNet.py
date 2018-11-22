'using  sliced rnn perform RSNet'

import pandas as pd
import numpy as np

'''from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense'''


class DRNN(object):
    """Dilated rnn"""
    def __init__(self, arg):
        super(DRNN, self).__init__()
        self.arg = arg
        
#load data
'df = pd.read_csv("yelp_2013.csv")'
#df = df.sample(5000)

Y = df.stars.values-1
Y = to_categorical(Y,num_classes=5)
X = df.text.values

#set hyper parameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
NUM_FILTERS = 50
MAX_LEN = 512
Batch_size = 100
EPOCHS = 10

#shuffle the data


#training set, validation set and testing set
nb_validation_samples_val = int((VALIDATION_SPLIT + TEST_SPLIT) * X.shape[0])
nb_validation_samples_test = int(TEST_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val =  X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val =  Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

#use tokenizer to build vocab


#pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)

#slice sequences into many subsequences
x_test_padded_seqs_split=[]
for i in range(x_test_padded_seqs.shape[0]):
    split1=np.split(x_test_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_test_padded_seqs_split.append(a)

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
embed = embedding_layer(input1)
gru1 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8,MAX_LEN/64,), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed2)
Encoder2 = Model(input2,gru2)

input3 = Input(shape=(8,8,MAX_LEN/64), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed3)
preds = Dense(5, activation='softmax')(gru3)
model = Model(input3, preds)

print Encoder1.summary()
print Encoder2.summary()
print model.summary()

#use adam optimizer
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#save the best model on validation set
from keras.callbacks import ModelCheckpoint             
savebestmodel = 'save_model/SRNN(8,2)_yelp2013.h5'
checkpoint = ModelCheckpoint(savebestmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks=[checkpoint] 
             
model.fit(np.array(x_train_padded_seqs_split), y_train, 
          validation_data = (np.array(x_val_padded_seqs_split), y_val),
          nb_epoch = EPOCHS, 
          batch_size = Batch_size,
          callbacks = callbacks,
          verbose = 1)

#use the best model to evaluate on test set
from keras.models import load_model
best_model= load_model(savebestmodel)          
print best_model.evaluate(np.array(x_test_padded_seqs_split),y_test,batch_size=Batch_size)


#use adam optimizer
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#save the best model on validation set
from keras.callbacks import ModelCheckpoint             
savebestmodel = 'save_model/SRNN(8,2)_yelp2013.h5'
checkpoint = ModelCheckpoint(savebestmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks=[checkpoint] 
             
model.fit(np.array(x_train_padded_seqs_split), y_train, 
          validation_data = (np.array(x_val_padded_seqs_split), y_val),
          nb_epoch = EPOCHS, 
          batch_size = Batch_size,
          callbacks = callbacks,
          verbose = 1)

#use the best model to evaluate on test set
from keras.models import load_model
best_model= load_model(savebestmodel)          
print best_model.evaluate(np.array(x_test_padded_seqs_split),y_test,batch_size=Batch_size)
