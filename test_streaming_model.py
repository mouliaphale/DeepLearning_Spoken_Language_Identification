
import librosa
import os
import numpy as np
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model,Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

dirName = sys.argv[0]
listOfFiles = getListOfFiles(dirName)

listOfFiles.sort()

train_seq_length = 10
feature_dim = 64
Demo = True


y_test = []
mat_test = []
count = 0
for f in listOfFiles:
  if Demo:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n !=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_test.append(y)
    mat_test.append(mat2)
    count+=1


model = 'streaming_model.hdf5'
streaming_model = tensorflow.keras.load_model(model, custom_objects = None, compile =True)

print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
for s in range(num_seqs):
    print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
    in_seq = x[s].reshape( (1, train_seq_length, feature_dim) )
    seq_pred = streaming_model.predict(in_seq)
    seq_pred = seq_pred.reshape(train_seq_length)
    for n in range(train_seq_length):
        in_feature_vector = x[s][n].reshape(1,1,feature_dim)
        single_pred = streaming_model.predict(in_feature_vector)[0][0]
        print(f'Seq-model Prediction, Streaming-Model Prediction, difference [{n}]: {seq_pred[n] : 3.2f}, {single_pred : 3.2f}, {seq_pred[n] - single_pred: 3.2f}')
    streaming_model.reset_states()
