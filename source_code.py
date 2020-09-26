
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

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

#os.chdir('/content/drive/My Drive/train')
dirName_english = '/home/ubuntu/train/train_english';
dirName_hindi = '/home/ubuntu/train/train_hindi';
dirName_mandarin = '/home/ubuntu/train/train_mandarin';


# Get the list of all files in directory tree at given path
listOfFiles_english = getListOfFiles(dirName_english)
listOfFiles_hindi = getListOfFiles(dirName_hindi)
listOfFiles_mandarin = getListOfFiles(dirName_mandarin)

listOfFiles_english.sort()
listOfFiles_hindi.sort()
listOfFiles_mandarin.sort()

Demo = False
Full = True
train_seq_length = 10
feature_dim = 64

y_english = []
mat_english = []
count = 0
for f in listOfFiles_english:
  if Demo and count<6:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n !=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_english.append(y)
    mat_english.append(mat2)
    count = count+1
    print('demo english',count,'mfcc done')
  if Full:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n !=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_english.append(y)
    mat_english.append(mat2)
    count+=1
print('english done')
#arr_english = np.concatenate(y_english[0:5],axis=0)
y_hindi = []
mat_hindi = []
count = 0
print('count reset')
for f in listOfFiles_hindi:
  if Demo and count<6:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n !=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_hindi.append(y)
    mat_hindi.append(mat2)
    count+=1
    print('demo hindi',count,'mfcc done')
  if Full:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n !=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_hindi.append(y)
    mat_hindi.append(mat2)
    count+=1
print('hindi done')

y_mandarin = []
mat_mandarin = []
count = 0
print('count reset')
for f in listOfFiles_mandarin:
  if Demo and count<6:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n!=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_mandarin.append(y)
    mat_mandarin.append(mat2)
    count+=1
    print('demo mandarin',count,'mfcc done')
  if Full:
    y,sr = librosa.load(f,sr=16000)
    mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=64,n_fft=int(sr*0.025),hop_length=int(sr*0.01))
    n = mat.shape[1]%10
    num_seqs = int(math.floor(mat.shape[1]/10))
    mat1 = np.transpose(mat)
    if n!=0:
      mat1 = mat1[:-n,:]
    mat2 = mat1.reshape(num_seqs, train_seq_length,feature_dim)
    y_mandarin.append(y)
    mat_mandarin.append(mat2)
    count+=1
print('mandarin done')

print('mfcc done')

array_english=np.concatenate(mat_english[0:len(listOfFiles_english)],axis=0)
eng = [1,0,0]
rows, cols = (array_english.shape[0], train_seq_length)
english_labels = [[eng for i in range(cols)] for j in range(rows)]
#english_labels = np.concatenate((arr),axis=0)

array_hindi=np.concatenate(mat_hindi[0:len(listOfFiles_hindi)],axis=0)
hin = [0,1,0]
rows, cols = (array_hindi.shape[0], train_seq_length)
hindi_labels = [[hin for i in range(cols)] for j in range(rows)]
#hindi_labels = np.concatenate((arr),axis=0)

array_mandarin=np.concatenate(mat_mandarin[0:len(listOfFiles_mandarin)],axis=0)
man = [0,0,1]
rows, cols = (array_mandarin.shape[0], train_seq_length)
mandarin_labels = [[man for i in range(cols)] for j in range(rows)]
#andarin_labels = np.concatenate((arr),axis=0)

train_english, val_english, ytrain_english, yval_english  = train_test_split(array_english, english_labels, test_size=0.2)
train_hindi, val_hindi, ytrain_hindi, yval_hindi  = train_test_split(array_hindi, hindi_labels, test_size=0.2)
train_mandarin, val_mandarin, ytrain_mandarin, yval_mandarin = train_test_split(array_mandarin, mandarin_labels, test_size=0.2)
print('train and val split finished')

train_data = np.concatenate((train_english,train_hindi,train_mandarin),axis=0)
X_val = np.concatenate((val_english,val_hindi,val_mandarin),axis=0)

y_train = np.vstack((ytrain_english,ytrain_hindi,ytrain_mandarin))
y_val = np.vstack((yval_english,yval_hindi,yval_mandarin))

print('train shape:',train_data.shape,'y_train shape:', y_train.shape)

validation_data = (X_val,y_val)

print('validation shape:',np.array(X_val).shape,'y_val shape:', np.array(y_val).shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  model = Sequential()
  model.add(Dense(16, activation='relu',input_shape=(train_seq_length,64)))
  model.add(GRU(16, return_sequences=True, stateful=False))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.summary()

plot_model(model, to_file='modelplot.png', show_shapes=True, show_layer_names=True)
model.fit(train_data, y_train, batch_size=16, epochs=5,shuffle=True,class_weight=None,validation_data = validation_data)

print('sequence model fit complete')
model.save('saved_model.hdf5')

model.save_weights('weights.hd5',overwrite = True)

streaming_model = Sequential()
streaming_model.add(Dense(16, activation='relu',batch_input_shape=(1,None,64)))
streaming_model.add(GRU(16, return_sequences=False, stateful=True))
streaming_model.add(Dense(3, activation='softmax'))
streaming_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

streaming_model.summary()

streaming_model.load_weights('weights.hd5')

print('streaming model loaded')
