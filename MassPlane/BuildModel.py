# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import h5py
#import matplotlib.pyplot as plt

import ROOT
from ROOT import TFile, TTree, TCanvas
from root_numpy import tree2array

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc 

from keras import utils
from keras.layers import Input, Dense
from keras.models import Model
from keras import losses
from keras.callbacks import EarlyStopping

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Takes root files, build Neural Network and save it') 
parser.add_argument("-a","--architecture", help="Name of the text file containing all the requested architectures -> COMPULSORY")
parser.add_argument("-r","--results", help="Name of the text file  where the results must be written -> COMPULSORY")

args = parser.parse_args()
print ('Architectures from ',str(args.architecture))
print ('Results to be output in ',str(args.results))
        
############################################################################### 
# Extract features from Root Files #
############################################################################### 
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
print ('='*80)
print ('[INFO] Starting input from files')
back_set = np.zeros((0,2))
sig_set = np.zeros((0,4))

for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('\t-> Signal')
        Sig = True #Signal case
    else: # Background
        print ('\t-> Background')
        Sig = False #Background case

    f = ROOT.TFile.Open(name)
    t = f.Get("t")
    N = t.GetEntries()
    
    jj_M = np.asarray(tree2array(t, branches='jj_M'))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M'))
    if Sig: #Signal
        num = [int(s) for s in re.findall('\d+',filename )]
        print ('\tmH = ',num[2],', mA = ',num[3])
        mH = np.ones(N)*num[2]
        mA = np.ones(N)*num[3]
        sig_data = np.stack((lljj_M,jj_M,mH,mA),axis=1)
        sig_set = np.concatenate((sig_data,sig_set),axis=0) 
        print ('\t-> Size = %i,\ttotal signal size = %i' %(sig_data.shape[0],sig_set.shape[0]))

    else : # Background
        back_data = np.stack((lljj_M,jj_M),axis=1)
        back_set = np.concatenate((back_data,back_set),axis=0)
        print ('\t-> Size = %i,\ttotal background size = %i' %(back_data.shape[0],back_set.shape[0]))

print ('\n\nTotal signal size = ',sig_set.shape[0])
print ('Total background size = ',back_set.shape[0])


############################################################################### 
# Background random mH,mA assignation + target association #
############################################################################### 
print ('='*80)
print ('[INFO] Starting target association')
np.random.shuffle(back_set) # Mix the different background contributions

sig_target = np.ones(sig_set.shape[0])
back_target = np.zeros(sig_set.shape[0])

back_select = back_set[:sig_target.shape[0],:]
back_select = np.c_[back_select,sig_set[:,2:4]] # adds mH,mA to background in same proportions as signal

sig_set = np.c_[sig_set,sig_target]
back_select = np.c_[back_select,back_target]

print ('Signal size = ',sig_set.shape[0])
print ('Background size = ',back_select.shape[0])

data = np.concatenate((sig_set,back_select),axis=0)
print ('Total learning size = ',data.shape[0])
np.random.shuffle(data)

############################################################################### 
# Learning Neural network #
############################################################################### 
print ('='*80)
print ('[INFO] Starting learning')

N = data.shape[0]
nf = data.shape[1]-1 # not including targets
X = data[:,:nf]
T = data[:,-1]

# Splitting #
X_train, X_test, T_train, T_test = train_test_split(X,T,test_size=0.25,shuffle=False)

# Build layers #
layers = np.array([100,100,100,100])
inputs = Input(shape=(X_train.shape[1],))
Dx = Dense(layers[0], activation="tanh")(inputs)
Dx = Dense(layers[1], activation="relu")(Dx)
Dx = Dense(layers[2], activation="relu")(Dx)
Dx = Dense(layers[3], activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)

# Compile  #
DNN = Model(inputs=[inputs], outputs=[Dx])
DNN.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# Callback #
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='min')
Callback_list = [early_stopping]

# Fit #
history = DNN.fit(X_train, T_train, epochs=100, batch_size=32, verbose=2, validation_data=(X_test,T_test), callbacks=Callback_list)

############################################################################### 
# Plot Section #
############################################################################### 
print ('='*80)
print ("[INFO] Starting the plot section")
print(history.history.keys())

# Set up directory for the plots and models#
archi_name = np.array2string(layers)
path_plots = '/home/ucl/cp3/fbury/Memoire/loss_plots/'+archi_name+'/'
path_model = '/home/ucl/cp3/fbury/Memoire/model_saved/'+archi_name+'/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)
if not os.path.exists(path_model):
    os.makedirs(path_model)

# Plot loss and accuracy #


# summarize history for accuracy 
#fig1, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=True)
#ax1.plot(history.history['acc'])
#ax1.plot(history.history['val_acc'])
#ax1.set_title('Model accuracy')
#ax1.set_ylabel('Accuracy')
#ax1.legend(['train', 'test'], loc='lower right')

# summarize history for loss #
#ax2.plot(history.history['loss'])
#ax2.plot(history.history['val_loss'])
#ax2.set_title('Model loss')
#ax2.set_ylabel('Loss')
#ax2.set_xlabel('Epoch')
#ax2.legend(['train', 'test'], loc='upper right')
#plt.show()

#fig1.savefig(path_plots+'loss.png', bbox_inches='tight')

# Save the model #
model_json = DNN.to_json()
with open(path_model+'model.json', "w") as json_file:
    json_file.write(model_json)
DNN.save_weights(path_model+'model.h5')
print("[INFO] Saved model to disk")


