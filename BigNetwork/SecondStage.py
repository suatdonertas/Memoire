# Libraries #
import sys 
import glob
import os
import re
import argparse
import math
import gc
import numpy as np
import ROOT
from ROOT import TFile, TTree
from root_numpy import tree2array, array2root
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error as mse

import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model, model_from_json, load_model
from keras import losses, optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint, History, Callback
from keras.models import model_from_json
from keras.regularizers import l1,l2
import keras.backend as K

import matplotlib.pyplot as plt

# Personal libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
from NNOutput import *


###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Building second stage of Frankeinstein network')
parser.add_argument('-l','--label', help='Label for naming model -> COMPULSORY',required=True,type=str)
parser.add_argument('-i','--input', help='Input from first stage -> COMPULSORY',required=True,type=str)
args = parser.parse_args()

path_out = '/home/ucl/cp3/fbury/storage/BigNetworkOutput/'+args.label+'/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

###############################################################################
# Import outputs from first stage #
###############################################################################
path_in = '/home/ucl/cp3/fbury/storage/BigNetwork_FistStage_Output/'+args.input+'/'

f = ROOT.TFile.Open(path_in+'input.root')
t = f.Get("tree")

inputs = tree2array(t,branches=['out_MP','dis_MEM','m_lljj','m_jj','mA','mH','target','weight','MEM_TT','MEM_DY'])

###############################################################################
# Build network of second stage #
###############################################################################
print ('='*80)
print ('[INFO] Building new network')
# Input preparation #

input_full = np.c_[inputs['out_MP'],inputs['dis_MEM'],inputs['MEM_TT'],inputs['MEM_DY'],inputs['mH'],inputs['mA'],inputs['target'],inputs['weight'],inputs['m_lljj'],inputs['m_jj']]
mHmA =  np.unique(np.c_[inputs['mH'],inputs['mA']],axis=0)


# Set separation #
mask = np.genfromtxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/mask.txt')
# True => Training test, False => Evaluation test

DNN_set = input_full[mask==True,:]
eval_set = input_full[mask==False,:]
print ('Training set : ',DNN_set.shape[0])
print ('Evaluation set : ',eval_set.shape[0])


# Preprocessing parametric masses #
min_max_scaler = preprocessing.MinMaxScaler()
DNN_set[:,:6] = min_max_scaler.fit_transform(DNN_set[:,:6])
#np.savetxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/preprocessing.txt',np.c_[min_max_scaler.scale_,min_max_scaler.min_])

################################################################################
# Cross Validation #
################################################################################
k = 10 # Cross validation steps
for i in range(1,k+1):
    print ('-'*80)
    print ('[INFO] Cross validation step : ',i,'/',k)
    # Train/Test set separation #
    train_set, test_set, train_target, test_target, train_weight, test_weight = train_test_split(DNN_set[:,:6],DNN_set[:,6],DNN_set[:,7], shuffle=True, train_size=0.7)
    # Building layers #
    inputs = Input(shape=(train_set.shape[1],))
    L1 = Dense(30,activation='relu',activity_regularizer=l2(0.01))(inputs)
    #L1 = Dense(30,activation='relu')(inputs)
    #L1 = Dropout(0.2)(L1)
    L2 = Dense(30,activation='relu',activity_regularizer=l2(0.01))(L1)
    #L2 = Dense(30,activation='relu')(L1)
    #L2 = Dropout(0.2)(L2)
    L3 = Dense(30,activation='relu',activity_regularizer=l2(0.01))(L2)
    #L3 = Dense(30,activation='relu')(L2)
    #L3 = Dropout(0.2)(L3)
    Dx = Dense(1, activation='sigmoid')(L3)

    DNN = Model(inputs=[inputs], outputs=[Dx])

    utils.print_summary(model=DNN)

    # Optimizer #
    learning_rate = 0.001
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    # Compile #
    DNN.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

    # Callbacks #
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
    csv_logger = CSVLogger(path_out+'training_'+str(i)+'.log')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='min', epsilon=0.00001, cooldown=3, min_lr=0.0001)
    checkpoint = ModelCheckpoint(path_out+'weight_'+str(i)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

    Callback_list = [csv_logger,checkpoint,early_stopping,reduceLR]

    # Fit #
    epoch = 500
    batch = 1000

    DNN.fit(train_set, train_target, sample_weight=train_weight, epochs=epoch, batch_size=batch, verbose=2, validation_data=(test_set,test_target,test_weight), callbacks=Callback_list)

    # Save model #
    model_json = DNN.to_json()
    with open(path_out+'model_'+str(i)+'.json', "w") as json_file:
        json_file.write(model_json)
    # The weights are saved via the checkpoint callback
    print ('[INFO] Model saved as '+path_out+'model_'+str(i)+'.json')

################################################################################
# Validation set output #
################################################################################
print ('='*80)
print ('[INFO] Model evaluation')

# Output of validation set #
eval_set_pre = min_max_scaler.transform(eval_set[:,:6])
#NN_output = DNN.predict(eval_set_pre)
NN_output = NNOutput(eval_set_pre,path_out)
model_output = np.c_[NN_output,eval_set]
# [NN_output,MassPlane_output,MEM_discriminant,mA,mH,target,weight]
print ('MSE : ',mse(NN_output,eval_set[:,6]))

model_output.dtype = [('output','float64'),('MP_discriminant','float64'),('MEM_discriminant','float64'),('MEM_TT','float64'),('MEM_DY','float64'),('mH','float64'),('mA','float64'),('target','float64'),('weight','float64'),('m_lljj','float64'),('m_jj','float64')]

array2root(model_output,path_out+'output_'+str(i)+'.root',mode='recreate')

print ('[INFO] Output saved as '+path_out+'output.root')




