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

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Building pre-stage Frankeinstein network')
parser.add_argument('-l','--label', help='Label for naming model -> COMPULSORY',required=True,type=str)
args = parser.parse_args()

path_out = '/home/ucl/cp3/fbury/storage/TrainMass/'+args.label+'/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

###############################################################################
# Import Root from MoMEMta Network #
###############################################################################
print ('='*80)
print ('[INFO] Starting input from files')
INPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output_transform/PtEtaPhiM/'

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)

inputs = np.empty((0,14)) # Pt, eta, phi x4 + met
targets = np.empty((0,3)) # mjj and mlljj
weight =np.empty((0,1))
id =np.empty((0,1))

for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    # Get branches into numpy arrays #
    total_weight = tree2array(t, branches='total_weight').reshape(-1,1)
    event_weight = tree2array(t, branches='event_weight').reshape(-1,1)
    event_weight_sum = f.Get('event_weight_sum').GetVal()
    L = 35922

    lep1_Pt = tree2array(t, branches='lep1_Pt')
    lep1_Eta = tree2array(t, branches='lep1_Eta')
    lep1_Phi = tree2array(t, branches='lep1_Phi')

    lep2_Pt = tree2array(t, branches='lep2_Pt')
    lep2_Eta = tree2array(t, branches='lep2_Eta')
    lep2_Phi = tree2array(t, branches='lep2_Phi')

    jet1_Pt = tree2array(t, branches='jet1_Pt')
    jet1_Eta = tree2array(t, branches='jet1_Eta')
    jet1_Phi = tree2array(t, branches='jet1_Phi')

    jet2_Pt = tree2array(t, branches='jet2_Pt')
    jet2_Eta = tree2array(t, branches='jet2_Eta')
    jet2_Phi = tree2array(t, branches='jet2_Phi')

    met_pt = tree2array(t, branches='met_pt')
    met_phi = tree2array(t, branches='met_phi')

    jj_M  = tree2array(t, branches='jj_M')
    ll_M  = tree2array(t, branches='ll_M')
    lljj_M  = tree2array(t, branches='lljj_M')

    filedata = np.c_[lep1_Pt,lep1_Eta,lep1_Phi,lep2_Pt,lep2_Eta,lep2_Phi,jet1_Pt,jet1_Eta,jet1_Phi,jet2_Pt,jet2_Eta,jet2_Phi,met_pt,met_phi]
    file_targets = np.c_[lljj_M,jj_M,ll_M]

    inputs = np.concatenate((inputs,filedata),axis=0)
    targets = np.concatenate((targets,file_targets),axis=0)
    

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('\t-> Signal')
        weight = np.concatenate((weight,total_weight/np.sum(total_weight)),axis=0)
        id = np.concatenate((id,np.ones(file_targets.shape[0]).reshape(-1,1)),axis=0)
    else:
        cross_section = f.Get('cross_section').GetVal()
        weight = np.concatenate((weight,L*(total_weight*cross_section/event_weight_sum)),axis=0)
        id = np.concatenate((id,np.zeros(file_targets.shape[0]).reshape(-1,1)),axis=0)

################################################################################
# Building network #        
################################################################################
print ('='*80)
print ('[INFO] Building new network')
# Input preparation #
inputs[:,5] -= inputs[:,2] # phi_lep2 - phi_lep1
inputs[:,8] -= inputs[:,2] # phi_bjet1 - phi_lep1
inputs[:,11] -= inputs[:,2] # phi_bjet2 - phi_lep1
inputs = np.c_[inputs[:,:2],inputs[:,3:]] # phi_lep1 set to 0 => not used

# Reweighting #

sig_weight_sum = np.sum(weight[id==1])
back_weight_sum = np.sum(weight[id==0])
weight[id==0] *= sig_weight_sum/back_weight_sum

# DNN and eval subset #
weight = weight.reshape(-1,)

DNN_inputs, eval_inputs, DNN_targets, eval_targets, DNN_weight, eval_weight = train_test_split(inputs,targets,weight,train_size=0.7)

train_inputs, test_inputs, train_targets, test_targets, train_weight, test_weight = train_test_split(DNN_inputs,DNN_targets,DNN_weight,train_size=0.7)


# Building layers #
inputs = Input(shape=(train_inputs.shape[1],))
L1 = Dense(50,activation='relu',name='L1',activity_regularizer=l2(0.01))(inputs)
L2 = Dense(50,activation='relu',name='L2',activity_regularizer=l2(0.01))(L1)
L3 = Dense(50,activation='relu',name='L3',activity_regularizer=l2(0.01))(L2)
Mjj = Dense(1, activation='relu',name='Mlljj')(L3)
Mlljj = Dense(1, activation='relu',name='Mjj')(L3)
Mll = Dense(1, activation='relu',name='Mll')(L3)

DNN = Model(inputs=[inputs], outputs=[Mjj,Mlljj,Mll])

utils.print_summary(model=DNN)

# Optimizer #
learning_rate = 0.0001
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

# Compile #
DNN.compile(optimizer=adam,loss={'Mjj':'mean_squared_error','Mlljj':'mean_squared_error', 'Mll':'mean_squared_error'},metrics=['accuracy'])

# set the weights #
try:
    with open('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/DYmodel.json', "r") as json_model_file:
        model_json_save = json_model_file.read()
    model_pre = model_from_json(model_json_save) # load model 
    model_pre.load_weights('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/DYweight.h5') # load weights
    print ('[INFO] Loading pre-stage DNN weights : Success')
except:
    sys.exit('Could not find pre-stage DNN model')

DNN.get_layer("L1").set_weights(model_pre.get_layer('dense_1').get_weights())
DNN.get_layer("L2").set_weights(model_pre.get_layer('dense_2').get_weights())
DNN.get_layer("L3").set_weights(model_pre.get_layer('dense_3').get_weights())
DNN.get_layer("Mjj").set_weights(model_pre.get_layer('dense_4').get_weights())
DNN.get_layer("Mll").set_weights(model_pre.get_layer('dense_5').get_weights())
DNN.get_layer("Mlljj").set_weights(model_pre.get_layer('dense_6').get_weights())

print (model_pre.get_config())
# Callbacks #
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
csv_logger = CSVLogger(path_out+'training.log')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.0001)
checkpoint = ModelCheckpoint(path_out+args.label+'weight.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

Callback_list = [csv_logger,checkpoint,early_stopping,reduceLR]

# Fit #
epoch = 200
batch = 1000

DNN.fit(train_inputs,{'Mlljj':train_targets[:,0], 'Mjj':train_targets[:,1], 'Mll':train_targets[:,2]}, sample_weight={'Mlljj':train_weight, 'Mjj':train_weight, 'Mll':train_weight}, epochs=epoch, batch_size=batch, verbose=2, validation_data=(test_inputs,[test_targets[:,0], test_targets[:,1], test_targets[:,2]],[test_weight, test_weight, test_weight]), callbacks=Callback_list)

# Save model #
model_json = DNN.to_json()
with open(path_out+args.label+'model.json', "w") as json_file:
    json_file.write(model_json)
# The weights are saved via the checkpoint callback
print ('[INFO] Model saved as '+path_out+args.label+'model.json')

# Output of validation set #
DNN.load_weights(path_out+args.label+'weight.h5')
NN_output = np.concatenate(DNN.predict(eval_inputs),axis=1)
model_output = np.c_[eval_targets,NN_output]

model_output.dtype = [('m_lljj','float64'),('m_jj','float64'),('m_ll','float64'),('NN_m_lljj','float64'),('NN_m_jj','float64'),('NN_m_ll','float64')]

array2root(model_output,path_out+'output.root',mode='recreate')

print ('[INFO] Output saved as '+path_out+'output.root')




