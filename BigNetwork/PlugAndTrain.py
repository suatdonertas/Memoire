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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Lambda, Dropout
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
parser = argparse.ArgumentParser(description='Plug the different networks and train the whole again')
parser.add_argument('-l','--label', help='Label for naming model -> COMPULSORY',required=True,type=str)
args = parser.parse_args()

path_out = '/home/ucl/cp3/fbury/storage/PlugBigNetwork/'+args.label+'/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

###############################################################################
# Import Root from MoMEMta Network #
###############################################################################
print ('='*80)
print ('[INFO] Starting input from files')
INPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output_transform/PtEtaPhiM/'

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)

sig_part = np.empty((0,14))
sig_invmass = np.empty((0,2))
sig_massparam = np.empty((0,2))
back_part = np.empty((0,14))
back_invmass = np.empty((0,2))
back_massparam = np.empty((0,2))

sig_weight =np.empty((0,1))
back_weight =np.empty((0,1))

for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    selection = "ll_M>70 && ll_M<110 && met_pt<80"

    # Get branches into numpy arrays #
    total_weight = tree2array(t, branches='total_weight',selection=selection).reshape(-1,1)
    event_weight = tree2array(t, branches='event_weight',selection=selection).reshape(-1,1)
    event_weight_sum = f.Get('event_weight_sum').GetVal()
    L = 35922

    lep1_Pt = tree2array(t, branches='lep1_Pt',selection=selection)
    lep1_Eta = tree2array(t, branches='lep1_Eta',selection=selection)
    lep1_Phi = tree2array(t, branches='lep1_Phi',selection=selection)

    lep2_Pt = tree2array(t, branches='lep2_Pt',selection=selection)
    lep2_Eta = tree2array(t, branches='lep2_Eta',selection=selection)
    lep2_Phi = tree2array(t, branches='lep2_Phi',selection=selection)

    jet1_Pt = tree2array(t, branches='jet1_Pt',selection=selection)
    jet1_Eta = tree2array(t, branches='jet1_Eta',selection=selection)
    jet1_Phi = tree2array(t, branches='jet1_Phi',selection=selection)

    jet2_Pt = tree2array(t, branches='jet2_Pt',selection=selection)
    jet2_Eta = tree2array(t, branches='jet2_Eta',selection=selection)
    jet2_Phi = tree2array(t, branches='jet2_Phi',selection=selection)

    met_pt  = tree2array(t, branches='met_pt',selection=selection)
    met_phi  = tree2array(t, branches='met_phi',selection=selection)

    jj_M  = tree2array(t, branches='jj_M',selection=selection)
    lljj_M  = tree2array(t, branches='lljj_M',selection=selection)

    file_part = np.c_[lep1_Pt,lep1_Eta,lep1_Phi,lep2_Pt,lep2_Eta,lep2_Phi,jet1_Pt,jet1_Eta,jet1_Phi,jet2_Pt,jet2_Eta,jet2_Phi,met_pt,met_phi]
    file_invmass = np.c_[lljj_M,jj_M]

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('\t-> Signal')

        # Find correct cross section #
        num = [int(s) for s in re.findall('\d+',filename)]

        cross_section = 0
        for c in range(0,xsec.shape[1]):
            if xsec[1,c]==num[2] and xsec[2,c]==num[3]:
                cross_section = xsec[0,c]
        if cross_section == 0:
            sys.exit('Could not find cross section in signal sample')
        
        mH = np.ones(file_part.shape[0])*num[2]
        mA = np.ones(file_part.shape[0])*num[3]
        file_massparam = np.c_[mH,mA]

        # Add values to full array #
        sig_weight = np.concatenate((sig_weight,total_weight/np.sum(total_weight)),axis=0)
        sig_part = np.concatenate((sig_part,file_part),axis=0)
        sig_invmass = np.concatenate((sig_invmass,file_invmass),axis=0)
        sig_massparam = np.concatenate((sig_massparam,file_massparam),axis=0)

    else:
        print ('\t-> Background')
        if filename.startswith('TTTo2L2Nu'):
            event_weight_sum /= 6
            print ('hello')

        cross_section = f.Get('cross_section').GetVal()
        weight = L*(total_weight*cross_section/event_weight_sum)

        # Add values to full array #
        back_weight = np.concatenate((back_weight,L*(total_weight*cross_section/event_weight_sum)),axis=0)
        back_part = np.concatenate((back_part,file_part),axis=0)
        back_invmass = np.concatenate((back_invmass,file_invmass),axis=0)

###############################################################################
# Input preparation #
###############################################################################
print ('='*80)
print ('[INFO] Input preparation')
print ('Signal set : ',sig_part.shape[0])
print ('Background set : ',back_part.shape[0])

# Reweighting #
mHmA = np.unique(sig_massparam,axis=0)
for mH,mA in mHmA:
    mask = np.logical_and(sig_massparam[:,0]==mH,sig_massparam[:,1]==mA)
    sig_weight[mask] *= sig_weight.shape[0]/sig_weight[mask].shape[0]
    #sig_weight /= np.sum(sig_weight[mask])
# all signal samples have the same importance with respect to their number
sig_weight *= 1e6

back_weight = back_weight*np.sum(sig_weight)/np.sum(back_weight)
# Sum weights (back) = sum weights (sig)

# Background random assignation of mH, mA #
proba = np.ones(mHmA.shape[0])/mHmA.shape[0]
indices = np.arange(0,mHmA.shape[0])

rs = np.random.RandomState(42)
back_massparam = mHmA[rs.choice(indices,size=back_invmass.shape[0],p=proba)]

print ('\n Total set')
for mH,mA in mHmA:
    mask_sig = np.logical_and(sig_massparam[:,0]==mH,sig_massparam[:,1]==mA)
    mask_back = np.logical_and(back_massparam[:,0]==mH,back_massparam[:,1]==mA)
    print ('mH = %0.f, mA = %0.f ->\tSignal : %0.f (%0.2f%%)\t Background : %0.f (%0.2f%%)'%(mH,mA,sig_massparam[mask_sig,:].shape[0],(sig_massparam[mask_sig,:].shape[0]/sig_massparam.shape[0]*100),back_massparam[mask_back,:].shape[0],(back_massparam[mask_back,:].shape[0]/back_massparam.shape[0]*100)))
print ('-'*60)
print ('Total signal sample : ',sig_massparam.shape[0],'\tTotal background sample : ',back_massparam.shape[0],'\n\n') 

# Phi rescaling (wrt first lepton) #
sig_part[:,5] -= sig_part[:,2] # phi_lep2 - phi_lep1
sig_part[:,8] -= sig_part[:,2] # phi_bjet1 - phi_lep1
sig_part[:,11] -= sig_part[:,2] # phi_bjet2 - phi_lep1
sig_part = np.c_[sig_part[:,:2],sig_part[:,3:]] # phi_lep1 set to 0 => not used
back_part[:,5] -= back_part[:,2] # phi_lep2 - phi_lep1
back_part[:,8] -= back_part[:,2] # phi_bjet1 - phi_lep1
back_part[:,11] -= back_part[:,2] # phi_bjet2 - phi_lep1
back_part = np.c_[back_part[:,:2],back_part[:,3:]] # phi_lep1 set to 0 => not used

# Targets #
sig_target = np.ones(sig_part.shape[0])
back_target = np.zeros(back_part.shape[0])

###############################################################################
# Set preparation #
###############################################################################
print ('[INFO] Input separation')
set_part = np.concatenate((sig_part,back_part),axis=0)
set_massparam = np.concatenate((sig_massparam,back_massparam),axis=0)
set_invmass = np.concatenate((sig_invmass,back_invmass),axis=0)
set_weight = np.concatenate((sig_weight,back_weight),axis=0)
set_target = np.concatenate((sig_target,back_target),axis=0)

set_weight = set_weight.reshape(-1,)

#mask = np.full((set_weight.shape[0],), False, dtype=bool)
#mask[:int(0.7*set_weight.shape[0])] = True
#np.random.shuffle(mask)
#np.savetxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/mask.txt',mask)
mask = np.genfromtxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/mask.txt')
# False => Evaluation set, True => Training set

DNN_part = set_part[mask==True]
DNN_massparam = set_massparam[mask==True]
DNN_invmass = set_invmass[mask==True]
DNN_weight = set_weight[mask==True]
DNN_target = set_target[mask==True]

eval_part = set_part[mask==False]
eval_massparam = set_massparam[mask==False]
eval_invmass = set_invmass[mask==False]
eval_weight = set_weight[mask==False]
eval_target = set_target[mask==False]

print ('Training set : ',DNN_part.shape[0],' (Signal = ',DNN_part[DNN_target==1].shape[0],' , Background = ',DNN_part[DNN_target==0].shape[0],')')
print ('Evaluation set : ',eval_part.shape[0],' (Signal = ',eval_part[eval_target==1].shape[0],' , Background = ',eval_part[eval_target==0].shape[0],')')

# Validation list #
eval_list = [eval_part,eval_massparam,eval_invmass]

# Full output declaration #
output_full = np.zeros((eval_part.shape[0],5))
###############################################################################
# Cross validation #
###############################################################################
k = 10 # Cross validation steps 

for i in range(1,k+1):
    print ('-'*80)
    print ('[INFO] Cross validation step : ',i)
    train_part, test_part, train_massparam, test_massparam, train_invmass, test_invmass, train_weight, test_weight, train_target, test_target = train_test_split(DNN_part,DNN_massparam,DNN_invmass,DNN_weight,DNN_target,shuffle=True, train_size=0.7)

    # DNN building #
    print ('[INFO] Building Model')

    # Inputs #
    input_part = Input(shape=(DNN_part.shape[1],),name='input_part')
    input_massparam = Input(shape=(DNN_massparam.shape[1],),name='input_massparam')
    input_invmass = Input(shape=(DNN_invmass.shape[1],),name='input_invmass')


    # Pre stage DNN for masses #
    #L1_pre = Dense(50,activation='relu',name='L1_pre',activity_regularizer=l2(0))(input_part)
    #L1_pre = Dropout(0)(L1_pre)
    #L2_pre = Dense(50,activation='relu',name='L2_pre',activity_regularizer=l2(0))(L1_pre)
    #L2_pre = Dropout(0)(L2_pre)
    #L3_pre = Dense(50,activation='relu',name='L3_pre',activity_regularizer=l2(0))(L2_pre)
    #L3_pre = Dropout(0)(L3_pre)
    #output_mlljj = Dense(1,activation='relu',name='output_mlljj')(L3_pre)
    #output_mjj = Dense(1,activation='relu',name='output_mjj')(L3_pre)


    # Mass Plane Branch #
    input_MP = Concatenate(axis=-1)([input_invmass,input_massparam])
    L1_MP = Dense(30,activation='relu',name='L1_MP',activity_regularizer=l2(0))(input_MP)
    L1_MP = Dropout(0)(L1_MP)
    L2_MP = Dense(30,activation='relu',name='L2_MP',activity_regularizer=l2(0))(L1_MP)
    L2_MP = Dropout(0)(L2_MP)
    L3_MP = Dense(30,activation='relu',name='L3_MP',activity_regularizer=l2(0))(L2_MP)
    L3_MP = Dropout(0)(L3_MP)
    output_MP = Dense(1,activation='sigmoid',name='output_MP')(L3_MP)

    # MoMEMta TT branch #
    norm_TT =  np.genfromtxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/scaler_TT.txt')
    mean_TT = norm_TT[:,0]
    scale_TT = norm_TT[:,1]
    def preprocessing_TT(a):
        return (a-mean_TT)/scale_TT

    pre_TT = Lambda(preprocessing_TT,output_shape=(13,))(input_part)
    L1_MEM_TT = Dense(50,activation='relu',name='L1_MEM_TT',activity_regularizer=l2(0))(pre_TT)
    L1_MEM_TT = Dropout(0)(L1_MEM_TT)
    L2_MEM_TT = Dense(50,activation='relu',name='L2_MEM_TT',activity_regularizer=l2(0))(L1_MEM_TT)
    L2_MEM_TT = Dropout(0)(L2_MEM_TT)
    L3_MEM_TT = Dense(50,activation='relu',name='L3_MEM_TT',activity_regularizer=l2(0))(L2_MEM_TT)
    L3_MEM_TT = Dropout(0)(L3_MEM_TT)
    L4_MEM_TT = Dense(50,activation='relu',name='L4_MEM_TT',activity_regularizer=l2(0))(L3_MEM_TT)
    L4_MEM_TT = Dropout(0)(L4_MEM_TT)
    output_MEM_TT = Dense(1,activation='selu',name='output_MEM_TT')(L4_MEM_TT)

    # MoMEMta DY branch #
    norm_DY =  np.genfromtxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/scaler_DY.txt')
    mean_DY = norm_DY[:,0]
    scale_DY = norm_DY[:,1]
    def preprocessing_DY(a):
        return (a-mean_DY)/scale_DY

    pre_DY = Lambda(preprocessing_DY,output_shape=(13,))(input_part)
    L1_MEM_DY = Dense(50,activation='relu',name='L1_MEM_DY',activity_regularizer=l2(0))(pre_DY)
    L1_MEM_DY = Dropout(0)(L1_MEM_DY)
    L2_MEM_DY = Dense(50,activation='relu',name='L2_MEM_DY',activity_regularizer=l2(0))(L1_MEM_DY)
    L2_MEM_DY = Dropout(0)(L2_MEM_DY)
    L3_MEM_DY = Dense(50,activation='relu',name='L3_MEM_DY',activity_regularizer=l2(0))(L2_MEM_DY)
    L3_MEM_DY = Dropout(0)(L3_MEM_DY)
    L4_MEM_DY = Dense(50,activation='relu',name='L4_MEM_DY',activity_regularizer=l2(0))(L3_MEM_DY)
    L4_MEM_DY = Dropout(0)(L4_MEM_DY)
    output_MEM_DY = Dense(1,activation='selu',name='output_MEM_DY')(L4_MEM_DY)

    # Discriminant layer #
    norm =  np.genfromtxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep/normalization.txt')
    norm_TT = norm[0] 
    norm_DY = norm[1] 
    def dicriminant(a):
        # a[0]->TT , a[1]->DY
        # Get original weights #
        TT = K.pow(10.,-a[0])*norm_TT
        DY = K.pow(10.,-a[1])*norm_DY
        return TT/(TT+200*DY)

    dis = Lambda(dicriminant,output_shape=(1,))([output_MEM_TT,output_MEM_DY])

    # Last layer #
    preprocess =  np.genfromtxt('/home/ucl/cp3/fbury/storage/BigNetworkOutput/preprocessing.txt')
    scale_pre = preprocess[:,0]
    min_pre = preprocess[:,1]

    def preprocessing_last(a):
        # a.shape = (6,0)
        return a*scale_pre+min_pre

    input_last = Concatenate(axis=-1)([output_MP,dis,output_MEM_TT,output_MEM_DY,input_massparam])
    pre_last = Lambda(preprocessing_last,output_shape=(6,))(input_last)
    L1_last = Dense(30,activation='relu',name='L1_last',activity_regularizer=l2(0))(pre_last)
    L1_last = Dropout(0)(L1_last)
    L2_last = Dense(30,activation='relu',name='L2_last',activity_regularizer=l2(0))(L1_last)
    L2_last = Dropout(0)(L2_last)
    L3_last = Dense(30,activation='relu',name='L3_last',activity_regularizer=l2(0))(L2_last)
    L3_last = Dropout(0)(L3_last)
    output_last = Dense(1,activation='sigmoid',name='output_last')(L3_last)

    # Define model #

    DNN = Model(inputs=[input_part,input_massparam,input_invmass], outputs=[output_MP,output_MEM_TT,output_MEM_DY,dis,output_last])
    #utils.print_summary(model=DNN)

    # Preset pre-stage weights #
    #try:
    #    with open('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/DYmodel.json', "r") as json_model_file:
    #        model_json_save = json_model_file.read()
    #    model_pre = model_from_json(model_json_save) # load model 
    #    model_pre.load_weights('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/DYweight.h5') # load weights
    #    print ('[INFO] Loading pre-stage DNN weights : Success')
    #except:
    #    sys.exit('Could not find pre-stage DNN model')

    #DNN.get_layer("L1_pre").set_weights(model_pre.get_layer('dense_1').get_weights())
    #DNN.get_layer("L2_pre").set_weights(model_pre.get_layer('dense_2').get_weights())
    #DNN.get_layer("L3_pre").set_weights(model_pre.get_layer('dense_3').get_weights())
    #DNN.get_layer("output_mjj").set_weights(model_pre.get_layer('dense_4').get_weights())
    #DNN.get_layer("output_mlljj").set_weights(model_pre.get_layer('dense_6').get_weights())


    # Preset MP DNN weights #
    try:
        with open('/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_l2/model_'+str(i)+'.json', "r") as json_model_file:
            model_json_save = json_model_file.read()
        model_MP = model_from_json(model_json_save) # load model 
        model_MP.load_weights('/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_l2/weight_'+str(i)+'.h5') # load weights
        print ('[INFO] Loading MP DNN weights : Success')
    except:
        sys.exit('Could not find MP DNN model')

    DNN.get_layer("L1_MP").set_weights(model_MP.get_layer('dense_'+str(1+4*(i-1))).get_weights())
    DNN.get_layer("L2_MP").set_weights(model_MP.get_layer('dense_'+str(2+4*(i-1))).get_weights())
    DNN.get_layer("L3_MP").set_weights(model_MP.get_layer('dense_'+str(3+4*(i-1))).get_weights())
    DNN.get_layer("output_MP").set_weights(model_MP.get_layer('dense_'+str(4+4*(i-1))).get_weights())

    # Preset MEM TT DNN weights #
    try:
        with open('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep_cross/TTmodel_'+str(i)+'.json', "r") as json_model_file:
            model_json_save = json_model_file.read()
        model_MEM_TT = model_from_json(model_json_save) # load model 
        model_MEM_TT.load_weights('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep_cross/TTweight_'+str(i)+'.h5') # load weights
        print ('[INFO] Loading MEM TT DNN weights : Success')
    except:
        sys.exit('Could not find MEM TT DNN model')

    DNN.get_layer("L1_MEM_TT").set_weights(model_MEM_TT.get_layer('dense_1').get_weights())
    DNN.get_layer("L2_MEM_TT").set_weights(model_MEM_TT.get_layer('dense_2').get_weights())
    DNN.get_layer("L3_MEM_TT").set_weights(model_MEM_TT.get_layer('dense_3').get_weights())
    DNN.get_layer("L4_MEM_TT").set_weights(model_MEM_TT.get_layer('dense_4').get_weights())
    DNN.get_layer("output_MEM_TT").set_weights(model_MEM_TT.get_layer('dense_5').get_weights())

# Preset MEM DY DNN weights #
    try:
        with open('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep_cross/DYmodel_'+str(i)+'.json', "r") as json_model_file:
            model_json_save = json_model_file.read()
        model_MEM_DY = model_from_json(model_json_save) # load model 
        model_MEM_DY.load_weights('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep_cross/DYweight_'+str(i)+'.h5') # load weights
        print ('[INFO] Loading MEM DY DNN weights : Success')
    except:
        sys.exit('Could not find MEM DY DNN model')


    DNN.get_layer("L1_MEM_DY").set_weights(model_MEM_DY.get_layer('dense_1').get_weights())
    DNN.get_layer("L2_MEM_DY").set_weights(model_MEM_DY.get_layer('dense_2').get_weights())
    DNN.get_layer("L3_MEM_DY").set_weights(model_MEM_DY.get_layer('dense_3').get_weights())
    DNN.get_layer("L4_MEM_DY").set_weights(model_MEM_DY.get_layer('dense_4').get_weights())
    DNN.get_layer("output_MEM_DY").set_weights(model_MEM_DY.get_layer('dense_5').get_weights())

    # Preset Last DNN weights #
    try:
        with open('/home/ucl/cp3/fbury/storage/BigNetworkOutput/3lay30_weights_dis_mass_pre_l2/model_'+str(i)+'.json', "r") as json_model_file:
            model_json_save = json_model_file.read()
        model_last = model_from_json(model_json_save) # load model 
        model_last.load_weights('/home/ucl/cp3/fbury/storage/BigNetworkOutput/3lay30_weights_dis_mass_pre_l2/weight_'+str(i)+'.h5') # load weights
        print ('[INFO] Loading Last DNN weights : Success')
    except :
        sys.exit('Could not find last DNN model')

    DNN.get_layer("L1_last").set_weights(model_last.get_layer('dense_1').get_weights())
    DNN.get_layer("L2_last").set_weights(model_last.get_layer('dense_2').get_weights())
    DNN.get_layer("L3_last").set_weights(model_last.get_layer('dense_3').get_weights())
    DNN.get_layer("output_last").set_weights(model_last.get_layer('dense_4').get_weights())


# Callback #
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='min')
    csv_logger = CSVLogger(path_out+'training_'+str(i)+'.log')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='min', epsilon=0.001, cooldown=3, min_lr=0.000001)
    checkpoint = ModelCheckpoint(path_out+'weight_'+str(i)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

    Callback_list = [csv_logger,checkpoint,early_stopping,reduceLR]

    # Compile #
    learning_rate = 0.0001
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False, clipvalue=0.5)
    DNN.compile(optimizer=adam,loss={'output_last':'mean_squared_error'},metrics=['accuracy'])

    # Fit #
    epoch = 300
    batch = 1000

    DNN.fit({'input_part':train_part, 'input_massparam':train_massparam, 'input_invmass':train_invmass},train_target,sample_weight=train_weight, epochs=epoch, batch_size=batch, verbose=2, callbacks=Callback_list,validation_data=({'input_part':test_part,'input_massparam':test_massparam,'input_invmass':test_invmass},test_target,test_weight))

    # Save model #
    model_json = DNN.to_json()
    with open(path_out+'model.json', "w") as json_file:
        json_file.write(model_json)
    # The weights are saved via the checkpoint callback
    print ('[INFO] Model saved as '+path_out+'model_'+str(i)+'.json')

    # Evaluating on eval_list #
    try:        
        DNN.load_weights(path_out+'weight_'+str(i)+'.h5')
    except:
        print ("Careful, no best weights to load !!!")

    NN_output = np.concatenate(DNN.predict(eval_list,verbose=1),axis=1)
    print ('MSE end : ',mse(eval_target,NN_output[:,-1],sample_weight=eval_weight))
    output_full += NN_output

################################################################################
# Performance evaluation #
################################################################################
print ("="*80)
print ('[INFO] Evaluating Model')
# average #
output_full /= k

print ('MSE from average : ',mse(eval_target,output_full[:,-1],sample_weight=eval_weight))

#print (roc_auc_score(eval_target,NN_output[-1]))
model_output = np.c_[output_full,eval_massparam,eval_invmass,eval_weight,eval_target]

model_output.dtype = [('output_MP','float64'),('output_MEM_TT','float64'),('output_MEM_DY','float64'),('discriminant','float64'),('output_last','float64'),('mH','float64'),('mA','float64'),('mlljj','float64'),('mjj','float64'),('weight','float64'),('target','float64')] 

array2root(model_output,path_out+'output.root',mode='recreate')

print ('[INFO] Output saved as '+path_out+'output.root')
