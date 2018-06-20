# Libraries #
import sys 
import glob
import os
import re
import argparse
import math
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TLorentzVector
from root_numpy import tree2array
from sklearn.model_selection import train_test_split
from keras import backend as K

# Personal Libraries #
from NeuralNet import *

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Build Graph for given subest of mA,mH') 
parser.add_argument('-l','--label', help='Label for naming model -> COMPULSORY',required=True,type=str)


args = parser.parse_args()

###############################################################################
# Import Root Files #
###############################################################################
print ('='*80)
print ('[INFO] Starting input from files')
INPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output_transform/PtEtaPhiM/'

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)


sig_set = np.empty((0,22))
TT_set = np.empty((0,22))
DY_set = np.empty((0,22))

sig_MEM = np.empty((0,2))
TT_MEM =np.empty((0,2))
DY_MEM = np.empty((0,2))

sig_weight =np.empty((0,1))
sig_learning_weight =np.empty((0,1))
TT_weight = np.empty((0,1))
DY_weight = np.empty((0,1)) 

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

    met_pt  = tree2array(t, branches='met_pt')
    met_phi  = tree2array(t, branches='met_phi')

    jj_M  = tree2array(t, branches='jj_M')
    ll_M  = tree2array(t, branches='ll_M')
    lljj_M  = tree2array(t, branches='lljj_M')

    MEM_TT = tree2array(t, branches='weight_TT').reshape(-1,1)
    MEM_TT_err = tree2array(t, branches='weight_TT_err').reshape(-1,1)
    MEM_DY = tree2array(t, branches='weight_DY').reshape(-1,1)
    MEM_DY_err = tree2array(t, branches='weight_DY_err').reshape(-1,1)

    MEM = np.c_[MEM_TT,MEM_DY]
    filedata = np.c_[lep1_Pt,lep1_Eta,lep1_Phi,lep2_Pt,lep2_Eta,lep2_Phi,jet1_Pt,jet1_Eta,jet1_Phi,jet2_Pt,jet2_Eta,jet2_Phi,met_pt,met_phi,jj_M,ll_M,lljj_M]

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
        mH = np.ones(filedata.shape[0])*num[2]
        mA = np.ones(filedata.shape[0])*num[3]
        vis_xsec = np.ones(filedata.shape[0])*cross_section*np.sum(total_weight)/event_weight_sum
        filedata = np.c_[filedata,mH,mA,vis_xsec,MEM_TT_err,MEM_DY_err]
        # Records data #
        sig_set = np.concatenate((sig_set,filedata),axis=0)
        sig_MEM = np.concatenate((sig_MEM,MEM),axis=0)
        sig_learning_weight = np.concatenate((sig_learning_weight,total_weight/np.sum(total_weight)),axis=0) 
        weight = L*(total_weight*cross_section/event_weight_sum)
        sig_weight = np.concatenate((sig_weight,weight),axis=0) 
    else:
        print ('\t-> Background',end=" ")
        cross_section = f.Get('cross_section').GetVal()
        weight = L*(total_weight*cross_section/event_weight_sum)
        vis_xsec = np.ones(filedata.shape[0])*cross_section*np.sum(total_weight)/event_weight_sum
        filedata = np.c_[filedata,np.zeros(filedata.shape[0]),np.zeros(filedata.shape[0]),vis_xsec,MEM_TT_err,MEM_DY_err]
        if filename.startswith('TT'): # Background TTBar
            print ('TTBar')
            TT_set = np.concatenate((TT_set,filedata),axis=0)
            TT_MEM = np.concatenate((TT_MEM,MEM),axis=0)
            TT_weight = np.concatenate((TT_weight,weight),axis=0)
        elif filename.startswith('DY'): # Background DY
            print ('DY')
            DY_set = np.concatenate((DY_set,filedata),axis=0)
            DY_MEM = np.concatenate((DY_MEM,MEM),axis=0)
            DY_weight = np.concatenate((DY_weight,weight),axis=0)
        else:
            sys.exit('File not found')
    print ('\tWeight sum =',np.sum(weight))
print ('Signal dataset size : ',sig_set.shape[0])
print ('TTBar dataset size : ',TT_set.shape[0])
print ('DY dataset size : ',DY_set.shape[0])

###############################################################################
# Create dataset # 
###############################################################################
print ('='*80)
print ('[INFO] Starting set preparation')

# Reweighting signal with respect to Backgrounds #
sig_learning_weight *= (np.sum(TT_weight)+np.sum(DY_weight))/np.sum(sig_learning_weight)

TT_learning_weight = TT_weight
DY_learning_weight = DY_weight


# Concatenating each dataset #
sig_id = np.zeros(sig_set.shape[0]) 
TT_id = np.ones(TT_set.shape[0])*1
DY_id = np.ones(DY_set.shape[0])*2

sig = np.c_[sig_set,sig_weight,sig_id]
TT = np.c_[TT_set,TT_weight,TT_id]
DY = np.c_[DY_set,DY_weight,DY_id]

data = np.concatenate((sig,TT,DY),axis=0) 
MEM_tot = np.concatenate((sig_MEM,TT_MEM,DY_MEM),axis=0)
learning_weight = np.concatenate((sig_learning_weight,TT_learning_weight,DY_learning_weight),axis=0) 

# Phi rescaling (wrt first lepton) #
data[:,5] -= data[:,2] # phi_lep2 - phi_lep1
data[:,8] -= data[:,2] # phi_bjet1 - phi_lep1
data[:,11] -= data[:,2] # phi_bjet2 - phi_lep1
data = np.c_[data[:,:2],data[:,3:]] # phi_lep1 set to 0 => not used

# Keep track of original weights #
data = np.c_[data,MEM_tot]

# Rescaling of the MEM weights (for log10) #
MEM_TT_tot = MEM_tot[:,0]/np.max(MEM_tot[:,0])
MEM_DY_tot = MEM_tot[:,1]/np.max(MEM_tot[:,1])

print ('Normalizing factor for TT weights : ',np.max(MEM_tot[:,0]))
print ('Normalizing factor for DY weights : ',np.max(MEM_tot[:,1]))

# Saving normalization in file #
path = '/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/'+args.label+'/'
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path+'normalization.txt', np.c_[np.max(MEM_tot[:,0]),np.max(MEM_tot[:,1])])

# Valid and invalid id's #

valid_id_TT = np.greater(MEM_TT_tot,10e-35)
valid_id_DY = np.greater(MEM_DY_tot,10e-35)
valid_both = np.logical_and(valid_id_TT,valid_id_DY)
invalid_id_TT = np.logical_and(valid_id_DY,np.logical_not(valid_both)) # Valid for DY but not TT
invalid_id_DY = np.logical_and(valid_id_TT,np.logical_not(valid_both)) # Valid for TT but not DY
MEM_TT_tot_valid = MEM_TT_tot[valid_both]
MEM_DY_tot_valid = MEM_DY_tot[valid_both]
learning_weight_valid = learning_weight[valid_both]
learning_weight_invalid_TT = learning_weight[invalid_id_TT]
learning_weight_invalid_DY = learning_weight[invalid_id_DY]
data_valid = data[valid_both]
data_invalid_TT = data[invalid_id_TT]
data_invalid_DY = data[invalid_id_DY]

# Valid in one but not two weights #
print ('Valid weights : ',data_valid.shape[0])
print ('Invalid DY but valid TT :',data_invalid_DY.shape[0])
print ('Invalid TT but valid DY :',data_invalid_TT.shape[0])

MEM_TT_tot_valid = -np.log10(MEM_TT_tot_valid)
MEM_DY_tot_valid = -np.log10(MEM_DY_tot_valid)
# MEM_tot = [MEM_TT,MEM_DY]

################################################################################
# Sets definition #
################################################################################
print ('='*80)
print ('Starting Separation')
# Dicriminating set #
#mask = np.full(data_valid.shape[0], False)
#mask[:round(data_valid.shape[0]*0.33)] = True
#np.random.shuffle(mask)
#np.savetxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/mask.txt',mask)
mask = np.genfromtxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/mask.txt').astype('bool')
not_mask = np.logical_not(mask)
# True => testing set, False => training set

# Separation #
set_train_TT = data_valid[not_mask,:13]
set_train_DY = data_valid[not_mask,:13]
set_test_TT = data_valid[mask,:13]
set_test_DY = data_valid[mask,:13]
target_train_TT = MEM_TT_tot_valid[not_mask]
target_train_DY = MEM_DY_tot_valid[not_mask]
target_test_TT = MEM_TT_tot_valid[mask]
target_test_DY = MEM_DY_tot_valid[mask]
masses_train_TT = data_valid[not_mask,13:16]
masses_train_DY = data_valid[not_mask,13:16]
masses_test_TT = data_valid[mask,13:16]
masses_test_DY = data_valid[mask,13:16]
weight_train_TT = learning_weight_valid[not_mask]
weight_train_DY = learning_weight_valid[not_mask]
weight_test_TT = learning_weight_valid[mask]
weight_test_DY = learning_weight_valid[mask]
other_train_TT = data_valid[not_mask,16:]
other_train_DY = data_valid[not_mask,16:]
other_test_TT = data_valid[mask,16:]
other_test_DY = data_valid[mask,16:]

# Add weights valid for one but not two in training set #
set_train_TT = np.concatenate((set_train_TT,data_invalid_DY[:,:13]),axis=0)
set_train_DY = np.concatenate((set_train_DY,data_invalid_TT[:,:13]),axis=0)
target_train_TT = np.concatenate((target_train_TT,-np.log10(MEM_TT_tot[invalid_id_DY])),axis=0)
target_train_DY = np.concatenate((target_train_DY,-np.log10(MEM_DY_tot[invalid_id_TT])),axis=0)
masses_train_TT = np.concatenate((masses_train_TT,data_invalid_DY[:,13:16]),axis=0)
masses_train_DY = np.concatenate((masses_train_DY,data_invalid_TT[:,13:16]),axis=0)
weight_train_TT = np.concatenate((weight_train_TT,learning_weight_invalid_DY),axis=0)
weight_train_DY = np.concatenate((weight_train_DY,learning_weight_invalid_TT),axis=0)
other_train_TT = np.concatenate((other_train_TT,data_invalid_DY[:,16:]),axis=0)
other_train_DY = np.concatenate((other_train_DY,data_invalid_TT[:,16:]),axis=0)

print ('Train set TT: ',set_train_TT.shape)
print ('Train set DY: ',set_train_DY.shape)
print ('Test set TT : ',set_test_TT.shape)
print ('Test set DY : ',set_test_DY.shape)

set_invalid_TT = data_invalid_TT[:,:13]
set_invalid_DY = data_invalid_DY[:,:13]
masses_invalid_TT = data_invalid_TT[:,13:16]
masses_invalid_DY = data_invalid_DY[:,13:16]
other_invalid_TT = data_invalid_TT[:,16:]
other_invalid_DY = data_invalid_DY[:,16:]
print ('Invalid set TT : ',set_invalid_TT.shape)
print ('Invalid set DY : ',set_invalid_DY.shape)

#set_train = np.c_[set_train,masses_train]
#set_test = np.c_[set_test,masses_test]
#set_invalid = np.c_[set_invalid,masses_invalid]
# Preprocessing #
scaler_TT = preprocessing.StandardScaler().fit(set_train_TT)
scaler_DY = preprocessing.StandardScaler().fit(set_train_DY)
set_train_TT = scaler_TT.transform(set_train_TT)
set_train_DY = scaler_DY.transform(set_train_DY)
set_test_TT = scaler_TT.transform(set_test_TT)
set_test_DY = scaler_DY.transform(set_test_DY)
set_invalid_TT = scaler_TT.transform(set_invalid_TT)
set_invalid_DY = scaler_DY.transform(set_invalid_DY)
#scaler_TT = preprocessing.MinMaxScaler().fit(set_train_TT)
#scaler_DY = preprocessing.MinMaxScaler().fit(set_train_DY)
#set_train_TT = scaler_TT.transform(set_train_TT)
#set_train_DY = scaler_DY.transform(set_train_DY)
#set_test_TT = scaler_TT.transform(set_test_TT)
#set_test_DY = scaler_DY.transform(set_test_DY)
#set_invalid_TT = scaler_TT.transform(set_invalid_TT)
#set_invalid_DY = scaler_DY.transform(set_invalid_DY)

np.savetxt(path+'scaler_TT.txt',np.c_[scaler_TT.mean_,scaler_TT.scale_])
np.savetxt(path+'scaler_DY.txt',np.c_[scaler_DY.mean_,scaler_DY.scale_])


################################################################################
# Neural Network Learning #
################################################################################
K.clear_session()
k = 10 # Cross validatiion steps
# Neural Net for TTbar weights #
print ('='*80)
print ("[INFO] Starting learning on TT weights")
for i in range(1,k+1):
    print ('\n[INFO] Cross validation step : ',i)
    instance_TT = NeuralNet(data=set_train_TT,target=target_train_TT,masses=masses_train_TT,weight=weight_train_TT,label_model=args.label,label_target='TT',model_number=i)
    instance_TT.BuildModel()
    instance_TT.PlotHistory()
    instance_TT.PrintModel()

print ('-'*80)
print ('\n[INFO] Evaluating model')
print ('\t[INFO] Valid weights')
mse_TT = instance_TT.UseModel(data=set_test_TT,target=target_test_TT,masses=masses_test_TT,other=other_test_TT,label_output='')
print ('\t[INFO] Invalid weights')
mse_TT_inv = instance_TT.UseModel(data=set_invalid_TT,target=np.zeros(set_invalid_TT.shape[0]),masses=masses_invalid_TT,other=other_invalid_TT,label_output='invalid')

K.clear_session()
# Neural Net for Drell Yann weights #
print ('='*80)
print ("[INFO] Starting learning on DY weights")
for i in range(0,k):
    print ('\n[INFO] Cross validation step : ',i)
    instance_DY = NeuralNet(data=set_train_DY,target=target_train_DY,masses=masses_train_DY,weight=weight_train_DY,label_model=args.label,label_target='DY',model_number=i)
    instance_DY.BuildModel()
    instance_DY.PlotHistory()
    instance_DY.PrintModel()

print ('-'*80)
print ('\n[INFO] Evaluating model')
print ('\t[INFO] Valid weights')
mse_DY = instance_DY.UseModel(data=set_test_DY,target=target_test_DY,masses=masses_test_DY,other=other_test_DY,label_output='')
print ('\t[INFO] Invalid weights')
mse_DY_inv = instance_DY.UseModel(data=set_invalid_DY,target=np.zeros(set_invalid_DY.shape[0]),masses=masses_invalid_DY,other=other_invalid_DY,label_output='invalid')

print ("-"*80)
print ('TT MSE : ',mse_TT)
print ('DY MSE : ',mse_DY)
print ('TT inv MSE : ',mse_TT_inv)
print ('DY inv MSE : ',mse_DY_inv)


