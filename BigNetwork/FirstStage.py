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

import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU
from keras.models import Model, model_from_json, load_model
from keras import losses, optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint, History, Callback
from keras.models import model_from_json
from keras.regularizers import l1,l2
import keras.backend as K

import matplotlib.pyplot as plt

# Personal libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
from NNOutput import NNOutput

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Getting output from first stage of Frankeistein network')
parser.add_argument('-l','--label', help='Label for naming model -> COMPULSORY',required=True,type=str)
args = parser.parse_args()

path_out = '/home/ucl/cp3/fbury/storage/BigNetwork_FistStage_Output/'+args.label+'/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

###############################################################################
# Import Root from MoMEMta Network #
###############################################################################
print ('='*80)
print ('[INFO] Starting input from files')
INPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output_transform/PtEtaPhiM/'

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)


sig_MP = np.empty((0,4))
sig_MEM = np.empty((0,14))
back_MP = np.empty((0,2))
back_MEM = np.empty((0,14))

sig_weight =np.empty((0,1))
back_weight =np.empty((0,1))

sig_vis_xsec = np.empty((0,))
back_vis_xsec = np.empty((0,))

for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    selection = 'met_pt<80 && ll_M>70 && ll_M<110'

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

    data_MEM = np.c_[lep1_Pt,lep1_Eta,lep1_Phi,lep2_Pt,lep2_Eta,lep2_Phi,jet1_Pt,jet1_Eta,jet1_Phi,jet2_Pt,jet2_Eta,jet2_Phi,met_pt,met_phi]
    data_MP = np.c_[lljj_M,jj_M]

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
        
        # Weights #
        sig_vis_xsec = np.concatenate((sig_vis_xsec,np.ones(total_weight.shape[0])*cross_section*np.sum(total_weight)/event_weight_sum),axis=0)
        sig_weight = np.concatenate((sig_weight,total_weight/np.sum(total_weight)),axis=0)
        # Mass Plane data #
        mH = np.ones(data_MP.shape[0])*num[2]
        mA = np.ones(data_MP.shape[0])*num[3]
        data_MP = np.c_[data_MP,mH,mA]
        sig_MP = np.concatenate((sig_MP,data_MP),axis=0)
        # MEM data #
        sig_MEM = np.concatenate((sig_MEM,data_MEM),axis=0)

    else:
        print ('\t-> Background')

        cross_section = f.Get('cross_section').GetVal()
        weight = L*(total_weight*cross_section/event_weight_sum)

        # Weights #
        back_vis_xsec = np.concatenate((back_vis_xsec,np.ones(total_weight.shape[0])*cross_section*np.sum(total_weight)/event_weight_sum),axis=0)
        back_weight = np.concatenate((back_weight,L*(total_weight*cross_section/event_weight_sum)),axis=0)
        # Mass Plane data #
        back_MP = np.concatenate((back_MP,data_MP),axis=0)
        # MEM data #
        back_MEM = np.concatenate((back_MEM,data_MEM),axis=0)

###############################################################################
# Input preparation #
###############################################################################
print ('='*80)
print ('[INFO] Input preparation')
print ('Signal set : ',sig_MEM.shape[0])
print ('Background set : ',back_MEM.shape[0])

# Reweighting #
mHmA = np.unique(sig_MP[:,2:4],axis=0)
for mH,mA in mHmA:
    mask = np.logical_and(sig_MP[:,2]==mH,sig_MP[:,3]==mA)
    sig_weight[mask] *= sig_weight.shape[0]/sig_weight[mask].shape[0]
    #sig_weight /= np.sum(sig_weight[mask])
# all signal samples have the same importance with respect to their number of events
sig_weight *= 1e6

back_weight = back_weight*np.sum(sig_weight)/np.sum(back_weight)
# Sum weights (back) = sum weights (sig)

# Background random assignation of mH, mA #
proba = np.ones(mHmA.shape[0])/mHmA.shape[0]
indices = np.arange(0,mHmA.shape[0])

rs = np.random.RandomState(42)
back_MP = np.c_[back_MP,mHmA[rs.choice(indices,size=back_MP.shape[0],p=proba)]]

print ('\n Total set')
for mH,mA in mHmA:
    mask_sig = np.logical_and(sig_MP[:,2]==mH,sig_MP[:,3]==mA)
    mask_back = np.logical_and(back_MP[:,2]==mH,back_MP[:,3]==mA)
    print ('mH = %0.f, mA = %0.f ->\tSignal : %0.f (%0.2f%%)\t Background : %0.f (%0.2f%%)'%(mH,mA,sig_MP[mask_sig,:].shape[0],(sig_MP[mask_sig,:].shape[0]/sig_MP.shape[0]*100),back_MP[mask_back,:].shape[0],(back_MP[mask_back,:].shape[0]/back_MP.shape[0]*100)))
print ('-'*60)
print ('Total signal sample : ',sig_MP.shape[0],'\tTotal background sample : ',back_MP.shape[0],'\n\n') 

# Phi rescaling (wrt first lepton) #
sig_MEM[:,5] -= sig_MEM[:,2] # phi_lep2 - phi_lep1
sig_MEM[:,8] -= sig_MEM[:,2] # phi_bjet1 - phi_lep1
sig_MEM[:,11] -= sig_MEM[:,2] # phi_bjet2 - phi_lep1
sig_MEM = np.c_[sig_MEM[:,:2],sig_MEM[:,3:]] # phi_lep1 set to 0 => not used
back_MEM[:,5] -= back_MEM[:,2] # phi_lep2 - phi_lep1
back_MEM[:,8] -= back_MEM[:,2] # phi_bjet1 - phi_lep1
back_MEM[:,11] -= back_MEM[:,2] # phi_bjet2 - phi_lep1
back_MEM = np.c_[back_MEM[:,:2],back_MEM[:,3:]] # phi_lep1 set to 0 => not used

# Preprocessing MEM inputs #
scaler = preprocessing.StandardScaler().fit(np.concatenate((sig_MEM,back_MEM),axis=0))
sig_MEM = scaler.transform(sig_MEM) 
back_MEM = scaler.transform(back_MEM) 

###############################################################################
# Mass Plane Output #
###############################################################################
print ('='*80)
print ('[INFO] Mass Plane Output')

path_MP = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_30_l2/'
#with open(path_MP+'model_step_1.json', "r") as json_model_file:
#    model_json_MP = json_model_file.read()
#model_MP = model_from_json(model_json_MP) # load model with best weights
#model_MP.load_weights(path_MP+'weight_step1.h5')
#out_MP_sig = model_MP.predict(sig_MP)
#out_MP_back = model_MP.predict(back_MP)

out_MP_sig = NNOutput(sig_MP,path_MP)
out_MP_back = NNOutput(back_MP,path_MP)
# Plot #
bins_MP = np.linspace(0,1,50)
fig_MP = plt.figure(1)
plt.hist(out_MP_sig,bins=bins_MP,color='g',alpha=0.7,label='Signal')
plt.hist(out_MP_back,bins=bins_MP,color='r',alpha=0.7,label='Background')
plt.legend(loc='upper center')
plt.title('Mass Plane DNN output')
plt.xlabel('DNN Output')
plt.ylabel('Occurences')
fig_MP.savefig(path_out+'MassPlaneOutput.png')
print ('Plot saved as '+path_out+'MassPlaneOutput.png')
plt.close()

###############################################################################
# TT MoMEMta Output #
###############################################################################
print ('='*80)
print ('[INFO] MEM TT Output')

path_MEM = '/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep_cross/' 
#with open(path_MEM+'TTmodel.json', "r") as json_model_file:
#    model_json_MEM_TT = json_model_file.read()
#model_MEM_TT = model_from_json(model_json_MEM_TT) # load model with best weights
#model_MEM_TT.load_weights(path_MEM+'TTweight.h5')

out_MEM_TT_sig = NNOutput(sig_MEM,path_MEM+'TT')
out_MEM_TT_back = NNOutput(back_MEM,path_MEM+'TT')

#out_MEM_TT_sig = model_MEM_TT.predict(sig_MEM)
#out_MEM_TT_back = model_MEM_TT.predict(back_MEM)

# Plot #
bins = np.linspace(0,40,100)
fig_MEM_TT = plt.figure(2)
plt.hist(out_MEM_TT_sig,bins=bins,color='g',alpha=0.7,label='Signal')
plt.hist(out_MEM_TT_back,bins=bins,color='r',alpha=0.7,label='Background')
plt.legend(loc='upper center')
plt.title('TT weight DNN Output')
plt.xlabel('DNN Output')
plt.ylabel('Occurences')
fig_MEM_TT.savefig(path_out+'MEMTTOut.png')
print ('Plot saved as '+path_out+'MEMTTOut.png')
plt.close()


###############################################################################
# DY MoMEMta Output #
###############################################################################
print ('='*80)
print ('[INFO] MEM DY Output')

#with open(path_MEM+'DYmodel.json', "r") as json_model_file:
#    model_json_MEM_DY = json_model_file.read()
#model_MEM_DY = model_from_json(model_json_MEM_DY) # load model with best weights
#model_MEM_DY.load_weights(path_MEM+'DYweight.h5')

out_MEM_DY_sig = NNOutput(sig_MEM,path_MEM+'DY')
out_MEM_DY_back = NNOutput(back_MEM,path_MEM+'DY')

#out_MEM_DY_sig = model_MEM_DY.predict(sig_MEM)
#out_MEM_DY_back = model_MEM_DY.predict(back_MEM)

# Plot #
bins = np.linspace(0,40,100)
fig_MEM_DY = plt.figure(2)
plt.hist(out_MEM_DY_sig,bins=bins,color='g',alpha=0.7,label='Signal')
plt.hist(out_MEM_DY_back,bins=bins,color='r',alpha=0.7,label='Background')
plt.legend(loc='upper center')
plt.title('DY weight DNN Output')
plt.xlabel('DNN Output')
plt.ylabel('Occurences')
fig_MEM_DY.savefig(path_out+'MEMDYOut.png')
print ('Plot saved as '+path_out+'MEMDYOut.png')
plt.close()


###############################################################################
# Build discriminant #
###############################################################################
print ('='*80)
print ('[INFO] Building discriminant')
# Import normalization #
norm = np.genfromtxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_pre_inv_sep/normalization.txt')

prob_MEM_TT_sig = np.zeros(out_MEM_TT_sig.shape[0])
prob_MEM_DY_sig = np.zeros(out_MEM_DY_sig.shape[0])
prob_MEM_TT_back = np.zeros(out_MEM_TT_back.shape[0])
prob_MEM_DY_back = np.zeros(out_MEM_DY_back.shape[0])

# Rescale MEM weights #
print ('[INFO] Going back to original signal weights')
for i in range(0,out_MEM_TT_sig.shape[0]):
    sys.stdout.write('\r\tProcessing : %0.f%%'%((i/out_MEM_TT_sig.shape[0])*100))
    sys.stdout.flush()
    prob_MEM_TT_sig[i] = math.pow(10,-out_MEM_TT_sig[i])*norm[0]
    prob_MEM_DY_sig[i] = math.pow(10,-out_MEM_DY_sig[i])*norm[1]

print ()
print ('[INFO] Going back to original background weights')
for i in range(0,out_MEM_TT_back.shape[0]):
    sys.stdout.write('\r\tProcessing : %0.f%%'%((i/out_MEM_TT_back.shape[0])*100))
    sys.stdout.flush()
    prob_MEM_TT_back[i] = math.pow(10,-out_MEM_TT_back[i])*norm[0]
    prob_MEM_DY_back[i] = math.pow(10,-out_MEM_DY_back[i])*norm[1]
print ()

# Discriminant #
dis_sig = np.divide(prob_MEM_TT_sig,np.add(prob_MEM_TT_sig,200*prob_MEM_DY_sig))
dis_back = np.divide(prob_MEM_TT_back,np.add(prob_MEM_TT_back,200*prob_MEM_DY_back))

# Plot #
bins = np.linspace(0,1,50)
fig_dis = plt.figure(1)
plt.hist(dis_sig,bins=bins,color='g',alpha=0.7,label='Signal')
plt.hist(dis_back,bins=bins,color='r',alpha=0.7,label='Background')
plt.legend(loc='upper center')
plt.title('Discriminant with DNN output')
plt.xlabel('Discriminant')
plt.ylabel('Occurences')
fig_dis.savefig(path_out+'Discriminant.png')
print ('Plot saved as '+path_out+'Discriminant.png')
plt.close()

###############################################################################
# Output First stage of Frankeinstein network #
###############################################################################
print ('='*80)
print ('[INFO] Output of First Stage')
# Input preparation #
input_sig = np.c_[out_MP_sig,dis_sig,sig_MP,out_MEM_TT_sig,out_MEM_DY_sig]
input_back = np.c_[out_MP_back,dis_back,back_MP,out_MEM_TT_back,out_MEM_DY_back]

target_sig = np.ones(input_sig.shape[0])
target_back = np.zeros(input_back.shape[0])

full_sig = np.c_[input_sig,target_sig,sig_weight]
full_back = np.c_[input_back,target_back,back_weight]
# [out_MP,dis_MEM,mH,mA,target,weight]

input_full = np.concatenate((full_sig,full_back),axis=0)

input_full.dtype = ([('out_MP','float64'),('dis_MEM','float64'),('m_lljj','float64'),('m_jj','float64'),('mH','float64'),('mA','float64'),('MEM_TT','float64'),('MEM_DY','float64'),('target','float64'),('weight','float64')])

array2root(input_full,path_out+'input.root',mode='recreate')
print ('[INFO] Saving output as '+path_out+'input.root')
