# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
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
from keras.models import model_from_json
from keras import losses
from keras.callbacks import EarlyStopping

# Personal Files #
from DrawMassPlane import DrawMassPlane

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Takes root files, build Neural Network and save it') 
parser.add_argument("-mA","--mA", help="Generated mass of A boson -> COMPULSORY")
parser.add_argument("-mH","--mH", help="Generated mass of H boson -> COMPULSORY")
parser.add_argument("-c","--cut", help="cut on the NN output (default : 0)",default=0.)

args = parser.parse_args()

mH_select = float(args.mH)
mA_select = float(args.mA)
cut_select = float(args.cut)
print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
print ('Cut applied : NN output < %f' %(cut_select))


############################################################################### 
# Extract features from Root Files #
############################################################################### 
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
print ('='*80)
print ('[INFO] Starting input from files')
back_set = np.zeros((0,2))
sig_set = np.zeros((0,4))

gen_choices = np.array(0,2) # records all the new configurations of (mH,mA)

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
        if np.isin(num[2:4],gen_choices)==False: # record new value of mH
            mH_choices = np.append(mH_choices,num[2])

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

# Assign random mH,mA to background #
back_mH = np.random.choice(mH_choices,size=back_set.shape[0]) 
back_mA = np.random.choice(mA_choices,size=back_set.shape[0]) 

back_set = np.c_[back_set,back_mH,back_mA]


############################################################################### 
# Load model #
############################################################################### 
print ('='*80)

path_model = '/home/ucl/cp3/fbury/Memoire/model_saved/[100 100 100 100]/'

json_file = open(path_model+'model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(path_model+'model.h5')
print("[INFO] Loaded model from disk")


############################################################################### 
# Apply model #
############################################################################### 
# Select signal sample #
mask = np.logical_and(sig_set[:,2]==mH_select,sig_set[:,3]==mA_select)
sig_select = sig_set[mask]
if sig_select.shape[0]==0:
    sys.exit('Wrong choice of mA,mH')

# Compute the output of the NN #
pred_select = model.predict(sig_select) 
pred_sig = model.predict(sig_set) 
pred_back = model.predict(back_set) 
pred_select_back = np.concatenate((pred_select,pred_back),axis=0)
pred_sig_back = np.concatenate((pred_sig,pred_back),axis=0)

# Pass to DrawMassPlane.py #
massplane_select = np.c_[sig_select[:,1],sig_select[:,0]]
massplane_sig = np.c_[sig_set[:,1],sig_set[:,0]]
massplane_select_back = np.concatenate((np.c_[sig_select[:,1],sig_select[:,0]],np.c_[back_set[:,1],back_set[:,0]]),axis=0)
massplane_sig_back = np.concatenate((np.c_[sig_set[:,1],sig_set[:,0]],np.c_[back_set[:,1],back_set[:,0]]),axis=0)
# Reversed because first input = X axis = m_jj

print ('\nMass plane for signal selection : mA = %0.f and mH = %0.f\n'%(mA_select,mH_select))
DrawMassPlane(data=massplane_select,mH=mH_select,mA=mA_select,title=('Mass plane for signal selection (mA = %0.f and mH = %0.f) and cut = %0.2f'%(mA_select,mH_select,cut_select)),prediction=pred_select,cut=cut_select)
print ('\nMass plane for all signal samples')
DrawMassPlane(data=massplane_sig,mH=mH_select,mA=mA_select,title=('Mass plane for all signal samples and cut = %0.2f'%(cut_select)),prediction=pred_sig,cut=cut_select)
print ('\nMass plane for signal selection : mA = %0.f and mH = %0.f + Background\n'%(mA_select,mH_select))
DrawMassPlane(data=massplane_select_back,mH=mH_select,mA=mA_select,title=('Mass plane for signal selection (mA = %0.f and mH = %0.f) + all backgrounds and cut = %0.2f'%(mA_select,mH_select,cut_select)),prediction=pred_select_back,cut=cut_select)
print ('\nMass plane for all signal samples + Background')
DrawMassPlane(data=massplane_sig_back,mH=mH_select,mA=mA_select,title=('Mass plane for all signal + background samples and cut = %0.2f'%(cut_select)),prediction=pred_sig_back,cut=cut_select)
