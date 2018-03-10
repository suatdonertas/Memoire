# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import json

from sklearn.metrics import roc_curve, auc

from scipy.optimize import newton
from scipy.stats import norm

import ROOT
from ROOT import TFile, TTree, TCanvas, TGraph2D
from ROOT import TPad, gPad, gStyle, TPaveText, gROOT
from root_numpy import tree2array,array2root

from keras.models import Model
from keras.models import model_from_json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoids boring message

import matplotlib.pyplot as plt

# Personal Files #
from NNOutput import NNOutput 
from cutWindow import massWindow
from ModelFunctions import*

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='generate output tress for given model and (mA,mH) configuration')

parser.add_argument('-m','--model', help='Which model to use from the learning_model directory (format 10_10_10)-> COMPULSORY',required=True)
parser.add_argument('-mA','--mA', help='Generated mass of A boson',required=True)
parser.add_argument('-mH','--mH', help='Generated mass of H boson',required=True)

args = parser.parse_args()
 
mH_select = int(args.mH)
mA_select = int(args.mA)
print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
print ('Model used : '+str(args.model))

path_tree = '/home/ucl/cp3/fbury/storage/NNAndELLipseOutputTrees/model_'+str(args.model)+'/'
path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_'+str(args.model)+'/'
############################################################################### 
#  Output Mode #
############################################################################### 
# Extract features from Root Files #
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/add_met_mll_forFlorian/slurm/output/'
print ('='*80)
print ('Starting input from files')

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)


# Get data from root files (sig+back)
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
    cs = f.Get('cross_section').GetVal()
    ws = f.Get('event_weight_sum').GetVal()
    
    selection = 'met_pt<80 && ll_M>70 && ll_M<110'
    jj_M = np.asarray(tree2array(t, branches='jj_M',selection=selection))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M',selection=selection))
    total_weight = np.asarray(tree2array(t, branches='total_weight', selection=selection))
    L = 35922 

    mH_array = np.ones(jj_M.shape[0])*mH_select
    mA_array = np.ones(jj_M.shape[0])*mA_select

    if Sig: #Signal
        # Check if correct file #
        num = [int(s) for s in re.findall('\d+',filename )]
        if num[2]!=mH_select or num[3]!=mA_select:
            continue
        else:
            print ('Our file')

        # Get the relative signal weights 
        cross_section = 0
        for c in range(0,xsec.shape[1]):
            if xsec[1,c]==num[2] and xsec[2,c]==num[3]:
                cross_section = xsec[0,c]
        if cross_section == 0:
            sys.exit('Could not find cross section in signal sample')
        print ('\tCross section = ',cross_section)

        relative_weight = cross_section/f.Get('event_weight_sum').GetVal()
        weight = L*(total_weight*relative_weight)#.reshape(-1,1)

        sig_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)
        sig_cs = cross_section 
        sig_ws = ws

    else: #Background
        relative_weight = f.Get('cross_section').GetVal()/f.Get('event_weight_sum').GetVal()
        weight = L*(total_weight*relative_weight)#.reshape(-1,1)
        if filename.startswith('DYToLL_0J'):
            DYToLL_0J_cs = cs
            DYToLL_0J_ws = ws
            DYToLL_0J_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)
        if filename.startswith('DYToLL_1J'):
            DYToLL_1J_cs = cs
            DYToLL_1J_ws = ws
            DYToLL_1J_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)
        if filename.startswith('DYToLL_2J'):
            DYToLL_2J_cs = cs
            DYToLL_2J_ws = ws
            DYToLL_2J_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)
        if filename.startswith('TT_Other'):
            TT_Other_cs = cs
            TT_Other_ws = ws
            TT_Other_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)
        if filename.startswith('TTTo2L2Nu'):
            TTTo2L2Nu_cs = cs
            TTTo2L2Nu_ws = ws
            TTTo2L2Nu_set = np.stack((lljj_M,jj_M,mH_array,mA_array,weight),axis=1)

if sig_cs == 0:
    sys.exit('Configuration not in data files')

################################################################################
# Output from NN and Ellipse #
################################################################################

print ('Starting NN Output')
print ('\tSignal')
z_sig = NNOutput(sig_set[:,0:4],path_model) 
print ('\tDYToLL_0J')
z_DYToLL_0J = NNOutput(DYToLL_0J_set[:,0:4],path_model)
print ('\tDYToLL_1J')
z_DYToLL_1J = NNOutput(DYToLL_1J_set[:,0:4],path_model)
print ('\tDYToLL_2J')
z_DYToLL_2J = NNOutput(DYToLL_2J_set[:,0:4],path_model)
print ('\tTT_Other')
z_TT_Other = NNOutput(TT_Other_set[:,0:4],path_model)
print ('\tTTTo2L2Nu')
z_TTTo2L2Nu = NNOutput(TTTo2L2Nu_set[:,0:4],path_model)

print ('Starting ellipse Output')
print ('\tSignal')
e_sig = EllipseOutput(sig_set[:,0:4])
print ('\tDYToLL_0J')
e_DYToLL_0J = EllipseOutput(DYToLL_0J_set[:,0:4])
print ('\tDYToLL_1J')
e_DYToLL_1J = EllipseOutput(DYToLL_1J_set[:,0:4])
print ('\tDYToLL_2J')
e_DYToLL_2J = EllipseOutput(DYToLL_2J_set[:,0:4])
print ('\tTT_Other')
e_TT_Other = EllipseOutput(TT_Other_set[:,0:4])
print ('\tTTTo2L2Nu')
e_TTTo2L2Nu = EllipseOutput(TTTo2L2Nu_set[:,0:4])


print ('Starting set concatenation')
id_sig = np.zeros(z_sig.shape[0])
id_DYToLL_0J = np.ones(z_DYToLL_0J.shape[0])*1
id_DYToLL_1J = np.ones(z_DYToLL_1J.shape[0])*2
id_DYToLL_2J = np.ones(z_DYToLL_2J.shape[0])*3
id_TT_Other = np.ones(z_TT_Other.shape[0])*4
id_TTTo2L2Nu = np.ones(z_TTTo2L2Nu.shape[0])*5

sig_cs_array = np.ones(z_sig.shape[0])*sig_cs
DYToLL_0J_cs_array = np.ones(z_DYToLL_0J.shape[0])*DYToLL_0J_cs
DYToLL_1J_cs_array = np.ones(z_DYToLL_1J.shape[0])*DYToLL_1J_cs
DYToLL_2J_cs_array = np.ones(z_DYToLL_2J.shape[0])*DYToLL_2J_cs
TT_Other_cs_array = np.ones(z_TT_Other.shape[0])*TT_Other_cs
TTTo2L2Nu_cs_array = np.ones(z_TTTo2L2Nu.shape[0])*TTTo2L2Nu_cs

sig_ws_array = np.ones(z_sig.shape[0])*sig_ws
DYToLL_0J_ws_array = np.ones(z_DYToLL_0J.shape[0])*DYToLL_0J_ws
DYToLL_1J_ws_array = np.ones(z_DYToLL_1J.shape[0])*DYToLL_1J_ws
DYToLL_2J_ws_array = np.ones(z_DYToLL_2J.shape[0])*DYToLL_2J_ws
TT_Other_ws_array = np.ones(z_TT_Other.shape[0])*TT_Other_ws
TTTo2L2Nu_ws_array = np.ones(z_TTTo2L2Nu.shape[0])*TTTo2L2Nu_ws

out_sig = np.c_[sig_set,z_sig,e_sig,sig_cs_array,sig_ws_array,id_sig]
out_DYToLL_0J = np.c_[DYToLL_0J_set,z_DYToLL_0J,e_DYToLL_0J,DYToLL_0J_cs_array,DYToLL_0J_ws_array,id_DYToLL_0J]
out_DYToLL_1J = np.c_[DYToLL_1J_set,z_DYToLL_1J,e_DYToLL_1J,DYToLL_1J_cs_array,DYToLL_1J_ws_array,id_DYToLL_1J]
out_DYToLL_2J = np.c_[DYToLL_2J_set,z_DYToLL_2J,e_DYToLL_2J,DYToLL_2J_cs_array,DYToLL_2J_ws_array,id_DYToLL_2J]
out_TT_Other = np.c_[TT_Other_set,z_TT_Other,e_TT_Other,TT_Other_cs_array,TT_Other_ws_array,id_TT_Other]
out_TTTo2L2Nu = np.c_[TTTo2L2Nu_set,z_TTTo2L2Nu,e_TTTo2L2Nu,TTTo2L2Nu_cs_array,TTTo2L2Nu_ws_array,id_TTTo2L2Nu]

out = np.concatenate((out_sig,out_DYToLL_0J,out_DYToLL_1J,out_DYToLL_2J,out_TT_Other,out_TTTo2L2Nu),axis=0)
out.dtype = [('mlljj','float64'),('mjj','float64'),('mH','float64'),('mA','float64'),('weight','float64'),('NN_out','float64'),('Ell_out','float64'),('cross_section','float64'),('weight_sum','float64'),('id','float64')]
out.dtype.names = ['mlljj','mjj','mH','mA','weight','NN_out','Ell_out','cross_section','weight_sum','id']

# Save in tree #
if not os.path.exists(path_tree):
    os.makedirs(path_tree)
name = path_tree+'mH_'+str(mH_select)+'_mA_'+str(mA_select)+'.root'
array2root(out,name, mode='recreate')

print ('Output root file : '+name)

