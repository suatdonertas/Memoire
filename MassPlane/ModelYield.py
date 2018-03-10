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

import matplotlib.pyplot as plt

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Evaluates perfomances of a given model, for each mA and mH specified')

parser.add_argument('-m','--model', help='Which model to use from the learning_model directory (format 10_10_10)-> COMPULSORY',required=True)
parser.add_argument('-mA','--mA', help='Generated mass of A boson',required=True)
parser.add_argument('-mH','--mH', help='Generated mass of H boson',required=True)
parser.add_argument('-cnn','--cutnn', help='Additional NN cut (in addition to 0.9)',required=False)
parser.add_argument('-ce','--cutell', help='Additional ellipse cut (in addition to 1 sigma)',required=False)

args = parser.parse_args()
 
mH_select = int(args.mH)
mA_select = int(args.mA)
print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
print ('Model used : '+str(args.model))
cut_NN = np.array([0,0.9])
cut_ell = np.array([100000,1.])
if args.cutnn is not None:
    cut_NN = np.append(cut_NN,float(args.cutnn))
if args.cutell is not None:
    cut_ell = np.append(cut_ell,float(args.cutell))
print ('Cuts on NN : ',cut_NN)
print ('Cuts on ellipse : ',cut_ell)

path_tree = '/home/ucl/cp3/fbury/storage/NNAndELLipseOutputTrees/model_'+str(args.model)+'/'

################################################################################
# Input Trees #
################################################################################
for name in glob.glob(path_tree+'*.root'):
    filename = name.replace(path_tree,'')
    num = [int(s) for s in re.findall('\d+',filename )]
    if num[0]!=mH_select or num[1]!=mA_select:
        continue

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    sig = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==0')
    DYToLL_0J = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==1')
    DYToLL_1J = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==2')
    DYToLL_2J = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==3')
    TT_Other = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==4')
    TTTo2L2Nu = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==5')

    for cn in cut_NN:
        print ('NN cut : ',cn)
        N_sig = np.sum(sig[sig[:]['NN_out']>cn]['weight'])
        N_DYToLL_0J = np.sum(DYToLL_0J[DYToLL_0J[:]['NN_out']>cn]['weight'])
        N_DYToLL_1J = np.sum(DYToLL_1J[DYToLL_1J[:]['NN_out']>cn]['weight'])
        N_DYToLL_2J = np.sum(DYToLL_2J[DYToLL_2J[:]['NN_out']>cn]['weight'])
        N_TT_Other = np.sum(TT_Other[TT_Other[:]['NN_out']>cn]['weight'])
        N_TTTo2L2Nu = np.sum(TTTo2L2Nu[TTTo2L2Nu[:]['NN_out']>cn]['weight'])

        print ('sig\t%0.f\t%0.5f'%(sig[sig[:]['NN_out']>cn].shape[0],N_sig))
        print ('DYToLL_0J\t%0.f\t%0.2f'%(DYToLL_0J[DYToLL_0J[:]['NN_out']>cn].shape[0],N_DYToLL_0J))
        print ('DYToLL_1J\t%0.f\t%0.2f'%(DYToLL_1J[DYToLL_1J[:]['NN_out']>cn].shape[0],N_DYToLL_1J))
        print ('DYToLL_2J\t%0.f\t%0.2f'%(DYToLL_2J[DYToLL_2J[:]['NN_out']>cn].shape[0],N_DYToLL_2J))
        print ('TT_Other\t%0.f\t%0.2f'%(TT_Other[TT_Other[:]['NN_out']>cn].shape[0],N_TT_Other))
        print ('TTTo2L2Nu\t%0.f\t%0.2f'%(TTTo2L2Nu[TTTo2L2Nu[:]['NN_out']>cn].shape[0],N_TTTo2L2Nu))

    for ce in cut_ell:
        print ('Ellipse cut : ',ce)

        N_sig = np.sum(sig[sig[:]['Ell_out']<ce]['weight'])
        N_DYToLL_0J = np.sum(DYToLL_0J[DYToLL_0J[:]['Ell_out']<ce]['weight'])
        N_DYToLL_1J = np.sum(DYToLL_1J[DYToLL_1J[:]['Ell_out']<ce]['weight'])
        N_DYToLL_2J = np.sum(DYToLL_2J[DYToLL_2J[:]['Ell_out']<ce]['weight'])
        N_TT_Other = np.sum(TT_Other[TT_Other[:]['Ell_out']<ce]['weight'])
        N_TTTo2L2Nu = np.sum(TTTo2L2Nu[TTTo2L2Nu[:]['Ell_out']<ce]['weight'])

        print ('sig\t%0.f\t%0.5f'%(sig[sig[:]['Ell_out']<ce].shape[0],N_sig))
        print ('DYToLL_0J\t%0.f\t%0.2f'%(DYToLL_0J[DYToLL_0J[:]['Ell_out']<ce].shape[0],N_DYToLL_0J))
        print ('DYToLL_1J\t%0.f\t%0.2f'%(DYToLL_1J[DYToLL_1J[:]['Ell_out']<ce].shape[0],N_DYToLL_1J))
        print ('DYToLL_2J\t%0.f\t%0.2f'%(DYToLL_2J[DYToLL_2J[:]['Ell_out']<ce].shape[0],N_DYToLL_2J))
        print ('TT_Other\t%0.f\t%0.2f'%(TT_Other[TT_Other[:]['Ell_out']<ce].shape[0],N_TT_Other))
        print ('TTTo2L2Nu\t%0.f\t%0.2f'%(TTTo2L2Nu[TTTo2L2Nu[:]['Ell_out']<ce].shape[0],N_TTTo2L2Nu))
    
