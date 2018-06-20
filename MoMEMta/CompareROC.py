# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import json

from sklearn.metrics import roc_curve, auc, roc_auc_score

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

parser.add_argument('-f','--file', help='Name of the file',required=True,type=str)
parser.add_argument('-m','--model', help='Model with weights',required=True,type=str)

args = parser.parse_args()
 
path_tree4 = '/home/ucl/cp3/fbury/storage/NNAndELLipseOutputTrees/model_30_30_30_l2/'
path_tree6 = '/home/ucl/cp3/fbury/storage/NNAndELLipseOutputTrees/'+args.model+'/'
path_plot = '/home/ucl/cp3/fbury/Memoire/MoMEMta/ROC_withweights/'+args.model+'/'
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

print ('Starting on file : ',args.file)
num = [int(s) for s in re.findall('\d+',args.file )]
mH = num[0]
mA = num[1]
################################################################################
# Get trees #
################################################################################
f4 = ROOT.TFile.Open(path_tree4+args.file)
f6 = ROOT.TFile.Open(path_tree6+args.file)
t4 = f4.Get("tree")
t6 = f6.Get("tree")

sig4 = tree2array(t4,branches=['NN_out','Ell_out','weight'],selection='id==0')
sig6 = tree2array(t6,branches=['NN_out','Ell_out','weight'],selection='id==0')
back4 = tree2array(t4,branches=['NN_out','Ell_out','weight'],selection='id!=0')
back6 = tree2array(t6,branches=['NN_out','Ell_out','weight'],selection='id!=0')
print (sig6['NN_out'])
print (sig6['NN_out'].mean())
print (sig4['NN_out'])
print (sig4['NN_out'].mean())

target_sig4 = np.ones(sig4[:]['weight'].shape[0])
target_sig6 = np.ones(sig6[:]['weight'].shape[0])
target_back4 = np.zeros(back4[:]['weight'].shape[0])
target_back6 = np.zeros(back6[:]['weight'].shape[0])

target4 = np.concatenate((target_sig4,target_back4),axis=0)
out_NN4 = np.concatenate((sig4[:]['NN_out'],back4[:]['NN_out']),axis=0)
out_Ell4 = np.concatenate((sig4[:]['Ell_out'],back4[:]['Ell_out']),axis=0)
weight4 = np.concatenate((sig4[:]['weight'],back4[:]['weight']),axis=0)
target6 = np.concatenate((target_sig6,target_back6),axis=0)
out_NN6 = np.concatenate((sig6[:]['NN_out'],back6[:]['NN_out']),axis=0)
#out_Ell6 = np.concatenate((sig6[:]['Ell_out'],back6[:]['Ell_out']),axis=0)
weight6 = np.concatenate((sig6[:]['weight'],back6[:]['weight']),axis=0)

out_Ell4 = 1-(out_Ell4/np.max(out_Ell4))
#out_Ell6 = 1-(out_Ell6/np.max(out_Ell6))
################################################################################
# Roc Curves #
################################################################################
back_eff_z4,sig_eff_z4,tresholds = roc_curve(target4,out_NN4,sample_weight=weight4)
back_eff_z6,sig_eff_z6,tresholds = roc_curve(target6,out_NN6,sample_weight=weight6)
back_eff_e4,sig_eff_e4,tresholds = roc_curve(target4,out_Ell4,sample_weight=weight4)
#back_eff_e6,sig_eff_e6,tresholds = roc_curve(target6,out_Ell6,sample_weight=weight6)

roc_auc_z4 = roc_auc_score(target4,out_NN4,sample_weight=weight4)
roc_auc_e4 = roc_auc_score(target4,out_Ell4,sample_weight=weight4)
roc_auc_z6 = roc_auc_score(target6,out_NN6,sample_weight=weight6)
#roc_auc_e6 = roc_auc_score(target6,out_Ell6,sample_weight=weight6)

fig1 = plt.figure(1)
ax1 = plt.subplot(111)
ax1.plot(sig_eff_z4, back_eff_z4,'b-', label=('NN Output (without weights) : AUC = %0.5f'%(roc_auc_z4)))
ax1.plot(sig_eff_z6, back_eff_z6,'r-', label=('NN Output (with weights) : AUC = %0.5f'%(roc_auc_z6)))
ax1.plot(sig_eff_e4, back_eff_e4, 'g-', label=('Ellipse Output : AUC = %0.5f'%(roc_auc_e4)))
plt.legend(loc='upper left')
ax1.grid(True)
ax1.set_ylim([0,1])
ax1.set_yscale('symlog',linthreshy=0.0001)
plt.title('ROC Curve : $m_H$ = %0.f GeV, $m_A$ = %0.f GeV'%(mH,mA))
ax1.set_xlabel('Signal Efficiency')
ax1.set_ylabel('Background Efficiency')
plt.show()
name = args.file.replace('.root','')
fig1.savefig(path_plot+'ROC_'+name+'.png', bbox_inches='tight')
print ('\tFigure saved at : '+path_plot+'ROC_'+name+'.png')
