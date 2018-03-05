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

# Personal Files #
from ModelFunctions import significance,NNOperatingPoint,EllipseOperatingPoint

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Evaluates perfomances of a given model, for each mA and mH specified')

parser.add_argument('-m','--model', help='Which model to use from the learning_model directory (format 10_10_10)-> COMPULSORY',required=True)

args = parser.parse_args()
 
print ('Model used : '+str(args.model))

path_tree = '/home/ucl/cp3/fbury/storage/NNAndELLipseOutputTrees/model_'+str(args.model)+'/'
path_plots = '/home/ucl/cp3/fbury/Memoire/MassPlane/graph_ROC/model_test_'+str(args.model)+'/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)
################################################################################
# Looping over trees #
################################################################################
sigma_ell = np.array([0.1,0.5,1,2,3])
cut_NN = np.array([0.99,0.95,0.9,0.8,0.5])

for name in glob.glob(path_tree+'*.root'):
    filename = name.replace(path_tree,'')
    num = [int(s) for s in re.findall('\d+',filename )]
    mH = num[0]
    mA = num[1]
    print ('Configuration : mH = %0.f, mA = %0.f'%(mH,mA))

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    sig = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id==0')
    back = tree2array(t,branches=['NN_out','Ell_out','weight'],selection='id!=0')

    S = sig[:]['weight'].shape[0]
    B = back[:]['weight'].shape[0]
    S_norm = np.sum(sig[:]['weight'])
    B_norm = np.sum(back[:]['weight'])
    print ('Output for signal, size = %0.f, normalised : %0.5f'%(S,S_norm))
    print ('Output for background, size = %0.f, normalised : %0.5f'%(B,B_norm))

    # NN and Ellipse Output #
    target_sig = np.ones(S)
    target_back = np.zeros(B)
    weight = np.concatenate((sig[:]['weight'],back[:]['weight']),axis=0)
    target = np.concatenate((target_sig,target_back),axis=0) 
    out_NN = np.concatenate((sig[:]['NN_out'],back[:]['NN_out']),axis=0)
    out_Ell = np.concatenate((sig[:]['Ell_out'],back[:]['Ell_out']),axis=0)
    out_Ell = 1-(out_Ell/np.max(out_Ell))

    # ROC evaluation #
    back_eff_z,sig_eff_z,tresholds = roc_curve(target,out_NN,sample_weight=weight)
    back_eff_e,sig_eff_e,tresholds = roc_curve(target,out_Ell,sample_weight=weight)

    roc_auc_z = roc_auc_score(target,out_NN,sample_weight=weight)
    roc_auc_e = roc_auc_score(target,out_Ell,sample_weight=weight)

    # Significance estimation #
    Z_NN = np.array([])
    Z_ell = np.array([])
    sig_frac_z = sig_eff_z*S_norm
    sig_frac_e = sig_eff_e*S_norm
    back_frac_z = back_eff_z*S_norm
    back_frac_e = back_eff_e*S_norm
    Z_NN = significance(sig_frac_z,back_frac_z)
    Z_ell = significance(sig_frac_e,back_frac_e)

    # Operating Point #
    op_sig_NN = NNOperatingPoint(sig[:]['NN_out'],sig[:]['weight'],cut_NN)
    op_back_NN = NNOperatingPoint(back[:]['NN_out'],back[:]['weight'],cut_NN)

    op_sig_Ell = EllipseOperatingPoint(sig[:]['Ell_out'],sig[:]['weight'],sigma_ell)
    op_back_Ell = EllipseOperatingPoint(back[:]['Ell_out'],back[:]['weight'],sigma_ell)

    # Plot Section #
    fig1 = plt.figure(1,figsize=(10,5))
    ax1 = plt.subplot(111)
    plt.title('ROC Curve : $m_H$ = %0.f, $m_A$ = %0.f'%(mH,mA))
    ax1.plot(sig_eff_z, back_eff_z,'b-', label=('NN Output : AUC = %0.5f'%(roc_auc_z)))
    ax1.plot(sig_eff_e, back_eff_e, 'g-', label=('Ellipse Output : AUC = %0.5f'%(roc_auc_e)))
    ax2 = ax1.twinx()
    ax2.plot(sig_eff_z,Z_NN, 'b--',label='NN Significance')
    ax2.plot(sig_eff_e,Z_ell,'g--',label='Ellipse Significance')
    for s in range(0,sigma_ell.shape[0]):
        gc = s*(1/(sigma_ell.shape[0]+1))
        ax1.plot(op_sig_Ell[s],op_back_Ell[s],'k*',markersize=10,color=str(gc),label='Ellipse operating point for $\sigma$ = %0.1f'%(sigma_ell[s]))
    for s in range(0,cut_NN.shape[0]):
        gc = s*(1/(cut_NN.shape[0]+1))
        ax1.plot(op_sig_NN[s],op_back_NN[s],'kX',markersize=10,color=str(gc),label='NN operating point for cut = %0.2f'%(cut_NN[s]))
    ax1.grid(True)
    ax1.set_yscale('symlog',linthreshy=0.0001)
    ax1.set_xlabel('Signal Efficiency')
    ax1.set_ylabel('Background Efficiency')
    ax2.set_ylabel('Significance [Number of sigmas]')
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,np.max([np.max(Z_NN),np.max(Z_ell)])])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.6, box.height])
    ax2.set_position([box.x0, box.y0, box.width*0.6, box.height])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1[0:2]+lines2+lines1[2:],labels1[0:2]+labels2+labels1[2:],loc='upper left',bbox_to_anchor=(1.1, 1),fancybox=True, shadow=True,labelspacing=0.8)
    #plt.show()
    fig1.savefig(path_plots+'mH_'+str(mH)+'_mA_'+str(mA)+'.png')
    print ('[INFO] ROC curve plot saved as : '+str(mH)+'_mA_'+str(mA)+'.png')
    plt.close()


