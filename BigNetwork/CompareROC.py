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
from sklearn.metrics import mean_squared_error as mse

from scipy.optimize import newton
from scipy.stats import norm

import ROOT
from ROOT import TFile, TTree, TCanvas, TGraph2D
from ROOT import TPad, gPad, gStyle, TPaveText, gROOT
from root_numpy import tree2array,array2root

import matplotlib.pyplot as plt

# Personal libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
import ModelFunctions

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Evaluates perfomances of a given model, for each mA and mH specified')

parser.add_argument('-mA','--mA', help='Mass of A boson',required=True,type=str)
parser.add_argument('-mH','--mH', help='Mass of H boson',required=True,type=str)
parser.add_argument('-mn','--model_notrain', help='Model with weights but no retraining',required=True,type=str)
parser.add_argument('-mr','--model_retrain', help='Model with weights and retraining',required=True,type=str)

args = parser.parse_args()
 
path_tree_notrain = '/home/ucl/cp3/fbury/storage/BigNetworkOutput/'+args.model_notrain+'/output.root'
path_tree_retrain = '/home/ucl/cp3/fbury/storage/PlugBigNetwork/'+args.model_retrain+'/output.root'
path_plot = '/home/ucl/cp3/fbury/Memoire/BigNetwork/ROC_Plug/'+args.model_notrain+'_'+args.model_retrain+'/'
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

print ('Model no train : '+path_tree_notrain)
print ('Model retrain : '+path_tree_retrain)
print ('mH : '+args.mH)
print ('mA : '+args.mA)
mH = args.mH
mA = args.mA
################################################################################
# Get trees #
################################################################################
f1 = ROOT.TFile.Open(path_tree_notrain)
t1 = f1.Get("tree")
f2 = ROOT.TFile.Open(path_tree_retrain)
t2 = f2.Get("tree")

out1 = tree2array(t1,branches=['output','target','mA','mH','weight','MP_discriminant','MEM_discriminant','MEM_TT','MEM_DY'],selection='mA=='+str(mA)+' && mH=='+str(mH))
#out2 = tree2array(t2,branches=['output_MP','output_MEM_TT','output_MEM_DY','discriminant','output_last','target','weight','mlljj','mjj','output_mlljj','output_mjj'],selection='mA=='+str(mA)+' && mH=='+str(mH))

out2 = tree2array(t2,branches=['output_MP','output_MEM_TT','output_MEM_DY','discriminant','output_last','target','weight','mlljj','mjj','mA','mH'],selection='mA=='+str(mA)+' && mH=='+str(mH))

################################################################################
# Output comparison #
################################################################################
# Defaults #
SMALL_SIZE = 10
MEDIUM_SIZE = 12 
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Output #
fig1 = plt.figure(1,figsize=(6,10))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
bins = np.linspace(0,1,50)
fig1.tight_layout()
fig1.subplots_adjust(left=0.15, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=0.5)
fig1.suptitle('Output comparison : $m_H$ = '+mH+' GeV, $m_A$ = '+mA+' GeV')
ax1.hist(out1['MP_discriminant'][out1['target']==1],bins=bins,color='g',alpha=0.7,label='Signal (no train)')
ax1.hist(out2['output_MP'][out2['target']==1],bins=bins,color='b',alpha=0.7,label='Signal (retrain)')
ax1.hist(out1['MP_discriminant'][out1['target']==0],bins=bins,color='r',alpha=0.7,label='Background (no train)')
ax1.hist(out2['output_MP'][out2['target']==0],bins=bins,color='yellow',alpha=0.7,label='Background (retrain)')
ax2.hist(out1['MEM_discriminant'][out1['target']==1],bins=bins,color='g',alpha=0.7,label='Signal (no train)')
ax2.hist(out2['discriminant'][out2['target']==1],bins=bins,color='b',alpha=0.7,label='Signal (retrain)')
ax2.hist(out1['MEM_discriminant'][out1['target']==0],bins=bins,color='r',alpha=0.7,label='Background (no train)')
ax2.hist(out2['discriminant'][out2['target']==0],bins=bins,color='yellow',alpha=0.7,label='Background (retrain)')
ax3.hist(out1['output'][out1['target']==1],bins=bins,color='g',alpha=0.7,label='Signal (no retrain)')
ax3.hist(out2['output_last'][out2['target']==1],bins=bins,color='b',alpha=0.7,label='Signal (retrain)')
ax3.hist(out1['output'][out1['target']==0],bins=bins,color='r',alpha=0.7,label='Background (no retrain)')
ax3.hist(out2['output_last'][out2['target']==0],bins=bins,color='yellow',alpha=0.7,label='Background (retrain)')
ax1.set_title('Mass Plane network')
ax2.set_title('MoMEMta network')
ax3.set_title('Frankenstein network')
#ax1.set_yscale('log')
#ax2.set_yscale('log')
#ax3.set_yscale('log')
ax1.set_xlabel('NN Output')
ax2.set_xlabel('NN Output')
ax3.set_xlabel('NN Output')
ax1.set_ylabel('Occurences')
ax2.set_ylabel('Occurences')
ax3.set_ylabel('Occurences')
ax1.legend(loc='upper center')
ax2.legend(loc='upper center')
ax3.legend(loc='upper center')
#plt.show()
fig1.savefig(path_plot+'Output_mH_'+mH+'_mA_'+mA+'.png')
print ('[INFO] Fig saved as '+path_plot+'Output_mH_'+mH+'_mA_'+mA+'.png')
plt.close()

################################################################################
# Invariant masses Comparison #
################################################################################
try: 
    fig5 = plt.figure(5,figsize=(10,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    bins_jj = np.linspace(0,int(mA)*3,50)
    bins_lljj = np.linspace(0,int(mH)*3,50)
    fig5.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85, wspace=0.3, hspace=0.3)
    fig5.suptitle('Invariant masses comparison : $m_H$ = '+mH+' GeV, $m_A$ = '+mA+' GeV')
    ax1.hist(out2['mjj'],bins=bins_jj,color='g',alpha=0.7,label='Before retrain')
    ax1.hist(out2['output_mjj'],bins=bins_jj,color='r',alpha=0.7,label='After retrain')
    ax2.hist(out2['mlljj'],bins=bins_lljj,color='g',alpha=0.7,label='Before retrain')
    ax2.hist(out2['output_mlljj'],bins=bins_lljj,color='r',alpha=0.7,label='After retrain')
    ax1.set_title('$M_{jj}$ comparison')
    ax2.set_title('$M_{lljj}$ comparison')
    ax1.set_xlabel('$M_{jj}$')
    ax2.set_xlabel('$M_{lljj}$')
    ax1.set_ylabel('Occurences')
    ax2.set_ylabel('Occurences')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    fig5.savefig(path_plot+'Masses_mH_'+mH+'_mA_'+mA+'.png')
    print ('[INFO] Fig saved as '+path_plot+'Masses_mH_'+mH+'_mA_'+mA+'.png')
except:
    print ("No invariant masses output")

################################################################################
# MEM weights comparison #
################################################################################
# Plot #
fig2 = plt.figure(2,figsize=(10,7))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
bins_TT = np.linspace(0,20,50)
bins_DY = np.linspace(0,20,50)
fig2.tight_layout()
fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
fig2.suptitle('MEM weights comparison : $m_H$ = '+mH+' GeV, $m_A$ = '+mA+' GeV')
ax1.hist(out1['MEM_TT'][out1['target']==1],bins=bins_TT,color='g',alpha=0.7,label='Before retrain')
ax1.hist(out2['output_MEM_TT'][out2['target']==1],bins=bins_TT,color='r',alpha=0.7,label='After retrain')
ax2.hist(out1['MEM_TT'][out1['target']==0],bins=bins_TT,color='g',alpha=0.7,label='Before retrain')
ax2.hist(out2['output_MEM_TT'][out2['target']==0],bins=bins_TT,color='r',alpha=0.7,label='After retrain')
ax3.hist(out1['MEM_DY'][out1['target']==1],bins=bins_DY,color='g',alpha=0.7,label='Before retrain')
ax3.hist(out2['output_MEM_DY'][out2['target']==1],bins=bins_DY,color='r',alpha=0.7,label='After retrain')
ax4.hist(out1['MEM_DY'][out1['target']==0],bins=bins_DY,color='g',alpha=0.7,label='Before retrain')
ax4.hist(out2['output_MEM_DY'][out2['target']==0],bins=bins_DY,color='r',alpha=0.7,label='After retrain')
ax1.set_title('DNN output of TT weight : Signal')
ax2.set_title('DNN output of TT weight : Background')
ax3.set_title('DNN output of DY weight : Signal')
ax4.set_title('DNN output of DY weight : Background')
ax1.set_xlabel('$-log_{10}(weight) [normalized]$')
ax2.set_xlabel('$-log_{10}(weight) [normalized]$')
ax3.set_xlabel('$-log_{10}(weight) [normalized]$')
ax4.set_xlabel('$-log_{10}(weight) [normalized]$')
ax1.set_ylabel('Occurences')
ax2.set_ylabel('Occurences')
ax3.set_ylabel('Occurences')
ax4.set_ylabel('Occurences')
#ax1.set_ylim(bottom=0,top=100)
#ax2.set_ylim(bottom=0,top=100)
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
fig2.savefig(path_plot+'MEM_mH_'+mH+'_mA_'+mA+'.png')
print ('[INFO] Fig saved as '+path_plot+'MEM_mH_'+mH+'_mA_'+mA+'.png')

################################################################################
# Roc Curves #
################################################################################
back_eff_F,sig_eff_F,tresholds = roc_curve(out1['target'],out1['output'],sample_weight=out1['weight'])
back_eff_MP,sig_eff_MP,tresholds = roc_curve(out1['target'],out1['MP_discriminant'],sample_weight=out1['weight'])
back_eff_dis,sig_eff_dis,tresholds = roc_curve(1-out1['target'],out1['MEM_discriminant'],sample_weight=out1['weight'])
back_eff_dis_re,sig_eff_dis_re,tresholds = roc_curve(1-out2['target'],out2['discriminant'],sample_weight=out2['weight'])
back_eff_F_re,sig_eff_F_re,tresholds = roc_curve(out2['target'],out2['output_last'],sample_weight=out2['weight'])
back_eff_MP_re,sig_eff_MP_re,tresholds = roc_curve(out2['target'],out2['output_MP'],sample_weight=out2['weight'])

# Remove overflows # 
#sig_eff_F = sig_eff_F[back_eff_F>=0]
#back_eff_F = back_eff_F[back_eff_F>=0]
#sig_eff_MP = sig_eff_MP[back_eff_MP>=0]
#back_eff_MP = back_eff_MP[back_eff_MP>=0]
#sig_eff_dis = sig_eff_dis[back_eff_dis>=0]
#back_eff_dis = back_eff_dis[back_eff_dis>=0]
#sig_eff_F_re = sig_eff_F_re[back_eff_F_re>=0]
#back_eff_F_re = back_eff_F_re[back_eff_F_re>=0]
#sig_eff_MP_re = sig_eff_MP_re[back_eff_MP_re>=0]
#back_eff_MP_re = back_eff_MP_re[back_eff_MP_re>=0]
#sig_eff_dis_re = sig_eff_dis_re[back_eff_dis_re>=0]
#back_eff_dis_re = back_eff_dis_re[back_eff_dis_re>=0]

# AUC score #
roc_auc_F = 1-auc(sig_eff_F,back_eff_F)
roc_auc_MP = 1-auc(sig_eff_MP,back_eff_MP)
roc_auc_dis = 1-auc(sig_eff_dis,back_eff_dis,reorder='True')
roc_auc_F_re = 1-auc(sig_eff_F_re,back_eff_F_re)
roc_auc_MP_re = 1-auc(sig_eff_MP_re,back_eff_MP_re)
roc_auc_dis_re = 1-auc(sig_eff_dis_re,back_eff_dis_re,reorder='True')
mse_F = mse(out1['target'],out1['output'],sample_weight=out1['weight'])
mse_MP = mse(out1['target'],out1['MP_discriminant'],sample_weight=out1['weight'])
mse_dis = mse(1-out1['target'],out1['MEM_discriminant'],sample_weight=out1['weight'])
mse_F_re = mse(out2['target'],out2['output_last'],sample_weight=out2['weight'])
mse_MP_re = mse(out2['target'],out2['output_MP'],sample_weight=out2['weight'])
mse_dis_re = mse(1-out2['target'],out2['discriminant'],sample_weight=out2['weight'])

#roc_auc_MEM = roc_auc_score(1-out['target'],out['MEM_discriminant'],sample_weight=out['weight'])

# ROC curves plots #
fig3 = plt.figure(3,figsize=(6,10))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
fig3.tight_layout()
fig3.subplots_adjust(left=0.15, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=0.5)
fig3.suptitle('ROC curve comparison : $m_H$ = '+mH+' GeV, $m_A$ = '+mA+' GeV',fontsize=14)
ax1.plot(sig_eff_MP, back_eff_MP,'g', label=('Before retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_MP,mse_MP)))
ax1.plot(sig_eff_MP_re, back_eff_MP_re,'r', label=('After retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_MP_re,mse_MP_re)))
#ax1.plot(0,0,color='w',label=('Gain in AUC : %0.2f%%\nGain in MSE : %0.2f%%'%(((roc_auc_MP_re-roc_auc_MP)/roc_auc_MP*100),((mse_MP_re-mse_MP)/mse_MP*100))))
ax2.plot(sig_eff_dis, back_eff_dis,'g', label=('Before retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_dis,mse_dis)))
ax2.plot(sig_eff_dis_re, back_eff_dis_re,'r', label=('After retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_dis_re,mse_dis_re)))
#ax2.plot(0,0,color='w',label=('Gain in AUC : %0.2f%%\nGain in MSE : %0.2f%%'%(((roc_auc_dis_re-roc_auc_dis)/roc_auc_dis*100),((mse_dis_re-mse_dis)/mse_dis*100))))
ax3.plot(sig_eff_F, back_eff_F,'g', label=('Before retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_F,mse_F)))
ax3.plot(sig_eff_F_re, back_eff_F_re,'r', label=('After retrain : \nAUC = %0.5f\nMSE = %0.5f'%(roc_auc_F_re,mse_F_re)))
#ax3.plot(0,0,color='w',label=('Gain in AUC : %0.2f%%\nGain in MSE : %0.2f%%'%(((roc_auc_F_re-roc_auc_F)/roc_auc_F*100),((mse_F_re-mse_F)/mse_F*100))))
ax1.set_title('Mass Plane DNN Output')
ax2.set_title('MoMEMta Discriminant')
ax3.set_title('Frankenstein DNN Output')
#ax1.plot(sig_eff_MEM, back_eff_MEM,'r', label=('MoMEMta DNN Output : AUC = %0.5f'%(roc_auc_MEM)))
#ax1.plot(0,0,color='w',label=('Gain in AUC : %0.2f%%\nGain in MSE : %0.2f%%'%(((roc_auc_F-roc_auc_MP)/roc_auc_MP*100),((mse_F-mse_MP)/mse_MP*100))))
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax1.set_ylim([0,1])
ax2.set_ylim([0,1])
ax3.set_ylim([0,1])
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
ax3.set_xlim([0,1])
ax1.set_yscale('symlog',linthreshy=0.01)
ax2.set_yscale('symlog',linthreshy=0.01)
ax3.set_yscale('symlog',linthreshy=0.01)
ax1.set_xlabel('Signal Efficiency')
ax2.set_xlabel('Signal Efficiency')
ax3.set_xlabel('Signal Efficiency')
ax1.set_ylabel('Background Efficiency')
ax2.set_ylabel('Background Efficiency')
ax3.set_ylabel('Background Efficiency')
#plt.show()
fig3.savefig(path_plot+'ROC_mH_'+mH+'_mA_'+mA+'.png')
print ('[INFO] Figure saved at : '+path_plot+'ROC_mH_'+mH+'_mA_'+mA+'.png')
plt.close()

################################################################################
# Full comparison #
################################################################################
# Process Ellipse output #
try:  # not all signal samples have corresponding ellipse configuration
    ell_sig_input = np.column_stack((out2['mlljj'][out2['target']==1],out2['mjj'][out2['target']==1],out2['mH'][out2['target']==1],out2['mA'][out2['target']==1]))
    ell_back_input = np.column_stack((out2['mlljj'][out2['target']==0],out2['mjj'][out2['target']==0],out2['mH'][out2['target']==0],out2['mA'][out2['target']==0]))

    ell_sig_output = ModelFunctions.EllipseOutput(ell_sig_input)
    ell_back_output = ModelFunctions.EllipseOutput(ell_back_input)

    ell_output = np.concatenate((ell_sig_output,ell_back_output),axis=0)
    ell_target = np.concatenate((np.ones(ell_sig_output.shape[0]),np.zeros(ell_back_output.shape[0])),axis=0)
    ell_weight = np.concatenate((out2['weight'][out2['target']==1],out2['weight'][out2['target']==0]),axis=0)

    ell_output = 1-(ell_output/np.max(ell_output))

    back_eff_e,sig_eff_e,tresholds = roc_curve(ell_target,ell_output,sample_weight=ell_weight)
    roc_auc_ell = 1-auc(sig_eff_e,back_eff_e)

except:
    print ('No ellipse configuration')

# Plot #
fig4 = plt.figure(4)
fig4.tight_layout()
fig4.subplots_adjust(left=0.15, bottom=0.15, right=None, top=0.9, wspace=0.2, hspace=0.5)
fig4.suptitle('ROC curve comparison : $m_H$ = '+mH+' GeV, $m_A$ = '+mA+' GeV',fontsize=14)
try:    
    plt.plot(sig_eff_e,back_eff_e,color='gold',label='Ellipse output')
    plt.plot(0,0,color='w',label=('AUC = %0.5f'%(roc_auc_ell)))
except:
    pass
plt.plot(sig_eff_MP,back_eff_MP,color='b',label='Mass plane DNN output')
plt.plot(0,0,color='w',label=('AUC = %0.5f'%(roc_auc_MP)))
plt.plot(sig_eff_F,back_eff_F,color='g',label='Frankenstein DNN output (before retrain)')
plt.plot(0,0,color='w',label=('AUC = %0.5f'%(roc_auc_F)))
plt.plot(sig_eff_F_re,back_eff_F_re,color='r',label='Frankenstein DNN output (after retrain)')
plt.plot(0,0,color='w',label=('AUC = %0.5f'%(roc_auc_F_re)))
plt.legend(loc='upper left')
plt.grid(True)
plt.ylim([0,1])
plt.xlim([0,1])
plt.yscale('symlog',linthreshy=0.1)
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Efficiency')
fig4.savefig(path_plot+'ROC_sup_mH_'+mH+'_mA_'+mA+'.png')
print ('[INFO] Figure saved at : '+path_plot+'ROC_sup_mH_'+mH+'_mA_'+mA+'.png')
plt.close()
