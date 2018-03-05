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
from root_numpy import tree2array

from keras.models import Model
from keras.models import model_from_json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoids boring message

import matplotlib.pyplot as plt

# Personal Files #
from NNOutput import NNOutput 
from cutWindow import massWindow

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Evaluates perfomances of a given model, for eaxh mA and mH specified')

parser.add_argument('-m','--model', help='Which model to use from the learning_model directory (format 10_10_10)-> COMPULSORY',required=True)
parser.add_argument('-a','--all_plots', help='Wether to plot the massplanes for all (mH,mA) configurations : yes (default) or no (then specifiy mA and mH)',default='yes',required=False)
parser.add_argument('-mA','--mA', help='Generated mass of A boson',required=False)
parser.add_argument('-mH','--mH', help='Generated mass of H boson',required=False)

args = parser.parse_args()
 
if args.all_plots == 'no':
    mH_select = float(args.mH)
    mA_select = float(args.mA)
    print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
if args.all_plots == 'yes':
    print ('Using all configurations')
print ('Model used : '+str(args.model))

############################################################################### 
# Ellpise model #
############################################################################### 
print ('='*80)
print ('[INFO] Extracting ellipses configuration from file')
# Extract ellipses configuration file #
path_json = '/home/ucl/cp3/fbury/Memoire/MassPlane/graph_ROC/ellipseParam.json'

with open(path_json) as json_file:
    data_json = json.load(json_file)
ellipse_conf = np.zeros((len(data_json),len(data_json[0])))

for idx, val in enumerate(data_json):
    ellipse_conf[idx,:] = val
# ellipse_conf => [mbb_reco, mllbb_reco, a, b, theta, MA_sim, MH_sim]
gen_ellipse = np.c_[ellipse_conf[:,6],ellipse_conf[:,5]] #all (mH,mA) config in json file

def getEllipseConf(mA_c,mH_c,ellipse):
    """Given mA_C, mH_c, returns a,b,theta from ellipse"""
    for (mbb, mllbb, a, b, theta, mA, mH) in ellipse:
        if mA_c == mA and mH_c == mH:
            return mbb,mllbb,a,b,theta
    sys.exit('Could not find config with mH = %0.f and mA = %0.f'%(mH_c,mA_c))

def EllipseOutput(data,sigma):
    """ 
        Returns the number of sigma of deviation from ellipse center
        Also returns fraction of points inside for given sigma (array)
    """
    # Loop over data points #
    out = np.ones(data.shape[0])*-1 
    points_kept = np.zeros(sigma.shape[0]) 
    for i in range(0,data.shape[0]):
        # Progress #
        if i/data.shape[0]*100%10==0:
            print ('\tProcessing ellipse : %0.f%%'%((i/data.shape[0])*100))

        # Get ellipse configuration #
        x,y,a,b,t = getEllipseConf(data[i,3],data[i,2],ellipse_conf)
        center = [x,y]
        masspoint = [data[i,1],data[i,0]] # (mjj,mlljj)
        instance = massWindow(filename=path_json)
        out[i] = instance.returnSigma(center=center,massPoint=masspoint)
        for idx,val in enumerate(sigma):
            inside = instance.isInWindow(center=center,size=val,massPoint=masspoint)
            points_kept[idx] += inside*1  # true => 1, false => 0
    frac = points_kept/data.shape[0]
    #print (sigma,' sample number = %0.f, cut number = %0.f'%(data.shape[0],frac*data.shape[0]))
    return out,frac
############################################################################### 
# Significance and normalization definition #
############################################################################### 
def significance(S,B):
    if S==0 or B==0 :
        Z = 0
    else:
        Z = math.sqrt(2*((S+B)*math.log(1+(S/B))-S))
    #Z = 2*(math.sqrt(S+B)-math.sqrt(B))
    if math.isnan(Z) or math.isinf(Z):
        print (S,B,Z)
        Z = 0
    return Z

def normalize(data):
    """Returns number of normalised events """
    # data = [cross section, number of generated events]
    N = 0
    L = 35922 # luminosity in pb^-1, cross section in pb
    split_idx = 0 
    data = np.concatenate((data,np.zeros((1,2))),axis=0) # add stop at the end

    for i in range(0,data.shape[0]):
        # Isolate subset of data with same Cross-section #
        if data[i,0]==data[split_idx,0]: # if no new cross section
            continue
        subset = data[split_idx:i,:]
        N += subset.shape[0]*subset[0,0]/subset[0,1] 
        # N = cross_section * L * efficiency 
        # Efficiency = n_samples/n_generated

        # change to new subset #
        split_idx = i
    N *= L
    print ('Sample number = %0.f, Normalised number = %0.f'%(data.shape[0],N))
    return N

def NNOperatingPoint(out,cut):
    """ Returns array of number fractions for given output and array of cuts """
    frac = np.zeros(cut.shape[0])
    for i in range(0,cut.shape[0]):
        check_in = np.greater(out,cut[i])
        frac[i] = out[check_in].shape[0]
    frac /= out.shape[0]
    #print (cut,' sample number = %0.f, cut number = %0.f'%(out.shape[0],frac*out.shape[0]))
    return frac



############################################################################### 
# Extract features from Root Files #
############################################################################### 
#INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/add_met_mll_forFlorian/slurm/output/'
print ('='*80)
print ('Starting input from files')
back_set = np.zeros((0,5))
sig_set = np.zeros((0,7))

gen_choices = np.zeros((0,2)) # records all the new configurations of (mH,mA)
N_sig = 0 # Number of different signal samples

# Get data from root files (sig+back)
for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('Opening file : ',filename)
        print ('\t-> Signal')
        Sig = True #Signal case
    else: # Background
        print ('Opening file : ',filename)
        print ('\t-> Background')
        Sig = False #Background case

    f = ROOT.TFile.Open(name)
    t = f.Get("t")
    N = t.GetEntries()
    cross_section = np.ones(N)*f.Get('cross_section').GetVal()
    event_weight_sum = np.ones(N)*f.Get('event_weight_sum').GetVal()
    
    #jj_M = np.asarray(tree2array(t, branches='jj_M'))
    #lljj_M = np.asarray(tree2array(t, branches='lljj_M'))
    # met_pt and ll_M
    selection = 'met_pt<80 && ll_M>70 && ll_M<110'
    jj_M = np.asarray(tree2array(t, branches='jj_M',selection=selection))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M',selection=selection))
    event_weight = np.asarray(tree2array(t, branches='event_weight', selection=selection))
    relative_weight = f.Get('cross_section').GetVal()/f.Get('event_weight_sum').GetVal()
    L = 35922 
    weight = L*(event_weight*relative_weight)#.reshape(-1,1)


    if Sig: #Signal
        # Extract mA, mH generated from file title
        num = [int(s) for s in re.findall('\d+',filename )]
        print ('\tmH = ',num[2],', mA = ',num[3])
        mH = np.ones(jj_M.shape[0])*num[2]
        mA = np.ones(jj_M.shape[0])*num[3]
        
        # Records new couple of generated mA and mH (dumps if not in ellipse config file)
        gen_config = np.c_[num[2],num[3]]

        test_isin = np.isin(gen_ellipse,gen_config)
        test_isin = np.logical_and(test_isin[:,0],test_isin[:,1])
        test_check = False
        for idx,val in enumerate(test_isin):
            if val == True:
                test_check = True
        if test_check == False:
            print ('Not an ellipse configuration')
            continue

        gen_choices = np.concatenate((gen_choices,gen_config),axis=0)

        # Append mlljj,mjj,mH,mA data to signal dataset
        sig_data = np.stack((lljj_M,jj_M,mH,mA,cross_section[:jj_M.shape[0]],event_weight_sum[:jj_M.shape[0]],weight),axis=1)
        sig_set = np.concatenate((sig_set,sig_data),axis=0) 
        print ('\t-> Size = %i,\ttotal signal size = %i' %(sig_data.shape[0],sig_set.shape[0]))

    else : # Background
        # Append mlljj and mjj data to background dataset
        back_data = np.stack((lljj_M,jj_M,cross_section[:jj_M.shape[0]],event_weight_sum[:jj_M.shape[0]],weight),axis=1)
        back_set = np.concatenate((back_set,back_data),axis=0)
        print ('\t-> Size = %i,\ttotal background size = %i' %(back_data.shape[0],back_set.shape[0]))

    print ('\tCross section = ',f.Get('cross_section').GetVal(),' Even weight sum ',f.Get('event_weight_sum').GetVal())
    print (weight)
    
print ('\n\nTotal signal size = ',sig_set.shape[0])
print ('Total background size = ',back_set.shape[0])

# Checks that the mA and mH selected (if the case) are in signal files
if args.all_plots == 'no':
    check = np.isin(np.c_[mH_select,mA_select],gen_choices)
    if not np.logical_and(check[:,0],check[:,1]) :
        sys.exit('[ERROR] : Selected values are not in signal samples')

############################################################################### 
# Complete background  
############################################################################### 
print ('='*80)
print ('Starting signal selection and background random assignation')
# Assign random (mH,mA) to background with same probabilities for each signal sample
N_sig = gen_choices.shape[0]
proba = np.ones(N_sig)/N_sig
indices = np.arange(0,N_sig)

rs = np.random.RandomState(42)
back_genrand = gen_choices[rs.choice(indices,size=back_set.shape[0],p=proba)]

back_set = np.c_[back_set[:,0:2],back_genrand,back_set[:,2:5]]

print ('\tDone')
############################################################################### 
# Loop over data #
############################################################################### 
print ('='*80)
print ('Starting ROC curve section')
# Generate Path For plots #
path_plots = '/home/ucl/cp3/fbury/Memoire/MassPlane/graph_ROC/model_test_'+str(args.model)+'/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_'+str(args.model)+'/'

# Background output #
print ('Output for background')
np.random.shuffle(back_set)
#back_set = back_set[:10000,:]
#B = back_set.shape[0]
B = sum(back_set[:,-1])
print (B)
t_back = np.zeros(back_set.shape[0])
print ('Output for background, size = %0.f, normalised : %0.5f'%(back_set.shape[0],B))

sigma_ell = np.array([0.1,0.5,1,2,3])
#sigma_ell = np.array([1])
cut_NN = np.array([0.99,0.95,0.9,0.8,0.5])
#cut_NN = np.array([0.9])

print ('\t-> Background NN Output')
z_back = NNOutput(back_set[:,:4],path_model)
op_back_NN = NNOperatingPoint(z_back,cut_NN)
print ('\tDone')
print ('\t-> Background Ellipse Output')
e_back,op_back_ell = EllipseOutput(data=back_set[:,:4],sigma=sigma_ell)
print ('\tDone')

# Signal Output #

for c in range(0,gen_choices.shape[0]):
    if args.all_plots == 'no':
        if gen_choices[c,1] != mA_select or gen_choices[c,0] != mH_select: # if not the requested configuration 
            continue
    print ('-'*80)
    print('[INFO] Using mH = %0.f and mA = %0.f (Process : %0.2f%%)'%(gen_choices[c,0],gen_choices[c,1],((c+1)/gen_choices.shape[0])*100))

    # Selects signal sample #
    mask = np.logical_and(sig_set[:,2]==gen_choices[c,0],sig_set[:,3]==gen_choices[c,1])
    sig_select = sig_set[mask]
    #S = sig_select.shape[0]
    S = sum(sig_set[:,-1])

    # Signal Output #
    print ('Output for signal, size = %0.f, normalised : %0.5f'%(sig_select.shape[0],S))
    t_sig = np.ones(sig_select.shape[0])

    print ('\t-> Signal NN Output')
    z_sig = NNOutput(sig_select[:,:4],path_model)
    print ('\tDone') 
    print ('\t-> Signal Ellipse Output')
    e_sig,op_sig_ell = EllipseOutput(data=sig_select[:,:4],sigma=sigma_ell)
    print ('\tDone')

    # Total (Sig + Back) output #
    z_tot = np.concatenate((z_sig,z_back),axis=0)
    e_tot = np.concatenate((e_sig,e_back),axis=0)
    t_tot = np.concatenate((t_sig,t_back),axis=0).reshape(-1,1)

    op_sig_NN = NNOperatingPoint(z_sig,cut_NN)

    # Ellpise output normalisation #
    e_tot_norm = 1-(e_tot/np.max(e_tot))
    e_sig_norm = e_tot_norm[:sig_select.shape[0]]
    e_back_norm = e_tot_norm[sig_select.shape[0]:]

    # ROC curve computations #
        # NN ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(t_tot,z_tot)
    roc_auc_z = auc(false_positive_rate, true_positive_rate)
    
    back_eff_z = false_positive_rate
    sig_eff_z = true_positive_rate
        # Ellipse ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(t_tot,e_tot_norm)
    roc_auc_e = auc(false_positive_rate, true_positive_rate)
    
    back_eff_e = false_positive_rate
    sig_eff_e = true_positive_rate

    # Find same operating point as ellipses for NN #
    op_diff = np.zeros(op_sig_ell.shape[0])
    for j in range(0,op_sig_ell.shape[0]): 
        idx = np.abs(sig_eff_z-op_sig_ell[j]).argmin() # find closest value in the NN case
        op_diff[j] = abs(back_eff_z[idx]-op_back_ell[j]) 

    # Significance estimation #
    #S_norm = normalize(sig_select[:,4:6])
    #B_norm = normalize(back_set[:,4:6])
    Z_NN = np.array([])
    Z_ell = np.array([])
    for s,b in zip(sig_eff_z,back_eff_z): # NN significancce
        sig_frac = s*S
        back_frac = b*B
        Z_NN = np.append(Z_NN,significance(sig_frac,back_frac))
    for s,b in zip(sig_eff_e,back_eff_e): # Ellipse significance
        sig_frac = s*S
        back_frac = b*B
        Z_ell = np.append(Z_ell,significance(sig_frac,back_frac))

    # Plot Section #
    fig1 = plt.figure(1,figsize=(10,5))
    ax1 = plt.subplot(111)
    plt.title('ROC Curve : $m_H$ = %0.f, $m_A$ = %0.f'%(gen_choices[c,0],gen_choices[c,1]))
    ax1.plot(sig_eff_z, back_eff_z,'b-', label=('NN Output : AUC = %0.5f'%(roc_auc_z)))
    ax1.plot(sig_eff_e, back_eff_e, 'g-', label=('Ellipse Output : AUC = %0.5f'%(roc_auc_e)))
    ax2 = ax1.twinx()
    ax2.plot(sig_eff_z,Z_NN, 'b--',label='NN Significance')
    ax2.plot(sig_eff_e,Z_ell,'g--',label='Ellipse Significance')
    for s in range(0,sigma_ell.shape[0]):
        gc = s*(1/(sigma_ell.shape[0]+1))
        ax1.plot(op_sig_ell[s],op_back_ell[s],'k*',markersize=10,color=str(gc),label='Ellipse operating point for $\sigma$ = %0.1f'%(sigma_ell[s]))
        #ax1.plot(op_sig_ell[s],op_back_ell[s],'k*',markersize=10,color=str(gc),label='Ellipse operating point for $\sigma$ = %0.1f\nDifference with NN = %0.5f'%(sigma_ell[s],op_diff[s]))
    for s in range(0,cut_NN.shape[0]):
        gc = s*(1/(cut_NN.shape[0]+1))
        ax1.plot(op_sig_NN[s],op_back_NN[s],'kX',markersize=10,color=str(gc),label='NN operating point for cut = %0.2f'%(cut_NN[s]))
    #plt.xlim([0,1])
    ax1.grid(True)
    ax1.set_yscale('symlog',linthreshy=0.01)
    ax1.set_xlabel('Signal Efficiency')
    ax1.set_ylabel('Background Efficiency')
    ax2.set_ylabel('Significance [arb. units]')
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,np.max([np.max(Z_NN),np.max(Z_ell)])])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.6, box.height])
    ax2.set_position([box.x0, box.y0, box.width*0.6, box.height])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1[0:2]+lines2+lines1[2:],labels1[0:2]+labels2+labels1[2:],loc='upper left',bbox_to_anchor=(1.1, 1),fancybox=True, shadow=True,labelspacing=0.8)
    #plt.show()
    fig1.savefig(path_plots+'mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1])+'.png')
    print ('[INFO] ROC curve plot saved as : '+path_plots+'mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1]))
    plt.close()

    fig2 = plt.figure(2)
    bins = np.linspace(0,1,50)
    plt.title('Ellipse : $m_H$ = %0.f, $m_A$ = %0.f'%(gen_choices[c,0],gen_choices[c,1]))
    plt.hist(e_sig_norm, color='g',density='normed',alpha=0.7,bins=bins, label=('Signal ellipse Output'))
    plt.hist(e_back_norm, color='r',density='normed',alpha=0.7,bins=bins, label=('Background ellipse Output'))
    plt.grid(True)
    plt.xlabel('Discriminant')
    plt.ylabel('Occurences')
    plt.legend(loc='upper right')
    #plt.show()
    fig2.savefig(path_plots+'out_ellipse_norm_mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1])+'.png')
    print ('[INFO] Ellipse output plot saved as : '+path_plots+'out_ellipse_norm_mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1]))
    plt.close()

    fig3 = plt.figure(3)
    bins = np.linspace(0,100,50)
    plt.title('Ellipse : $m_H$ = %0.f, $m_A$ = %0.f'%(gen_choices[c,0],gen_choices[c,1]))
    plt.hist(e_sig, color='g',density='normed',bins=bins, alpha=0.7, label=('Signal ellipse Output'))
    plt.hist(e_back, color='r',density='normed',bins=bins, alpha=0.7, label=('Background ellipse Output'))
    plt.grid(True)
    plt.xlabel('Number of sigmas')
    plt.ylabel('Occurences')
    plt.legend(loc='upper right')
    #plt.show()
    fig3.savefig(path_plots+'out_ellipse_mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1])+'.png')
    print ('[INFO] Ellipse output plot saved as : '+path_plots+'out_ellipse_mH_'+str(gen_choices[c,0])+'_mA_'+str(gen_choices[c,1]))
    plt.close()

