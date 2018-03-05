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
from ROOT import TFile, TTree, TCanvas, TGraph2D, TEllipse, TH2F
from ROOT import TPad, gPad, gStyle, TPaveText, gROOT
from root_numpy import tree2array, fill_hist

import matplotlib.pyplot as plt

# Personal Files #
from cutWindow import massWindow

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Evaluates perfomances of a given model, for eaxh mA and mH specified')

parser.add_argument('-a','--all_plots', help='Wether to plot the massplanes for all (mH,mA) configurations : yes (default) or no (then specifiy mA and mH) -> COMPULSORY',default='yes',required=False)
parser.add_argument('-mA','--mA', help='Generated mass of A boson',required=False)
parser.add_argument('-mH','--mH', help='Generated mass of H boson',required=False)

args = parser.parse_args()
 
if args.all_plots == 'no':
    mH_select = float(args.mH)
    mA_select = float(args.mA)
    print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
if args.all_plots == 'yes':
    print ('Using all configurations')

############################################################################### 
# Get ellipses configurations #
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

def getEllipseConf(mA_c,mH_c,ellipse):
    """Given mA_C, mH_c, returns a,b,theta from ellipse"""
    for (mbb, mllbb, a, b, theta, mA, mH) in ellipse:
        if mA_c == mA and mH_c == mH:
            return mbb,mllbb,a,b,theta
    sys.exit('Could not find config with mH = %0.f and mA = %0.f'%(mH_c,mA_c))
############################################################################### 
# Extract features from Root Files #
############################################################################### 
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
print ('='*80)
print ('Starting input from files')

# Get data from root files  : Signal 
for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('\t-> Signal')
    else: # Background
        continue
    # Extract mA, mH generated from file title
    num = [int(s) for s in re.findall('\d+',filename )]
    if mH_select != num[2] or mA_select != num[3]:
        print ('Not the file we want')
        continue
    else:
        print ('File we want')

    f = ROOT.TFile.Open(name)
    t = f.Get("t")
    #N = t.GetEntries()

    jj_M = np.asarray(tree2array(t, branches='jj_M',selection='met_pt<80 && ll_M>70 && ll_M<110'))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M',selection='met_pt<80 && ll_M>70 && ll_M<110'))

    sig_cs = f.Get('cross_section').GetVal()
    sig_ws = f.Get('event_weight_sum').GetVal()
    
    sig_set = np.stack((lljj_M,jj_M),axis=1)

# Get data from root files  : Background 
for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    if filename.startswith('HToZATo2L2B'): # Signal
        continue
    else: # Background
    
    f = ROOT.TFile.Open(name)
    t = f.Get("t")

    jj_M = np.asarray(tree2array(t, branches='jj_M',selection='met_pt<80 && ll_M>70 && ll_M<110'))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M',selection='met_pt<80 && ll_M>70 && ll_M<110'))

    cs = f.Get('cross_section').GetVal()
    ws = f.Get('event_weight_sum').GetVal()

    if filename.startswith('DYToll_0J'):
        DYToll_0J_cs = cs
        DYToll_0J_ws = ws 
        DYToll_0J_set = np.stack((lljj_M,jj_M),axis=1)
    if filename.startswith('DYToll_1J'):
        DYToll_1J_cs = cs
        DYToll_1J_ws = ws 
        DYToll_1J_set = np.stack((lljj_M,jj_M),axis=1)
    if filename.startswith('DYToll_2J'):
        DYToll_2J_cs = cs
        DYToll_2J_ws = ws 
        DYToll_2J_set = np.stack((lljj_M,jj_M),axis=1)
    if filename.startswith('TTTo2L2Nu'):
        TTTo2L2Nu_cs = cs
        TTTo2L2Nu_ws = ws 
        TTTo2L2Nu_set = np.stack((lljj_M,jj_M),axis=1)
    if filename.startswith('TT_Other'):
        TT_Other_cs = cs
        TT_Other_ws = ws 
        TT_Other_set = np.stack((lljj_M,jj_M),axis=1)



