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

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Merge trees by averaging the output (intermediate and final)')

parser.add_argument('-l','--label', help='Path to files',required=True,type=str)
parser.add_argument('-w','--weight', help='either TT or DY',required=True,type=str)
parser.add_argument('-v','--valid', help='either "invalid" or ""',required=True,type=str)

args = parser.parse_args()
 
if not os.path.exists(args.label):
    sys.exit('Not valid path') 

print (args.label)
################################################################################
# Get trees #
################################################################################
f = ROOT.TFile(args.label+'output_1'+args.weight+args.valid+'.root')
t = f.Get('tree')
N = t.GetEntries()

n_trees = 1
output = np.c_[tree2array(t,branches=['NNOut_'+args.weight])].astype('float64')
other = tree2array(t,branches=['lep1_Pt','lep1_Eta','lep2_Pt','lep2_Eta','lep2_DPhi','jet1_Pt','jet1_Eta','jet1_DPhi','jet2_Pt','jet2_Eta','jet2_DPhi','met_pt','met_phi','jj_M','ll_M','lljj_M','mH','mA','visible_cross_section','original_MEM_TT_err','original_MEM_DY_err','weight','id','original_MEM_TT','original_MEM_DY','MEM_'+args.weight])

other = np.c_[other['lep1_Pt'],other['lep1_Eta'],other['lep2_Pt'],other['lep2_Eta'],other['lep2_DPhi'],other['jet1_Pt'],other['jet1_Eta'],other['jet1_DPhi'],other['jet2_Pt'],other['jet2_Eta'],other['jet2_DPhi'],other['met_pt'],other['met_phi'],other['jj_M'],other['ll_M'],other['lljj_M'],other['mH'],other['mA'],other['visible_cross_section'],other['original_MEM_TT_err'],other['original_MEM_DY_err'],other['weight'],other['id'],other['original_MEM_TT'],other['original_MEM_DY'],other['MEM_'+args.weight]]

for name in glob.glob(args.label+'output_*'+args.weight+args.valid+'.root'):
    print (name)
    if 'avg' in name:
        print ('average')
        continue
    number = name.replace(args.label+'output_','')
    number = number.replace('.root','')
    print ('Number of the model : ',number)
    if number==1:
        continue

    f = ROOT.TFile(name)
    t = f.Get('tree')

    # Get branches #
    output_new = tree2array(t,branches=['NNOut_'+args.weight]).astype('float64')
    for idx,val in enumerate(output_new):
        output[idx] += val
    #output = np.add(output,output_new)
    n_trees += 1

output /= float(n_trees)

full = np.c_[other,output]
full.dtype = [('lep1_Pt','float64'),('lep1_Eta','float64'),('lep2_Pt','float64'),('lep2_Eta','float64'),('lep2_DPhi','float64'),('jet1_Pt','float64'),('jet1_Eta','float64'),('jet1_DPhi','float64'),('jet2_Pt','float64'),('jet2_Eta','float64'),('jet2_DPhi','float64'),('met_pt','float64'),('met_phi','float64'),('jj_M','float64'),('ll_M','float64'),('lljj_M','float64'),('mH','float64'),('mA','float64'),('visible_cross_section','float64'),('original_MEM_TT_err','float64'),('original_MEM_DY_err','float64'),('weight','float64'),('id','float64'),('original_MEM_TT','float64'),('original_MEM_DY','float64'),('MEM_'+args.weight,'float64'),('NNOut_'+args.weight,'float64')]

array2root(full,args.label+'output_avg'+args.weight+args.valid+'.root',mode='recreate')
print (args.label+'output_avg'+args.weight+args.valid+'.root')

