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
parser = argparse.ArgumentParser(description='Merge trees by averaging the outputs (intermediate and final)')

parser.add_argument('-l','--label', help='Model with weights and retraining',required=True,type=str)

args = parser.parse_args()
 
if not os.path.exists(args.label):
    sys.exit('Not valid path') 

print (args.label)
################################################################################
# Get trees #
################################################################################
f = ROOT.TFile(args.label+'output_1.root')
t = f.Get('tree')
N = t.GetEntries()

n_trees = 1
outputs = tree2array(t,branches=['output_MP','output_MEM_TT','output_MEM_DY','discriminant','output_last'])
other = tree2array(t,branches=['mH','mA','mlljj','mjj','weight','target'])

outputs = np.c_[outputs['output_MP'],outputs['output_MEM_TT'],outputs['output_MEM_DY'],outputs['discriminant'],outputs['output_last']]
other = np.c_[other['mH'],other['mA'],other['mlljj'],other['mjj'],other['weight'],other['target']]

for name in glob.glob(args.label+'output_*.root'):
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
    output_new = tree2array(t,branches=['output_MP','output_MEM_TT','output_MEM_DY','discriminant','output_last'])
    output_new = np.c_[output_new['output_MP'],output_new['output_MEM_TT'],output_new['output_MEM_DY'],output_new['discriminant'],output_new['output_last']]
    outputs = np.add(outputs,output_new)

    n_trees += 1

outputs /= n_trees

full = np.c_[outputs,other]
full.dtype = [('output_MP','float64'),('output_MEM_TT','float64'),('output_MEM_DY','float64'),('discriminant','float64'),('output_last','float64'),('mH','float64'),('mA','float64'),('mlljj','float64'),('mjj','float64'),('weight','float64'),('target','float64')]

array2root(full,args.label+'output_avg.root',mode='recreate')
print (args.label+'output_avg.root')

