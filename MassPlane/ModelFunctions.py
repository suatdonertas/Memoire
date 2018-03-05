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
# Ellpise model #
############################################################################### 
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

def EllipseOutput(data):
    """ 
        Returns the number of sigma of deviation from ellipse center
    """
    # Loop over data points #
    out = np.ones(data.shape[0])*-1 
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
    return out

############################################################################### 
#  Operating Point #
############################################################################### 

def NNOperatingPoint(out,weight,cut):
    """ Returns array of number fractions for given output and array of cuts """
    frac = np.zeros(cut.shape[0])
    for i in range(0,cut.shape[0]):
        check_in = np.greater_equal(out,cut[i])
        frac[i] = np.sum(weight[check_in])/np.sum(weight)
    return frac

def EllipseOperatingPoint(out,weight,cut):
    """ Returns array of number fractions for given output and array of cuts """
    frac = np.zeros(cut.shape[0])
    for i in range(0,cut.shape[0]):
        check_in = np.less_equal(out,cut[i])
        frac[i] = np.sum(weight[check_in])/np.sum(weight)
    return frac
############################################################################### 
# Significance and normalization definition #
############################################################################## 
def significance(S,B):
    Z = np.sqrt(2*((S+B)*np.log(1+(S/B))-S))
    Z[Z == np.inf] = 0 # avoids inf, when B=0
    Z = np.nan_to_num(Z) # avoids nan, when S=B=0
    return Z
