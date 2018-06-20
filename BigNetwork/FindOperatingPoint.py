# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import json
from sklearn.metrics import roc_curve

# Personal libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
from ModelFunctions import *

def FindOperatingPoint_cutA(output_A,target_A,weight_A,cut_A,output_B,target_B,weight_B):
    """
    Given the cut applied on model A, and given the output and target of model B, find the cut to apply on model B in order to get the same signal efficiency
    """
    cut_A_arr = np.array([cut_A])
    sig_expected = NNOperatingPoint(output_A[target_A==1],weight_A[target_A==1],cut_A_arr)
    sig_set = output_A[target_A==1]

    # Find signal efficiency spectrum for model B#
    back_B,sig_B,treshold = roc_curve(target_B,output_B,sample_weight=weight_B) 

    # Find closest point to sig_expected #
    sig_found = sig_B[find_nearest(sig_B,sig_expected)]

    dif = 1000
    # Find operating point thas as sig_found as signal efficiency #
    cut_test = np.linspace(0,1,100000)
    sig_test = NNOperatingPoint(output_B[target_B==1],weight_B[target_B==1],cut_test)
    cut_out = cut_test[find_nearest(sig_test,sig_found)]
    return cut_out

def FindOperatingPoint(sig_eff,output,target,weight):
    """
    For the given signal efficiency, find the corresponding cut
    """
    cut_test = np.linspace(0,1,100000)
    sig_test = NNOperatingPoint(output[target==1],weight[target==1],cut_test)
    cut_out = cut_test[find_nearest(sig_test,sig_eff)]
    return cut_out


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    

