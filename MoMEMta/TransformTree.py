# Libraries #
import sys 
import glob
import os
import re
import argparse
import math
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TObject
from ROOT import TLorentzVector
from root_numpy import tree2array, array2root

###############################################################################
# Transformation Function #
###############################################################################
def Transform(Px,Py,Pz,E):
    N = Px.shape[0]
    Pt = np.zeros(N)
    Eta = np.zeros(N)
    Phi = np.zeros(N)
    M = np.zeros(N)
    
    LV = TLorentzVector()
    print ('\tStarting Transformation')
    for i in range(0,N):
        sys.stdout.write('\r')
        sys.stdout.write('\tCurrent process : %0.2f%%'%(i%N/N*100))
        sys.stdout.flush()
        LV.SetPx(Px[i])
        LV.SetPy(Py[i])
        LV.SetPz(Pz[i])
        LV.SetE(E[i])
        Pt[i] = LV.Pt()
        Eta[i] = LV.Eta()
        Phi[i] = LV.Phi()
        M[i] = LV.M()
    print ()
    return Pt,Eta,Phi,M

###############################################################################
# Import Root Files #
###############################################################################
print ('='*80)
print ('[INFO] Starting input from files')
INPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output/'
OUTPUT_FOLDER = '/home/ucl/cp3/fbury/storage/MoMEMta_output_transform/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

sig_set = np.empty((0,21))
TT_set = np.empty((0,21))
DY_set = np.empty((0,21))

sig_MEM = np.empty((0,2))
TT_MEM =np.empty((0,2))
DY_MEM = np.empty((0,2))

sig_weight =np.empty((0,1))
sig_learning_weight =np.empty((0,1))
TT_weight = np.empty((0,1))
DY_weight = np.empty((0,1)) 

for name in glob.glob(INPUT_FOLDER+'*.root'):
    # Input the files #
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    f = ROOT.TFile.Open(name)
    t = f.Get("tree")

    # Get branches into numpy arrays #
    total_weight = tree2array(t, branches='total_weight').reshape(-1,1)
    event_weight = tree2array(t, branches='event_weight').reshape(-1,1)
    event_weight_sum = f.Get('event_weight_sum').GetVal()
    cross_section = f.Get('cross_section').GetVal()
    cs = TObject()
    ws = TObject()
    f.GetObject("cross_section",cs)
    f.GetObject("event_weight_sum",ws)

    lep1_E = tree2array(t, branches='lep1_p4.E()')
    lep1_Px = tree2array(t, branches='lep1_p4.Px()')
    lep1_Py = tree2array(t, branches='lep1_p4.Py()')
    lep1_Pz = tree2array(t, branches='lep1_p4.Pz()')

    lep2_E = tree2array(t, branches='lep2_p4.E()')
    lep2_Px = tree2array(t, branches='lep2_p4.Px()')
    lep2_Py = tree2array(t, branches='lep2_p4.Py()')
    lep2_Pz = tree2array(t, branches='lep2_p4.Pz()')

    jet1_E = tree2array(t, branches='jet1_p4.E()')
    jet1_Px = tree2array(t, branches='jet1_p4.Px()')
    jet1_Py = tree2array(t, branches='jet1_p4.Py()')
    jet1_Pz = tree2array(t, branches='jet1_p4.Pz()')

    jet2_E = tree2array(t, branches='jet2_p4.E()')
    jet2_Px = tree2array(t, branches='jet2_p4.Px()')
    jet2_Py = tree2array(t, branches='jet2_p4.Py()')
    jet2_Pz = tree2array(t, branches='jet2_p4.Pz()')


    met_pt  = tree2array(t, branches='met_pt')
    met_phi  = tree2array(t, branches='met_phi')

    jj_M  = tree2array(t, branches='jj_M')
    ll_M  = tree2array(t, branches='ll_M')
    lljj_M  = tree2array(t, branches='lljj_M')

    MEM_TT = tree2array(t, branches='weight_TT').reshape(-1,1)
    MEM_TT_err = tree2array(t, branches='weight_TT_err').reshape(-1,1)
    MEM_DY = tree2array(t, branches='weight_DY').reshape(-1,1)
    MEM_DY_err = tree2array(t, branches='weight_DY_err').reshape(-1,1)

    # LorentzVector transformation #
    lep1_Pt,lep1_Eta,lep1_Phi,lep1_M = Transform(lep1_Px,lep1_Py,lep1_Pz,lep1_E)
    lep2_Pt,lep2_Eta,lep2_Phi,lep2_M = Transform(lep2_Px,lep2_Py,lep2_Pz,lep2_E)
    jet1_Pt,jet1_Eta,jet1_Phi,jet1_M = Transform(jet1_Px,jet1_Py,jet1_Pz,jet1_E)
    jet2_Pt,jet2_Eta,jet2_Phi,jet2_M = Transform(jet2_Px,jet2_Py,jet2_Pz,jet2_E)


    # Concatenating into one set #
    output = np.c_[total_weight,event_weight,lep1_Pt,lep1_Eta,lep1_Phi,lep1_M,lep2_Pt,lep2_Eta,lep2_Phi,lep2_M,jet1_Pt,jet1_Eta,jet1_Phi,jet1_M,jet2_Pt,jet2_Eta,jet2_Phi,jet2_M,met_pt,met_phi,jj_M,ll_M,lljj_M,MEM_TT,MEM_TT_err,MEM_DY,MEM_DY_err]

    output.dtype = [('total_weight','float64'),('event_weight','float64'),('lep1_Pt','float64'),('lep1_Eta','float64'),('lep1_Phi','float64'),('lep1_M','float64'),('lep2_Pt','float64'),('lep2_Eta','float64'),('lep2_Phi','float64'),('lep2_M','float64'),('jet1_Pt','float64'),('jet1_Eta','float64'),('jet1_Phi','float64'),('jet1_M','float64'),('jet2_Pt','float64'),('jet2_Eta','float64'),('jet2_Phi','float64'),('jet2_M','float64'),('met_pt','float64'),('met_phi','float64'),('jj_M','float64'),('ll_M','float64'),('lljj_M','float64'),('weight_TT','float64'),('weight_TT_err','float64'),('weight_DY','float64'),('weight_DY_err','float64')]
    output.dtype.names = ['total_weight','event_weight','lep1_Pt','lep1_Eta','lep1_Phi','lep1_M','lep2_Pt','lep2_Eta','lep2_Phi','lep2_M','jet1_Pt','jet1_Eta','jet1_Phi','jet1_M','jet2_Pt','jet2_Eta','jet2_Phi','jet2_M','met_pt','met_phi','jj_M','ll_M','lljj_M','weight_TT','weight_TT_err','weight_DY','weight_DY_err']

    array2root(output,OUTPUT_FOLDER+filename, mode='recreate')

    # Opens the root file produced and adds the TObjects #
    f_object = TFile(OUTPUT_FOLDER+filename, "update")
    t_object = TTree("tree", "tree")

    cs.Write()
    ws.Write()
