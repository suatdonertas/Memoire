# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import json
import itertools

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error as mse

from scipy.optimize import newton
from scipy.stats import norm

import ROOT
from ROOT import TFile, TTree, TCanvas, TGraph2D, TH2F
from ROOT import TPad, gPad, gStyle, TPaveText, gROOT
from ROOT import kBlack, kBlue, kRed, kOrange, kYellow, kGreen
from root_numpy import tree2array,array2root

from keras.models import model_from_json

import matplotlib.pyplot as plt


# Personal libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
from NNOutput import NNOutput
from FindOperatingPoint import *

gROOT.SetBatch(True)

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
path_plot = '/home/ucl/cp3/fbury/Memoire/BigNetwork/Graph/'+args.model_notrain+'_'+args.model_retrain+'/'
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

print ('Model no train : '+path_tree_notrain)
print ('Model retrain : '+path_tree_retrain)
print ('mH : '+args.mH)
print ('mA : '+args.mA)
mH = args.mH
mA = args.mA

f1 = ROOT.TFile.Open(path_tree_notrain)
t1 = f1.Get("tree")
f2 = ROOT.TFile.Open(path_tree_retrain)
t2 = f2.Get("tree")

################################################################################
# Find equivalent working point #
################################################################################
sig_eff_wanted = 0.7

out1 = tree2array(t1,branches=['MP_discriminant','target','weight'],selection='mA=='+str(mA)+' && mH=='+str(mH))
out2 = tree2array(t2,branches=['output_last','target','weight'],selection='mA=='+str(mA)+' && mH=='+str(mH))

#cut = FindOperatingPoint(output_A=out1['output'],target_A=out1['target'],weight_A=out1['weight'],cut_A=0.9,output_B=out2['output_last'],target_B=out2['target'],weight_B=out2['weight'])
cut1 = FindOperatingPoint(sig_eff=sig_eff_wanted,output=out1['MP_discriminant'],target=out1['target'],weight=out1['weight'])
cut2 = FindOperatingPoint(sig_eff=sig_eff_wanted,output=out2['output_last'],target=out2['target'],weight=out2['weight'])
print ('Mass Plane cut : ',cut1)
print ('Frankenstein cut : ',cut2)

################################################################################
# Dictionnary of plot configs #
################################################################################
config = {
    (200,50):(300,100),
    (200,100):(270,150),
    (250,50):(350,100),
    (250,100):(350,150),
    (300,100):(420,170),
    (300,200):(400,250),
    (500,50):(700,200),
    (500,100):(800,200),
    (500,200):(750,300),
    (500,300):(700,400),
    (500,400):(750,550),
    (650,50):(2100,500),
    (800,50):(4000,1200),
    (800,100):(2500,500),
    (800,200):(1600,500),
    (800,400):(1300,600),
    (800,700):(1600,900),
    (1000,50):(4000,2000),
    (1000,200):(4000,2000),
    (1000,500):(2000,800),
    (2000,1000):(6000,2000),
    (3000,2000):(8000,4000),
}
# (mH,mA),(mH_max_plot,mA_max_plot)

mH_max_plot = config[int(mH),int(mA)][0]
mA_max_plot = config[int(mH),int(mA)][1]
print ('mH_max_plot : ',mH_max_plot)
print ('mA_max_plot : ',mA_max_plot)


################################################################################
# Hist from Frankenstein #
################################################################################
t2.Draw("mlljj:mjj>>hist_F(100,0,"+str(mA_max_plot)+",100,0,"+str(mH_max_plot)+")","target==1 && mH=="+mH+" && mA =="+mA+" && output_last>"+str(cut2))
hist_F = gROOT.FindObject("hist_F")


################################################################################
# TGraph2D #
################################################################################

c1 = TCanvas( 'c1', 'c1', 200, 10, 800, 700 )
gROOT.SetStyle("Plain")
gStyle.SetOptStat(0)

gStyle.SetTitleFontSize(.11)
gStyle.SetLabelSize(.05, "XY")

gStyle.SetPadBottomMargin(0.2)
gStyle.SetPadLeftMargin(0.2)
gStyle.SetPadRightMargin(0.2)
gStyle.SetPadTopMargin(0.2)
#gStyle.SetPalette(9)


# Build grid #
mlljj = np.linspace(0,mH_max_plot,500)
mjj = np.linspace(0,mA_max_plot,500)
mass = np.asarray(list(itertools.product(mlljj,mjj,repeat=1))).astype(float)
mass_triangle = mass[mass[:,0]>mass[:,1]] # only keep mlljj>mjj
N_grid = mass_triangle.shape[0]

mA_vec = np.ones(N_grid)*int(mA)
mH_vec = np.ones(N_grid)*int(mH)
data = np.c_[mass_triangle,mH_vec,mA_vec]
print (data.shape)

path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_l2/'
z = NNOutput(data,path_model)
print (z)
print (z[z>cut1])

# Build TGraph2D #
graph = TGraph2D()
n = 0
for i in range(0,N_grid):
    if i%1000==0:
        sys.stdout.write('\r\tProcessing TGraph2D : %0.f%%'%((n/data.shape[0])*100))
        sys.stdout.flush()
    graph.SetPoint(n,data[i,1],data[i,0],z[i])
    n+=1
print ()

# Title and pad #
title = TPaveText( .2, 0.9, .8, .99 )
title.SetFillColor(0)
title.AddText('Mass Plane (m_{H} = %0.f GeV, m_{A} = %0.f GeV)'%(int(mH),int(mA)))
title.Draw()
Cut = TPaveText( 0.3, 0.85, .8, .95,'brNDC')
Cut.SetFillColor(0)
Cut.AddText('Mass Plane cut : %0.5f'%(cut1))
Cut.AddText('Frankenstein cut : %0.5f'%(cut2))


pad1 = TPad( 'pad1', 'Surface', 0, 0, 1, 0.89 )
#pad1 = TPad( 'pad1', 'Surface', 0, 0, 0.45, 0.89 )
#pad2 = TPad( 'pad1', 'Surface', 0.55, 0, 1, 0.89 )
pad1.Draw()
#pad2.Draw()


# Contour Plot #
pad1.cd()
pad1.SetLeftMargin(0.15)
pad1.SetRightMargin(0.2)
pad1.SetTopMargin(0.05)
pad1.SetBottomMargin(0.2)

hist_F.Draw("colz")
#hist_F.GetXaxis().SetLimits(0, mA_max_plot)
#hist_F.GetYaxis().SetLimits(0, mH_max_plot)
#hist_F.Draw("colz")
hist_F.SetTitle(';M_{bb} [GeV];M_{llbb} [GeV]; Occurences')
hist_F.GetXaxis().SetTitleSize(.06)
hist_F.GetYaxis().SetTitleSize(.06)
hist_F.GetZaxis().SetTitleSize(.06)
hist_F.GetXaxis().SetTitleOffset(1.1)
hist_F.GetYaxis().SetTitleOffset(1.1)
hist_F.GetZaxis().SetTitleOffset(1.1)

gPad.Update()

#pad2.cd()

ROOT.SetOwnership( graph, True )
mH_binmax = mH_max_plot
mH_binmin = 0
mH_nbin = 1000
mH_bins = np.linspace(mH_binmin,mH_binmax,mH_nbin)
mA_binmax = mA_max_plot 
mA_binmin = 0
mA_nbin = 1000
mA_bins = np.linspace(mA_binmin,mA_binmax,mA_nbin)
graph_contour = TH2F('mass_plane',';M_{bb} [GeV];M_{llbb} [GeV]',mA_nbin,mA_binmin,mA_binmax,mH_nbin,mH_binmin,mH_binmax)
k = 0
for mh in mH_bins:
    for ma in mA_bins:
        if k%1000==0:    
            sys.stdout.write('\r\tInterpolate TGraph2D : %0.f%%'%((k/(mH_nbin*mA_nbin))*100))
            sys.stdout.flush()
        out_graph = graph.Interpolate(ma,mh)
        graph_contour.Fill(ma,mh,out_graph)
        k += 1
print ()
graph_contour.SetLineWidth(4)
graph_contour.SetContour(1)
graph_contour.SetContourLevel(0,cut1)
graph_contour.GetXaxis().SetLimits(0, mA_max_plot)
graph_contour.GetYaxis().SetLimits(0, mH_max_plot)
graph_contour.Draw('CONT2 same')
Cut.Draw()

gPad.Update()
#input("Press key to end")

c1.Print(path_plot+'Graph_mH_'+mH+'_mA_'+mA+'.pdf')
c1.Clear()
