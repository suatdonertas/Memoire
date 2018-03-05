#!/usr/bin/env python3

# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import itertools
import json

import ROOT
from ROOT import TFile, TTree, TCanvas, TGraph2D, TEllipse, TH2F
from ROOT import TPad, gPad, gStyle, TPaveText, gROOT
from root_numpy import tree2array

from keras.models import Model
from keras.models import model_from_json

# Personal Files #
from NNOutput import NNOutput

gROOT.SetBatch(True) # No display
gStyle.SetOptStat("") #no stat frame
###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Build Graph for given subest of mA,mH') 
parser.add_argument('-m','--model', help='Which model to use from the learning_model directory (format 10_10_10)-> COMPULSORY',required=True)
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
print ('Model used : '+str(args.model))

############################################################################### 
# Get all (mH,mA) configurations #
############################################################################### 
mHmA = np.zeros((0,2))
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
for name in glob.glob(INPUT_FOLDER+'HToZATo2L2B*.root'): # Just signals
    filename = name.replace(INPUT_FOLDER,'')
    # Extract mA, mH generated from file title
    num = [int(s) for s in re.findall('\d+',filename )]
    mH = num[2]
    mA = num[3]
    mHmA = np.concatenate((mHmA,np.c_[mH,mA]),axis=0)

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
gen_ellipse = np.c_[ellipse_conf[:,6],ellipse_conf[:,5]] #all (mH,mA) config in json file

def getEllipseConf(mA_c,mH_c,ellipse):
    """Given mA_C, mH_c, returns a,b,theta from ellipse"""
    for (mbb, mllbb, a, b, theta, mA, mH) in ellipse:
        if mA_c == mA and mH_c == mH:
            return mbb,mllbb,a,b,theta
    sys.exit('Could not find config with mH = %0.f and mA = %0.f'%(mH_c,mA_c))


############################################################################### 
# Generate Graph #
############################################################################### 
# Generate Path For plots #
path = '/home/ucl/cp3/fbury/Memoire/MassPlane/graph_plots/'+str(args.model)+'/'
#filename = 'model_'+str(args.model)+'.pdf'
filename = 'model_'+str(args.model)
if not os.path.exists(path):
    os.makedirs(path)

path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_'+str(args.model)+'/'

# Change mHmA array #
#mH_poss = np.linspace(100,1000,20)
#mA_poss = np.linspace(50,1000,20)
#mHmA_all = np.asarray(list(itertools.product(mH_poss,mA_poss,repeat=1))).astype(float)
#mHmA = mHmA_all[mHmA_all[:,1]<mHmA_all[:,0]-100]
mH_pos = np.array([100])
mA_pos = np.array([50]) #starting point
s = 0
inc_y = 20 # vertical increment
inc_x = 20 # horizontal increment 
way_x = 1 #direction of increment (+1 right, -1 left)
while mH_pos[s]<=1000:
    if way_x == 1: # Go right 
        if(mH_pos[s]>mA_pos[s]+30+inc_x): #check horizontal displacement
            mH_pos = np.append(mH_pos,mH_pos[s])
            mA_pos = np.append(mA_pos,mA_pos[s]+inc_x)
        else : # Vertical displacement
            mH_pos = np.append(mH_pos,mH_pos[s]+inc_y)
            mA_pos = np.append(mA_pos,mA_pos[s])
            way_x = -1 # after vertical, go left
    elif way_x == -1: # Go left
        if(mA_pos[s]-inc_x>0):
            mH_pos = np.append(mH_pos,mH_pos[s])
            mA_pos = np.append(mA_pos,mA_pos[s]-inc_x)
        else:
            mH_pos = np.append(mH_pos,mH_pos[s]+inc_y)
            mA_pos = np.append(mA_pos,mA_pos[s])
            way_x = 1 # after vertical, go right
    if mH_pos[s]>400:
        inc_y = 30
        inc_x = 30
    if mH_pos[s]>600:
        inc_y = 40
        inc_x = 40
    if mH_pos[s]>700:
        inc_y = 50
        inc_x = 50
    s += 1

mHmA = np.c_[mH_pos,mA_pos]

print (mHmA)
print (mHmA.shape)

print ('='*80)
# Design Canvas #
print('[INFO] Printing Canvas')
#c1.Print(path+filename+'[')  # opens pdf


print ('[INFO] Starting loop over (mH,mA)')
N = 100
rho = 2
for c in range(0,mHmA.shape[0]):
    c1 = TCanvas( 'c1', 'MassPlane', 200, 10, 1200, 700 )
    c1.SetFillColor( 10 )
    c1.GetFrame().SetFillColor( 1 ) 
    c1.GetFrame().SetBorderSize( 6 ) 
    c1.GetFrame().SetBorderMode( -1 )


    if args.all_plots == 'no':
        if mHmA[c,0] != mH_select or mHmA[c,1] != mA_select: # if not the requested configuration 
            continue

    print ('-'*80)
    # Generate grid #
    
    print('[INFO] Using mH = %0.f and mA = %0.f (%0.f/%0.f => Process : %0.2f%%)'%(mHmA[c,0],mHmA[c,1],c,mHmA.shape[0],(c/mHmA.shape[0])*100))
    max_mH = 1000#3*mHmA[c,0]
    max_mA = 1000 #*mHmA[c,1]
    mlljj = np.linspace(0,max_mH,N)
    mjj = np.linspace(0,max_mH,N)


    mass = np.asarray(list(itertools.product(mlljj,mjj,repeat=1))).astype(float)
    mass_triangle = mass[mass[:,0]>mass[:,1]] # only keep mlljj>mjj
    N_grid = mass_triangle.shape[0]
    print ('[INFO] Grid size : ',N_grid)

    mA = np.ones(N_grid)*mHmA[c,1]
    mH = np.ones(N_grid)*mHmA[c,0]
    data = np.c_[mass_triangle,mH,mA]

    z = NNOutput(data,path_model)
    z [z>0.999] = 0.995 # avoids overflow, Root does not like 0.9999..

    # Build TGraph2D #
    print('[INFO] Generating Graph')
    graph = TGraph2D()
    n = 0
    for i in range(0,N_grid):
        graph.SetPoint(n,data[i,1],data[i,0],z[i]) # x = mjj = data[:,1]
        n += 1

    print ('\tDone')

    title = TPaveText( .3, 0.9, .7, .99 )
    title.SetFillColor( 33 )
    title.AddText('Mass Plane (m_{H} = %0.f, m_{A} = %0.f)'%(mHmA[c,0],mHmA[c,1]))
    title.Draw()

    pad1 = TPad( 'pad1', 'Surface', 0.03, 0.10, 0.50, 0.85, 21 )
    pad2 = TPad( 'pad2', 'Contour', 0.53, 0.10, 0.98, 0.85, 21 )
    pad1.Draw()
    pad2.Draw()

    # Draw Graph (TRI) #
    graph_tri = graph.GetHistogram()
    pad1.cd()
    graph_tri.SetTitle('Surface Plot;M_{bb};M_{llbb}')
    gPad.SetLeftMargin(0.1)
    graph_tri.Draw('TRI2')
    gPad.Update
    graph_tri.SetTitle('Surface Plot')
    graph_tri.GetZaxis().SetRangeUser(0,1.0)
    graph_tri.GetXaxis().SetTitle('M_{bb}')
    graph_tri.GetXaxis().SetTitleOffset(2)
    graph_tri.GetYaxis().SetTitle('M_{llbb}')
    graph_tri.GetYaxis().SetTitleOffset(2)
    graph_tri.GetZaxis().SetTitle('DNN Output')
    graph_tri.GetZaxis().SetTitleOffset(1.5)
    gPad.SetRightMargin(0.1)
    gPad.SetLeftMargin(0.15)
    gPad.Update

    #c1.Update()

    # Draw Graph (CONT) #
    pad2.cd()
    ROOT.SetOwnership( graph, True )
    # Generate Th2F from TGraph2D #
    mH_binmax = 1000
    mH_nbin = 2000
    mH_bins = np.linspace(0,mH_binmax,mH_nbin)
    mA_binmax = 1000
    mA_nbin = 2000
    mA_bins = np.linspace(0,mA_binmax,mA_nbin)
    graph_cont = TH2F('mass_plane','Contour Plot;M_{bb};M_{llbb}',mA_nbin,0,mA_binmax,mH_nbin,0,mH_binmax)
    for mh in mH_bins:
        for ma in mA_bins:
            out_graph = graph.Interpolate(ma,mh)
            graph_cont.Fill(ma,mh,out_graph)

    #graph_cont = graph.GetHistogram() # Needed to use SetContour
    #ROOT.SetOwnership( graph_cont, True )
    #graph_cont.SetContour(6)
    #graph_cont.SetContourLevel(0,0.5)
    #graph_cont.SetContourLevel(1,0.6)
    #graph_cont.SetContourLevel(2,0.7)
    #graph_cont.SetContourLevel(3,0.8)
    #graph_cont.SetContourLevel(4,0.9)
    #graph_cont.SetContourLevel(5,0.95)
    graph_cont.Draw('CONT1')
    graph_cont.Draw('CONTZ same')
    graph_cont.SetTitle('Contour Plot;M_{bb};M_{llbb}')
    graph_cont.GetZaxis().SetTitle('DNN Output')
    graph_cont.GetZaxis().SetRangeUser(0,1.0)
    gPad.SetRightMargin(0.15)
    gPad.SetLeftMargin(0.15)

    # Print Ellipse #
    test_isin = np.isin(gen_ellipse,mHmA[c,:])
    test_isin = np.logical_and(test_isin[:,0],test_isin[:,1])
    test_check = False
    for idx,val in enumerate(test_isin):
        if val == True:
            test_check = True
    test_check = False # Don't want the ellipses now
    if test_check == True:
        x,y,a,b,theta = getEllipseConf(mHmA[c,1],mHmA[c,0],ellipse_conf)
        t = theta * 57.29 # radians -> Degrees
        ell = TEllipse(x,y,rho*math.sqrt(a),rho*math.sqrt(b),0,360,t)
        ell.SetFillStyle(0)
        ell.SetLineWidth(2)
        ell.Draw("same")
    else:
        print ('Not an ellipse configuration')

    #c1.Modified()
    #c1.Update()

    #c1.Print(path+filename,'Title:mH_%0.f_mA_%0.f'%(mHmA[c,0],mHmA[c,1]))
    c1.Print(path+filename+'_'+str(c)+'.jpg')
    c1.Clear()

#c1.Print(path+filename+']')
#if gROOT.IsBatch(): 
    #c1.Print(path+filename+'++')

#input('Press enter to end')


