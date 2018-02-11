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

import ROOT
from ROOT import TFile, TTree, TCanvas, TGraph2D, TPad, gPad, gStyle, TPaveText
from root_numpy import tree2array

from keras.models import Model
from keras.models import model_from_json

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Build Graph for given subest of mA,mH') 
parser.add_argument("-mA","--mA", help="Generated mass of A boson -> COMPULSORY")
parser.add_argument("-mH","--mH", help="Generated mass of H boson -> COMPULSORY")
parser.add_argument("-m","--model", help="Which model to use from the learning_model directory (format '10_10_10')-> COMPULSORY")

args = parser.parse_args()

mH_select = float(args.mH)
mA_select = float(args.mA)
print ('Use signal sample with mH = %i and mA = %i' %(mH_select,mA_select))
print ('Model used : '+str(args.model))


############################################################################### 
# Load model #
############################################################################### 
print ('='*80)
def NN_output(data):
    path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_'+str(args.model)+'/'
    output = np.zeros((data.shape[0],1))
    n_model = 0
    
    # Load weights
    for f in glob.glob(path_model+'*.json'):
        # Load model
        json_file = open(path_model+'model_step_1.json', 'r') 
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        
        # Load weights
        num = [int(s) for s in re.findall('\d+',f.replace(path_model,''))]
        f_weight = path_model+'weight_step'+str(num[0])+'.h5'
        model.load_weights(f_weight)
        output += model.predict(data)
        print ('Using model : '+f.replace(path_model,'')+'\tand weigths : '+f_weight.replace(path_model,''))
        n_model += 1

    output /= n_model

    return output

############################################################################### 
# Generate Graph #
############################################################################### 
print ('='*80)
print("[INFO] Generating Datapoints")
# Generate grid #
N = 100
max_mH = 2*mH_select
max_mA = 2*mA_select
mlljj = np.linspace(0,max_mH,N)
mjj = np.linspace(0,max_mH,N)


mass = np.asarray(list(itertools.product(mlljj,mjj,repeat=1))).astype(float)
mass_triangle = mass[mass[:,0]>mass[:,1]] # only keep mlljj>mjj
N_grid = mass_triangle.shape[0]
print ('Grid size : ',N_grid)

mA = np.ones(N_grid)*mA_select
mH = np.ones(N_grid)*mH_select
data = np.c_[mass_triangle,mH,mA]

z = NN_output(data)

print ('\tDone')

# Build TGraph2D #
print("[INFO] Generating Graph")
graph = TGraph2D()
n = 0
for i in range(0,N_grid):
    graph.SetPoint(n,data[i,1],data[i,0],z[i]) # x = mjj = data[:,1]
    n += 1

print ('\tDone')
# Design Canvas #
print("[INFO] Printing Canvas")
c1 = TCanvas( 'c1', 'MassPlane', 200, 10, 1200, 700 )
c1.SetFillColor( 10 )
c1.GetFrame().SetFillColor( 1 ) 
c1.GetFrame().SetBorderSize( 6 ) 
c1.GetFrame().SetBorderMode( -1 )

title = TPaveText( .3, 0.9, .7, .99 )
title.SetFillColor( 33 )
title.AddText('Mass Plane (m_{H} = %0.f, m_{A} = %0.f)'%(mH_select,mA_select))
title.Draw()

pad1 = TPad( 'pad1', 'Surface', 0.03, 0.10, 0.50, 0.80, 21 )
pad2 = TPad( 'pad2', 'Contour', 0.53, 0.10, 0.98, 0.80, 21 )
pad1.Draw()
pad2.Draw()

# Draw Graph (TRI) #
pad1.cd()
graph.SetTitle('Surface Plot;M_{jj};M_{lljj}')
graph.GetZaxis().SetTitle('DNN Output')
graph.Draw('TRI2')

c1.Update()

# Draw Graph (CONT) #
pad2.cd()
graph_new = graph.GetHistogram() # Needed to use SetContour
graph_new.SetContour(4)
graph_new.SetContourLevel(0,0.5)
graph_new.SetContourLevel(1,0.8)
graph_new.SetContourLevel(2,0.9)
graph_new.SetContourLevel(3,0.95)
graph_new.Draw('CONT1')
graph_new.Draw('CONTZ same')
graph_new.SetTitle('Contour Plot;M_{jj};M_{lljj}')
graph_new.GetZaxis().SetTitle('DNN Output')
gPad.SetRightMargin(0.15)

#c1.Modified()
c1.Update()

# Save Plots #
path = '/home/ucl/cp3/fbury/Memoire/MassPlane/graph_plots/'
if not os.path.exists(path):
    os.makedirs(path)

c1.Print(path+'graph_mH_'+str(mH_select)+'_mA_'+str(mA_select)+'_model_'+str(args.model)+'.pdf')


input('Press enter to end')


