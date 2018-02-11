# Libraries #
import glob
import os
import math
import numpy as np
import ROOT
from ROOT import TChain, TFile, TTree, TCanvas, TH2F
from array import array

import root_numpy 

############################################################################### 
# Open root files and create new one with lljj_M and jj_M #
############################################################################### 
INPUT_FOLDER = '/home/ucl/cp3/swertz/scratch/CMSSW_8_0_25/src/cp3_llbb/HHTools/slurm/170728_skimForTrainingExtra/slurm/output/*.root'

OUTPUT_FOLDER = '/home/ucl/cp3/fbury/storage/'

c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
c1.SetFillColor( 10 )
c1.GetFrame().SetFillColor( 1 )
c1.GetFrame().SetBorderSize( 6 )
c1.GetFrame().SetBorderMode( -1 )

mass_plane = TH2F( 'mass_plane', 'M_{lljj} vs M_{jj};M_{jj};M_{lljj}', 40, 0, 1000, 40, 0, 1000 )

for f_in in glob.glob(INPUT_FOLDER):
    print ('\nOpening : ',f_in)
    file_in = ROOT.TFile.Open(f_in)

    t_in = file_in.Get("t")
    
    for entry in t_in:
        mass_plane.Fill(entry.jj_M,entry.lljj_M)

    mass_plane.Draw('COLZ')
    c1.Modified()
    c1.Update()

raw_input("Press enter to end")
