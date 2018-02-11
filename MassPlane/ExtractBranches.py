# Libraries #
import glob
import os
import math
import numpy as np
import ROOT
from ROOT import TChain, TFile, TTree
from array import array

############################################################################### 
# Open root files and create new one with lljj_M and jj_M #
############################################################################### 
INPUT_FOLDER = '/home/ucl/cp3/swertz/scratch/CMSSW_8_0_25/src/cp3_llbb/HHTools/slurm/170728_skimForTrainingExtra/slurm/output/*.root'

OUTPUT_FOLDER = '/home/ucl/cp3/fbury/storage/'

f_out = TFile( OUTPUT_FOLDER+'invmass.root', 'RECREATE' )
t_out = TTree( 't', 'Tree' )

jj_M = array( 'f', [ 0. ] )
lljj_M = array( 'f', [ 0. ] )
t_out.Branch( 'jj_M', jj_M, 'jj_M/F' )
t_out.Branch( 'lljj_M', lljj_M, 'lljj_M/F' )

for f_in in glob.glob(INPUT_FOLDER):
    print '\nOpening : ',f_in
    file_in = ROOT.TFile.Open(f_in)

    t_in = file_in.Get("t")
    
    for entry in t_in:
        jj_M[0] = entry.jj_M
        lljj_M[0] = entry.lljj_M
        t_out.Fill()


f_out.Write()
f_out.Close()

