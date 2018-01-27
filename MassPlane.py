import glob
import os
import math

import numpy as np

from ROOT import TChain, TFile, TTree


############################################################################### 
# Import Ntuples #
############################################################################### 
INPUT_FOLDER = '/home/ucl/cp3/swertz/scratch/CMSSW_8_0_25/src/cp3_llbb/HHTools/slurm/170728_skimForTrainingExtra/slurm/output/*.root'

for f in glob.glob(INPUT_FOLDER):
    #print 'Opening : ',f
