# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas
from root_numpy import tree2array

# Personal files #
from NeuralNet import NeuralNet

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Takes root files and DNN architecture, tests them with cross validation and outputs RMS error and AUC score for each architexture.')
parser.add_argument("-a","--architecture", help="Name of the text file containing all the requested architectures -> COMPULSORY")

args = parser.parse_args()
print ('Architectures from ',str(args.architecture))

###############################################################################
# Importing architecture #
###############################################################################
path_hyper = '/home/ucl/cp3/fbury/Memoire/MassPlane/architectures/'
with open(path_hyper+str(args.architecture),'r') as f:
    for line in f:
        n_layers = len(np.asarray(line.split(),int))
 
archi = np.zeros((0,n_layers))

with open(path_hyper+str(args.architecture),'r') as f: 
    for line in f:
        a = np.asarray(line.split(),int).reshape(1,-1)
        archi = np.concatenate((archi,a)).astype(int)
        
############################################################################### 
# Extract features from Root Files #
############################################################################### 
#INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/skimmedPlots_for_Florian/slurm/output/'
INPUT_FOLDER = '/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/add_met_mll_forFlorian/slurm/output/'
print ('='*80)
print ('Starting input from files')
back_set = np.zeros((0,2))
sig_set = np.zeros((0,4))
sig_weight = np.zeros((0,1))
back_weight = np.zeros((0,1))

gen_choices = np.zeros((0,2)) # records all the new configurations of (mH,mA)

xsec = np.array([[0.63389,0.53676,0.41254,0.39846,0.30924,0.29973,0.22547,0.11122,0.10641,0.08736,0.04374,7.5130e-6,0.03721,0.01086,0.01051,0.0092366,2.4741e-5,2.1591e-7,0.0029526,0.0025357,5.51057e-6,3.99284e-8,9.82557e-10],[200,200,250,250,300,300,300,500,500,500,500,500,650,800,800,800,800,800,1000,1000,1000,2000,3000,],[50,100,50,100,50,100,200,50,100,200,300,400,50,50,100,200,400,700,50,200,500,1000,2000]]) #Computed with Olivier's code (xsec,mH,mA)

#for i in range(0,xsec.shape[1]):
    #print (xsec[0,i],xsec[1,i],xsec[2,i])

S = 0 # Number of signal events
N_sig = 0 # Number of events with same (mH,mA)
# Get number of signal events
for name in glob.glob(INPUT_FOLDER+'HToZATo2L2B*.root'):
    f = ROOT.TFile.Open(name)
    t = f.Get("t")
    N = t.GetEntries()
    S += N
    N_sig += 1

# Get data from root files (sig+back)
for name in glob.glob(INPUT_FOLDER+'*.root'):
    filename = name.replace(INPUT_FOLDER,'')
    print ('Opening file : ',filename)

    if filename.startswith('HToZATo2L2B'): # Signal
        print ('\t-> Signal')
        Sig = True #Signal case
    else: # Background
        print ('\t-> Background')
        Sig = False #Background case

    f = ROOT.TFile.Open(name)
    t = f.Get("t")
    
    selection = 'met_pt<80 && ll_M>70 && ll_M<110'
    jj_M = np.asarray(tree2array(t, branches='jj_M',selection=selection))
    lljj_M = np.asarray(tree2array(t, branches='lljj_M',selection=selection))
    total_weight = np.asarray(tree2array(t, branches='total_weight',selection=selection))
    N = jj_M.shape[0]
    if Sig: #Signal
        # Extract mA, mH generated from file title
        num = [int(s) for s in re.findall('\d+',filename )]
        print ('\tmH = ',num[2],', mA = ',num[3])
        mH = np.ones(N)*num[2]
        mA = np.ones(N)*num[3]

        # Get the relative signal weights 
        cross_section = 0
        for c in range(0,xsec.shape[1]):
            if xsec[1,c]==num[2] and xsec[2,c]==num[3]:
                cross_section = xsec[0,c]
        if cross_section == 0:
            sys.exit('Could not find cross section in signal sample') 
        print ('\tCross section = ',cross_section)
        #relative_weight = cross_section/f.Get('event_weight_sum').GetVal()
        weight = (total_weight/np.sum(total_weight)).reshape(-1,1)
        # Renormalize signal to make them equiprobable
        weight *= S/N
        sig_weight = np.concatenate((sig_weight,weight),axis=0)
        

        # Records new couple of generated mA and mH
        gen_config = np.c_[num[2],num[3]]
        gen_choices = np.concatenate((gen_choices,gen_config),axis=0)

        # Append mlljj,mjj,mH,mA data to signal dataset
        sig_data = np.stack((lljj_M,jj_M,mH,mA),axis=1)
        sig_set = np.concatenate((sig_set,sig_data),axis=0) 
        print ('\t-> Size = %i,\ttotal signal size = %i' %(sig_data.shape[0],sig_set.shape[0]))

    else : # Background
        # Set the background weights
        relative_weight = f.Get('cross_section').GetVal()/f.Get('event_weight_sum').GetVal()
        weight = (total_weight*relative_weight).reshape(-1,1)
        back_weight = np.concatenate((back_weight,weight),axis=0)

        # Append mlljj and mjj data to background dataset
        back_data = np.stack((lljj_M,jj_M),axis=1)
        back_set = np.concatenate((back_set,back_data),axis=0)
        print ('\t-> Size = %i,\ttotal background size = %i' %(back_data.shape[0],back_set.shape[0]))

print ('\n\nTotal signal size = ',sig_set.shape[0])
print ('Total background size = ',back_set.shape[0])

############################################################################### 
# Background random mH,mA assignation + target association #
############################################################################### 
print ('='*80)
print ('Starting target association')

# Background weights normalization wrt to signal #
sum_back_weight = np.sum(back_weight)
sum_sig_weight = np.sum(sig_weight)
back_weight = back_weight*(sum_sig_weight/sum_back_weight)
sig_weight *= 100
back_weight *= 100

# Assign random (mH,mA) to background with same probabilities for each signal sample
proba = np.ones(N_sig)/N_sig
indices = np.arange(0,N_sig)

rs = np.random.RandomState(42)
back_genrand = gen_choices[rs.choice(indices,size=back_set.shape[0],p=proba)]

back_set = np.c_[back_set,back_genrand]

# Assign targets #
sig_target = np.ones(sig_set.shape[0])
back_target = np.zeros(back_set.shape[0])

sig_set = np.c_[sig_set,sig_target,sig_weight]
back_set = np.c_[back_set,back_target,back_weight]

data = np.concatenate((sig_set,back_set),axis=0)
# data = [sig target weights
#         back targets weigths]     


print ('Total learning size = ',data.shape[0])


############################################################################### 
# Cross validation Neural Network #
############################################################################### 
K = 10 # Cross validation steps

print ('='*80)
print ('Starting Cross Validation')
for l in range(0,archi.shape[0]): # Loop over the architectures in the file
    print ('-'*80)
    print ('\nStarting architecture :',end='')
    print (np.array2string(archi[l,:]),end='\n')

    err_array = np.zeros((0,1))
    AUC_array = np.zeros((0,1))
    
    # Starting Cross Validation #
    i = 1 # Iterator
    fail = 0 # Number of failed models to stop loop
    err_avg = 0
    AUC_avg = 0
    nf = data.shape[1]-2 # minus targets and weights
    L2 = 0.01

    while i <= K:
        print ('Cross validation step, %i/%i'%(i,K))
        np.random.shuffle(data) # Random shuffleling of data
        X = data[:,:nf]
        T = data[:,nf:nf+1]
        W = data[:,nf+1:nf+2]
        
        err,AUC = NeuralNet(data=X,target=T,weight=W,layers=archi[l,:],step=i,L2=L2,print_plots=True,print_model=False)
        if AUC<0.6:
            fail += 1
            print ('\t-> Error, wrong learning (%i times)\n'%(fail))
            continue
        else:
            i += 1
        err_avg += err/K
        AUC_avg += AUC/K
        
        err_array = np.append(err_array,err) 
        AUC_array = np.append(AUC_array,AUC) 
        
    path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/'
    with open(path_model+'comparison', 'a') as outfile:
        outfile.write('architecture : '+np.array2string(archi[l,:])+' l2 = '+str(L2)) 
        outfile.write('\n\tRMS : '+np.array2string(err_array)+' => '+str(err_avg))
        outfile.write('\n\tAUC : '+np.array2string(AUC_array)+' => '+str(AUC_avg)+'\n')


    print ('\nLayer architecture : ',end='')
    print (np.array2string(archi[l,:]),end='')
    print ('\tRMS error = %0.5f, AUC score = %0.5f\n'%(err_avg,AUC_avg)) 
            
        
