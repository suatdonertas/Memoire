# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np
import itertools

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Generate different architectures given the number of layers and the increment in number of neurons per layer')
parser.add_argument("-n","--n_layers", help="Number of layers in the architecture -> COMPULSORY")
parser.add_argument("-s","--step", help="Steps for the modification of the number of neurons -> COMPULSORY")
parser.add_argument("-max","--max_neurons", help="Maximum number of neurons per file -> COMPULSORY")
parser.add_argument("-min","--min_neurons", help="Minimum number of neurons per file -> COMPULSORY")
parser.add_argument("-o","--output", help="Name of the output file -> COMPULSORY")


args = parser.parse_args()
print ('Number of layers : ',str(args.n_layers))
print ('Increment steps : ',str(args.step))
print ('Maximum number of neurons : ',str(args.max_neurons))
print ('Minimum number of neurons : ',str(args.min_neurons))
print ('Output file : ',str(args.output))

############################################################################## 
# Generating combinations #
############################################################################## 
poss = np.arange(int(args.min_neurons),int(args.max_neurons)+int(args.step),int(args.step))

comb = np.asarray(list(itertools.product(poss,repeat=int(args.n_layers)))).astype(int)
print ('Number of combinations : ',comb.shape[0])


############################################################################## 
# Write to file  #
############################################################################## 

with open(str(args.output),'w') as f:
    for i in range(0,comb.shape[0]):
        for j in range(0,comb.shape[1]):
            f.write(str(comb[i,j]))
            f.write(' ')
        f.write('\n')










