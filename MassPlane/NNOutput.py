#!/usr/bin/env python3

# Libraries #
import sys
import glob
import os
import re
import argparse
import math
import numpy as np

from keras.models import Model
from keras.models import model_from_json

def NNOutput(data,path_model):
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
        print ('\tUsing model : '+f.replace(path_model,'')+'\tand weigths : '+f_weight.replace(path_model,''))
        model.load_weights(f_weight)
        output += model.predict(data)
        n_model += 1

    output /= n_model

    return output

