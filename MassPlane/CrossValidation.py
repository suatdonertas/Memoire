# Libraries #
import sys 
import os
import numpy as np

# Personal files #
from NeuralNet import NeuralNet

def CrossValidation(data,layers,K):
    """ 
    Perfoms K times the cross validation using the neural network
    Inputs :    - data : [N,n_features+2] : contains features+targets+weigths 
                - layers : number of neurons in each layer
                - K (int) : number of cross validation operations
    Outputs :   - err_avg : averaged RMS error of the network
                - AUC_avg : averaged AUC score of the model
    """
    N = data.shape[0]
    nf = data.shape[1]-2
    err_avg = 0
    AUC_avg = 0
    data_copy = np.copy(data) # Avoids shuffeling initial dataset

    # Starting cross-validation #
    i = 1
    fail = 0
    while i<=K:
        print ('Cross validation step, %i/%i'%(i,K))
        # Shuffeling the order and splitting x and t #
        np.random.shuffle(data_copy) 
        X = data_copy[:,:nf]
        T = data_copy[:,nf:nf+1]
        W = data_copy[:,nf+1:nf+2]

        # Model building and testing #
        if fail>=5 :
            print ('********** Cannot develop model *************')
            return 0,0
        err,AUC = NeuralNet(X,T,W,layers) 
        if AUC<0.6:
            print ('\t-> Error, wrong learning\n')
            continue
            fail += 1
        else:
            i+=1
        err_avg += err
        AUC_avg += AUC


    # Error averaging # 
    err_avg /= K
    AUC_avg /= K

    return err_avg,AUC_avg
