# Libraries #
import sys 
import glob
import os
import math
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas
from root_numpy import tree2array

from keras import utils
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import losses, optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json
from keras.regularizers import l1,l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def NeuralNet(data,target,weight,layers,step=1,L2=0,print_plots=False,print_model=False):
    """ 
    Build the neural network
    Inputs :    - data = [N,n_features] 
                - target = [N,1]
                - weight = [N,1]
                - layers = number of neurons per layer
                - step =  corss validation step (for naming files)
                - L2 = regularization term (0 if no regularization)
                - print_plots = wether to print the loss+acc and ROC plots
                - print_model = wether to display the model
    Function Outputs :  - RMS error of the model
                        - AUC score of the model
    Other outputs :     - Loss and accuracy in same plots
                        - ROC curve
                        - Model and weights for each step
                
    """
    # Splitting the dataset #
    X_train, X_test, T_train, T_test, W_train, W_test = train_test_split(data,target,weight,test_size=0.25,shuffle=False)

    W_train = W_train.reshape(-1,)
    W_test = W_test.reshape(-1,)

    # Building layers #
    inputs = Input(shape=(X_train.shape[1],))
    Dx = Dense(layers[0], activation="relu")(inputs) # First layer
    #B = BatchNormalization()(Dx)
    Dx = Dense(layers[1], activation="relu",kernel_regularizer=l2(L2))(Dx)
    for i in range(2,len(layers)): # Hidden layers
        Dx = Dense(layers[i], activation="relu",kernel_regularizer=l2(L2))(Dx)
    Dx = Dense(1, activation="tanh")(Dx)

        # Make directory for the model #
    if L2 == 0:
        model_directory = np.array2string(layers).replace('[','_').replace(']','').replace(' ','_')+'_withweights_log_rescale_preset'
    else:
        model_directory = np.array2string(layers).replace('[','_').replace(']','').replace(' ','_')+'_l2_withweights_log_rescale_preset'
    path_model = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model'+model_directory+'/'
    path_plots = '/home/ucl/cp3/fbury/Memoire/MassPlane/learning_plots/model'+model_directory+'/'
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Optimizer #
    learning_rate = 0.01 
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    # Build model  #
    print ('[INFO] Starting Learning')
    if L2 != 0:
        print ('L2 Regularization = ',str(L2))
    DNN = Model(inputs=[inputs], outputs=[Dx])

    # Pre-set the weights #
    with open('/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_l2/model_step_1.json', "r") as json_model_file:
        model_json_save = json_model_file.read()
    model_save = model_from_json(model_json_save) 
    model_save.load_weights('/home/ucl/cp3/fbury/Memoire/MassPlane/learning_model/model_30_30_30_l2/weight_step1.h5')
    weights_layer1 = model_save.layers[1].get_weights()
    weights_layer2 = model_save.layers[2].get_weights()
    weights_layer3 = model_save.layers[3].get_weights()
    
    weights_layer1[0] = np.concatenate((weights_layer1[0],np.random.uniform(0,1,size=(2,30))),axis=0)

    del model_save
    del model_json_save

    DNN.layers[1].set_weights(weights_layer1)
    DNN.layers[2].set_weights(weights_layer2)
    DNN.layers[3].set_weights(weights_layer3)


    # Compile #

    DNN.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

    # Callback #
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    csv_logger = CSVLogger(path_model+'training_step_'+str(step)+'.log')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.001)
    checkpoint = ModelCheckpoint(path_model+'weight_step'+str(step)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    Callback_list = [csv_logger,checkpoint,reduceLR,early_stopping]


    # Fit #
    epoch = 100
    batch = 1000
    history = DNN.fit(X_train, T_train, sample_weight=W_train, epochs=epoch, batch_size=batch, verbose=2, validation_data=(X_test,T_test,W_test), callbacks=Callback_list)

    # Print history #

    # summarize history for accuracy 
    fig1, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=True)

    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['train', 'test'], loc='lower right')

    # summarize history for loss #
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'test'], loc='upper right')
    #plt.show()
    if print_plots: 
        fig1.savefig(path_plots+'loss+acc_step_'+str(step)+'.png',bbox_inches='tight')
        print ('[INFO] Accuracy and Loss plots saved as : '+path_plots+'loss+acc_step_'+str(step)+'.png')
    plt.close()

    # Error evaluation #
    print ('[INFO] Starting Evaluation')
    Y_test = DNN.predict(X_test)
    err = mean_squared_error(Y_test,T_test)

    # AUC score evaluation #
    false_positive_rate, true_positive_rate, thresholds = roc_curve(T_test, Y_test)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print ('RMS = %0.5f, AUC = %0.5f'%(err,roc_auc))

    # Plot ROC curve #
    bkgd_rej = 1-false_positive_rate
    sig_eff = true_positive_rate

    fig2 = plt.figure(2)
    plt.title('Receiver Operating Characteristic')
    line, = plt.plot(sig_eff, bkgd_rej, color='b', label=('AUC = %0.5f\nRMS = %0.5f'%(roc_auc,err)))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.legend(loc='lower left')
    #plt.show()
    if print_plots:
        fig2.savefig(path_plots+'roc_step_'+str(step)+'.png', bbox_inches='tight')
        print ('[INFO] ROC curve plot saved as : '+path_plots+'roc_step_'+str(step)+'.png') 
    plt.close()

    # Show DNN #
    if print_model:
        print ('[INFO] Scheme of the model')
        utils.print_summary(model=DNN)

    # Save Model #
    model_json = DNN.to_json()
    with open(path_model+'model_step_'+str(step)+'.json', "w") as json_file:
        json_file.write(model_json)
    # The weights are saved via the checkpoint callbakc
    print ('[INFO] Model saved as '+path_model+'model_step_'+str(step)+'.json')


    return err,roc_auc

