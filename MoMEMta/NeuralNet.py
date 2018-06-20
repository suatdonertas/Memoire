# Libraries #
import sys 
import glob
import os
import math
import numpy as np
import timeit

import ROOT
from ROOT import TFile, TTree, TCanvas
from root_numpy import tree2array,array2root

import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU
from keras.models import Model, model_from_json, load_model
from keras import losses, optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint, History, Callback
from keras.models import model_from_json
from keras.regularizers import l1,l2
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

import matplotlib.pyplot as plt

# Personal Libraries #
sys.path.append('/home/ucl/cp3/fbury/Memoire/MassPlane/')
from NNOutput import *

class NeuralNet:
    """ A class to build a specific neural network and use it (BuildModel must be called for the others to work""" 
    def __init__(self,data,target,masses,weight,label_model,label_target,model_number):
        print ('[INFO] Starting Initialization')
        K.clear_session()
        self.data = data
        # [Pt,Eta,Delta_Phi] x 4particles (no phi for first lepton) + [met_pt,met_phi]
        self.masses = masses 
        # [jj_M,ll_M,lljj_M]
        self.weight = weight
        self.label_model = label_model
        self.label_target = label_target
        self.model_number = model_number
        self.target = target

        # Make directory for the model #
        #model_directory = '/home/ucl/cp3/fbury/Memoire/MoMEMta/models/'+self.label_model+'/'
        model_directory = '/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/'+self.label_model+'/'

        self.path_model =  model_directory #+ '/model/'
        self.path_plots =  model_directory #+ '/plots/'
        self.path_output =  model_directory
        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_model)

    def BuildModel(self):
        """ Build the neural network"""
        print ('[INFO] Starting Building')
        # Splitting the dataset #
        X_train, X_test, T_train, T_test, W_train, W_test, M_train, M_test = train_test_split(self.data,self.target,self.weight,self.masses,test_size=0.25,shuffle=False)
        # X=data, T=target, W=weight (for learning), M=masses (mjj,mll,mlljj)

        W_train = W_train.reshape(-1,)
        W_test = W_test.reshape(-1,)


        # Building layers #
        inputs = Input(shape=(X_train.shape[1],))
        L1 = Dense(50, activation='relu')(inputs)
        L2 = Dense(50, activation='relu')(L1)
        L3 = Dense(50, activation='relu')(L2)
        #LM1 = Dense(1, activation='relu')(L3)
        #LM2 = Dense(1, activation='relu')(L3)
        #LM3 = Dense(1, activation='relu')(L3)
        #concL3LM = Concatenate(axis=-1)([L3,LM1,LM2,LM3])
        L4 = Dense(50, activation='relu')(L3)
        #L4 = Dense(50, activation='relu')(concL3LM)
        #L5 = Dense(50, activation=arelu')(L4)
        Dx = Dense(1, activation='selu')(L4)

        # Optimizer #
        learning_rate = 0.001 
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False,clipvalue=0.5)

        # Define model #
        #self.DNN = Model(inputs=[inputs], outputs=[Dx,LM1,LM2,LM3])
        self.DNN = Model(inputs=[inputs], outputs=[Dx])

        # Pre set the weights #
        #with open('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/'+self.label_target+'model.json', "r") as json_model_file:
        #    model_json_save = json_model_file.read()
        #model_save = model_from_json(model_json_save) # load model with best weights
        #model_save.load_weights('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/4lay50_inv_trainmass/'+self.label_target+'weight.h5')
        #weights_layer1 = model_save.layers[1].get_weights()
        #weights_layer2 = model_save.layers[2].get_weights()
        #weights_layer3 = model_save.layers[3].get_weights()
        #weights_layer4 = model_save.layers[8].get_weights()
        
        #weights_layer4[0] = np.concatenate((weights_layer4[0],np.random.uniform(0,1,size=(3,50))),axis=0)
        #weights_layer4[0] = weights_layer4[0][:50,:]

        #del model_save
        #del model_json_save

        #self.DNN.layers[1].set_weights(weights_layer1)
        #self.DNN.layers[2].set_weights(weights_layer2)
        #self.DNN.layers[3].set_weights(weights_layer3)
        #self.DNN.layers[4].set_weights(weights_layer4)
        #self.DNN.layers[8].set_weights(weights_layer4)
        #for layer in model_save.layers: 
            #print(layer.get_config())
             
        # Compile #
        #self.DNN.compile(optimizer=adam,loss=['mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error'],metrics=['accuracy'],loss_weights=[1,0.01,0.01,0.01])
        self.DNN.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

        # Callback #
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='min')
        csv_logger = CSVLogger(self.path_model+'training_'+str(self.model_number)+'.log')
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, verbose=1, mode='min', epsilon=0.001, cooldown=3, min_lr=0.00001)
        checkpoint = ModelCheckpoint(self.path_model+self.label_target+'weight_'+str(self.model_number)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

        # Loss history #
        class LossHistory(Callback):
            def on_train_begin(self, logs={}):
                self.losses_tot = np.array([])
                self.losses_out = np.array([])
                self.losses_mjj = np.array([])
                self.losses_mll = np.array([])
                self.losses_mlljj = np.array([])
                self.val_losses_tot = np.array([])
                self.val_losses_out = np.array([])
                self.val_losses_mjj = np.array([])
                self.val_losses_mll = np.array([])
                self.val_losses_mlljj = np.array([])

            def on_epoch_end(self, epoch, logs={}):
                self.losses_tot=np.append(self.losses_tot,logs.get('loss')) 
                self.losses_out=np.append(self.losses_out,logs.get('dense_8_loss')) 
                self.losses_mjj=np.append(self.losses_mjj,logs.get('dense_4_loss')) 
                self.losses_mll=np.append(self.losses_mll,logs.get('dense_5_loss')) 
                self.losses_mlljj=np.append(self.losses_mlljj,logs.get('dense_6_loss')) 
                self.val_losses_tot=np.append(self.val_losses_tot,logs.get('val_loss')) 
                self.val_losses_out=np.append(self.val_losses_out,logs.get('val_dense_8_loss')) 
                self.val_losses_mjj=np.append(self.val_losses_mjj,logs.get('val_dense_4_loss')) 
                self.val_losses_mll=np.append(self.val_losses_mll,logs.get('val_dense_5_loss')) 
                self.val_losses_mlljj=np.append(self.val_losses_mlljj,logs.get('val_dense_6_loss')) 


        self.history = LossHistory()

        # Fit #
        epoch = 500
        batch = 500

        Callback_list = [csv_logger,checkpoint,self.history,early_stopping,reduceLR]
        #self.DNN.fit(X_train, [T_train,M_train[:,0],M_train[:,1],M_train[:,2]], sample_weight=[W_train,W_train,W_train,W_train], epochs=epoch, batch_size=batch, verbose=2, validation_data=(X_test,[T_test,M_test[:,0],M_test[:,1],M_test[:,2]],[W_test,W_test,W_test,W_test]), callbacks=Callback_list)
        self.DNN.fit(X_train, T_train, sample_weight=W_train, epochs=epoch, batch_size=batch, verbose=2, validation_data=(X_test,T_test,W_test), callbacks=Callback_list)

        # Save model #
        model_json = self.DNN.to_json()
        with open(self.path_model+self.label_target+'model_'+str(self.model_number)+'.json', "w") as json_file:
            json_file.write(model_json)
        # The weights are saved via the checkpoint callbakc
        print ('[INFO] Model saved as '+self.path_model+self.label_target+'model_'+str(self.model_number)+'.json')

        # Evaluate MSE #
        self.DNN.load_weights(self.path_model+self.label_target+'weight_'+str(self.model_number)+'.h5')
        print ('Test MSE : ',mean_squared_error(T_test,self.DNN.predict(X_test),sample_weight=W_test))
        

    def PlotHistory(self):
        n_epoch = np.arange(1,self.history.losses_tot.shape[0]+1) 
   
        ## Loss
        fig1 = plt.figure(1,figsize=(12,15))
        fig1.tight_layout()

        ax1 = plt.subplot(511)
        ax2 = plt.subplot(512)
        ax3 = plt.subplot(513)
        ax4 = plt.subplot(514)
        ax5 = plt.subplot(515)
    
        plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=0.2, hspace=0.5)
        try:
            ax1.plot(n_epoch,self.history.losses_tot,'r',label='Training')
            ax1.plot(n_epoch,self.history.val_losses_tot,'b',label='Validation')
            ax1.set_title('Total loss')
            ax1.set_ylabel('Loss')
            ax1.set_yscale('symlog')
            ax1.legend(loc='upper right')
            ax1.grid(True)
            ax1.set_ylim(bottom=np.amin(np.minimum(self.history.losses_tot,self.history.val_losses_tot)),top=np.amax(np.maximum(self.history.losses_tot,self.history.val_losses_tot)))
        except: 
            print ('Could not find val_loss')
        
        try:
            ax2.plot(n_epoch,self.history.losses_out,'r',label='Training')
            ax2.plot(n_epoch,self.history.val_losses_out,'b',label='Validation')
            ax2.set_title('Output loss')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('symlog')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            ax2.set_ylim(bottom=np.amin(np.minimum(self.history.losses_out,self.history.val_losses_out)),top=np.amax(np.maximum(self.history.losses_out,self.history.val_losses_out)))
            
            ax3.plot(n_epoch,self.history.losses_mjj,'r',label='Training')
            ax3.plot(n_epoch,self.history.val_losses_mjj,'b',label='Validation')
            ax3.set_title('$M_{jj}$ loss')
            ax3.set_ylabel('Loss')
            ax3.set_yscale('symlog')
            ax3.legend(loc='upper right')
            ax3.grid(True)
            ax3.set_ylim(bottom=np.amin(np.minimum(self.history.losses_mjj,self.history.val_losses_mjj)),top=np.amax(np.maximum(self.history.losses_mjj,self.history.val_losses_mjj)))

            ax4.plot(n_epoch,self.history.losses_mll,'r',label='Training')
            ax4.plot(n_epoch,self.history.val_losses_mll,'b',label='Validation')
            ax4.set_title('$M_{ll}$ loss')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('symlog')
            ax4.legend(loc='upper right')
            ax4.grid(True)
            ax4.set_ylim(bottom=np.amin(np.minimum(self.history.losses_mll,self.history.val_losses_mll)),top=np.amax(np.maximum(self.history.losses_mll,self.history.val_losses_mll)))

            ax5.plot(n_epoch,self.history.losses_mlljj,'r',label='Training')
            ax5.plot(n_epoch,self.history.val_losses_mlljj,'b',label='Validation')
            ax5.set_title('$M_{lljj}$ loss')
            ax5.set_xlabel('Epochs')
            ax5.set_ylabel('Loss')
            ax5.set_yscale('symlog')
            ax5.legend(loc='upper right')
            ax5.grid(True)
            ax5.set_ylim(bottom=np.amin(np.minimum(self.history.losses_mlljj,self.history.val_losses_mlljj)),top=np.amax(np.maximum(self.history.losses_mlljj,self.history.val_losses_mlljj)))
        except:
            print ('Could not find other losses')
            print (self.history.losses_out)
            print (self.history.losses_mll)
            print (self.history.losses_mjj)
            print (self.history.losses_mlljj)
            print (self.history.val_losses_out)
            print (self.history.val_losses_mll)
            print (self.history.val_losses_mjj)
            print (self.history.val_losses_mlljj)

        fig1.savefig(self.path_plots+self.label_target+'lossplot_'+str(self.model_number)+'.png')

    def PrintModel(self):
        print ('[INFO] Starting Model Printing')
        # Show DNN #
        utils.print_summary(model=self.DNN)

    def UseModel(self,data,target,masses,other,label_output):
        #print ('[INFO] Starting Model Output : '+label_output)
        #with open(self.path_model+self.label_target+'model_'+str(self.model_number)+'.json', "r") as json_model_file:
        #    model_json_save = json_model_file.read()
        #model = model_from_json(model_json_save) # load model with best weights
        #model.load_weights(self.path_model+self.label_target+'weight_'+str(self.model_number)+'.h5')


        start_time = timeit.default_timer()
       
        NN_output = NNOutput(data,self.path_model+self.label_target) 
        #NN_output = np.asarray(model.predict(data))[:,:,0]
        #NN_output = np.asarray(model.predict(data))

        elapsed = timeit.default_timer() - start_time

        print (elapsed)
        print (NN_output.shape[0])

        #NN_output = np.transpose(NN_output)
        print (NN_output)
        #print (np.transpose(np.asarray(self.DNN.predict(self.set_test))[:,:,0]))
        #mse_best = mean_squared_error(NN_output[:,0],target)
        mse_best = mean_squared_error(NN_output,target)
        print ('mse_best',mse_best)
        model_output = np.c_[target,NN_output]
        # output = [MEM,NNOut]
        output = np.c_[data[:,:13],masses,other,model_output]
        
        #output.dtype = [('lep1_Pt','float64'),('lep1_Eta','float64'),('lep2_Pt','float64'),('lep2_Eta','float64'),('lep2_DPhi','float64'),('jet1_Pt','float64'),('jet1_Eta','float64'),('jet1_DPhi','float64'),('jet2_Pt','float64'),('jet2_Eta','float64'),('jet2_DPhi','float64'),('met_pt','float64'),('met_phi','float64'),('jj_M','float64'),('ll_M','float64'),('lljj_M','float64'),('mH','float64'),('mA','float64'),('visible_cross_section','float64'),('weight','float64'),('id','float64'),('original_MEM_TT','float64'),('original_MEM_DY','float64'),('MEM_'+self.label_target,'float64'),('NNOut_'+self.label_target,'float64'),('NN_jj_M','float64'),('NN_ll_M','float64'),('NN_lljj_M','float64')]
        output.dtype = [('lep1_Pt','float64'),('lep1_Eta','float64'),('lep2_Pt','float64'),('lep2_Eta','float64'),('lep2_DPhi','float64'),('jet1_Pt','float64'),('jet1_Eta','float64'),('jet1_DPhi','float64'),('jet2_Pt','float64'),('jet2_Eta','float64'),('jet2_DPhi','float64'),('met_pt','float64'),('met_phi','float64'),('jj_M','float64'),('ll_M','float64'),('lljj_M','float64'),('mH','float64'),('mA','float64'),('visible_cross_section','float64'),('original_MEM_TT_err','float64'),('original_MEM_DY_err','float64'),('weight','float64'),('id','float64'),('original_MEM_TT','float64'),('original_MEM_DY','float64'),('MEM_'+self.label_target,'float64'),('NNOut_'+self.label_target,'float64')]

        #output.dtype.names = ['lep1_Pt','lep1_Eta','lep2_Pt','lep2_Eta','lep2_DPhi','jet1_Pt','jet1_Eta','jet1_DPhi','jet2_Pt','jet2_Eta','jet2_DPhi','met_pt','met_phi','jj_M','ll_M','lljj_M','mH','mA','visible_cross_section','weight','id','original_MEM_TT','original_MEM_DY','MEM_'+self.label_target,'NNOut_'+self.label_target,'NN_jj_M','NN_ll_M','NN_lljj_M']
        output.dtype.names = ['lep1_Pt','lep1_Eta','lep2_Pt','lep2_Eta','lep2_DPhi','jet1_Pt','jet1_Eta','jet1_DPhi','jet2_Pt','jet2_Eta','jet2_DPhi','met_pt','met_phi','jj_M','ll_M','lljj_M','mH','mA','visible_cross_section','original_MEM_TT_err','original_MEM_DY_err','weight','id','original_MEM_TT','original_MEM_DY','MEM_'+self.label_target,'NNOut_'+self.label_target]

        array2root(output,self.path_output+'output'+self.label_target+label_output+'.root',mode='recreate')

        return mse_best
