# Libraries #
import glob
import os
import math
import numpy as np
from array import array

import ROOT
from ROOT import TChain, TFile, TTree, TCanvas, TH2F, TPaveLabel

from root_numpy import fill_hist

def DrawMassPlane(data,mA,mH,title,**kwargs):
    """
    Draw the mass plane (mlljj,mjj) with our without a cut on the NN output
    Inputs :    -data : [mlljj,mjj]
    kwargs :    -prediciton : output of the NN for data
                -cut : cut on the NN output
    Output : plot the Mass Plane
    """

    c1 = TCanvas( 'c1', 'MassPlane', 200, 10, 700, 500 )
    c1.SetFillColor( 10 )
    c1.GetFrame().SetFillColor( 1 )
    c1.GetFrame().SetBorderSize( 6 )
    c1.GetFrame().SetBorderMode( -1 )

    #mass_plane = TH2F( 'mass_plane', 'M_{lljj} vs M_{jj};M_{jj};M_{lljj}',100,mA-50,mA+50, 100,mH-50,mH+50)
    mass_plane = TH2F( 'mass_plane', title+';M_{jj};M_{lljj}',100,0,2000, 100,0,2000)

    if kwargs != {}:
        #print (kwargs)
        pred = kwargs.get("prediction")
        cut = kwargs.get('cut')
        mask = np.squeeze(pred>cut) # Only takes the events satisfying the NN output cut
        print ('Number of events kept after cut = %i (%0.2f%%)'%(data[mask].shape[0],(data[mask].shape[0]/data.shape[0])*100))
        data = data[mask]

    fill_hist(mass_plane,data)

    mass_plane.Draw('COLZ')
    #c1.Modified()
    c1.Update()
    title = title.replace(' ','_')
    c1.Print('massplane_plots/'+title+'.pdf')

    input("Press enter to end")
