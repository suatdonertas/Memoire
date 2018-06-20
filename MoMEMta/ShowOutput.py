# Libraries #
import sys 
import glob
import os
import re
import argparse
import math
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TPaveText, TPad, gPad, gStyle, TLegend, TH1F, gROOT, TLegend, TRatioPlot, TStyle, TF1, THStack, TColor, TImage, TLine, TPaveStats, TMath
from ROOT import kBlack, kBlue, kRed, kOrange, kYellow, kGreen
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error as MSE
from root_numpy import tree2array, array2tree, hist2array
import matplotlib.pyplot as plt

gStyle.SetOptStat("")

###############################################################################
# User's inputs #
###############################################################################
parser = argparse.ArgumentParser(description='Build Graph for given subest of mA,mH') 
parser.add_argument('-l','--label', help='Label of the model -> COMPULSORY',required=True,type=str)

args = parser.parse_args()

###############################################################################
# Import Root Files #
###############################################################################
print ('='*80)
path = '/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/'+args.label+'/'
if not os.path.exists(path):
    sys.exit('Can not find root file path')
print ('[INFO] Starting input from files')

f_TT = ROOT.TFile.Open(path+'outputTT.root')
t_TT = f_TT.Get("tree")
f_DY = ROOT.TFile.Open(path+'outputDY.root')
t_DY = f_DY.Get("tree")
f_TT_inv = ROOT.TFile.Open(path+'outputTTinvalid.root')
t_TT_inv = f_TT_inv.Get("tree")
f_DY_inv = ROOT.TFile.Open(path+'outputDYinvalid.root')
t_DY_inv = f_DY_inv.Get("tree")


#MyStyle = TStyle("MyStyle","My Style")
#MyStyle.SetCanvasBorderMode(0)
#MyStyle.SetCanvasColor(0)
#MyStyle.SetPadBorderMode(0)
#MyStyle.SetPadColor(0)
#MyStyle.SetPadBottomMargin(0.15)
#MyStyle.SetPadLeftMargin(0.15)
#MyStyle.SetPaperSize(18,24)
#MyStyle.SetLabelSize(0.03)
#MyStyle.SetLabelOffset(0.01,"Y")
#MyStyle.SetTitleOffset(2)
#MyStyle.SetTitleSize(0.06)
#MyStyle.SetStatFont(0)
#MyStyle.SetStatBorderSize(0)
#MyStyle.SetStatColor(0)
#MyStyle.SetStatFontSize(0.06)
#MyStyle.SetTitleBorderSize(0)
#MyStyle.SetTitleFont(0)
#MyStyle.SetTitleFontSize(0.08)

#MyStyle.SetTitleColor(0)
#MyStyle.SetOptStat(0)
gROOT.SetStyle("Plain")
gStyle.SetOptStat(0)

gStyle.SetTitleFontSize(.11)
gStyle.SetLabelSize(.05, "XY")

gStyle.SetPadBottomMargin(0.15)
gStyle.SetPadLeftMargin(0.15)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadTopMargin(0.15)


###############################################################################
# Extract branches #
##############################################################################
gROOT.SetBatch(True)
# Extract histograms #

t_TT.Draw("MEM_TT>>MEM_TT_sig(100,0,30)","id==0")
t_TT.Draw("MEM_TT>>MEM_TT_TT(100,0,30)","id==1")
t_TT.Draw("MEM_TT>>MEM_TT_DY(100,0,30)","id==2")
t_TT.Draw("NNOut_TT>>NNOut_TT_sig(100,0,30)","id==0")
t_TT.Draw("NNOut_TT>>NNOut_TT_TT(100,0,30)","id==1")
t_TT.Draw("NNOut_TT>>NNOut_TT_DY(100,0,30)","id==2")
t_DY.Draw("MEM_DY>>MEM_DY_sig(100,0,30)","id==0")
t_DY.Draw("MEM_DY>>MEM_DY_TT(100,0,30)","id==1")
t_DY.Draw("MEM_DY>>MEM_DY_DY(100,0,30)","id==2")
t_DY.Draw("NNOut_DY>>NNOut_DY_sig(100,0,30)","id==0")
t_DY.Draw("NNOut_DY>>NNOut_DY_TT(100,0,30)","id==1")
t_DY.Draw("NNOut_DY>>NNOut_DY_DY(100,0,30)","id==2")
MEM_TT_sig = gROOT.FindObject("MEM_TT_sig")
MEM_TT_TT = gROOT.FindObject("MEM_TT_TT")
MEM_TT_DY = gROOT.FindObject("MEM_TT_DY")
NNOut_TT_sig = gROOT.FindObject("NNOut_TT_sig")
NNOut_TT_TT = gROOT.FindObject("NNOut_TT_TT")
NNOut_TT_DY = gROOT.FindObject("NNOut_TT_DY")
MEM_DY_sig = gROOT.FindObject("MEM_DY_sig")
MEM_DY_TT = gROOT.FindObject("MEM_DY_TT")
MEM_DY_DY = gROOT.FindObject("MEM_DY_DY")
NNOut_DY_sig = gROOT.FindObject("NNOut_DY_sig")
NNOut_DY_TT = gROOT.FindObject("NNOut_DY_TT")
NNOut_DY_DY = gROOT.FindObject("NNOut_DY_DY")

t_TT.Draw("NNOut_TT:MEM_TT>>COMP_TT_sig(50,0,30,50,0,30)","id==0")
t_TT.Draw("NNOut_TT:MEM_TT>>COMP_TT_TT(50,0,30,50,0,30)","id==1")
t_TT.Draw("NNOut_TT:MEM_TT>>COMP_TT_DY(50,0,30,50,0,30)","id==2")
t_DY.Draw("NNOut_DY:MEM_DY>>COMP_DY_sig(50,0,20,50,0,20)","id==0")
t_DY.Draw("NNOut_DY:MEM_DY>>COMP_DY_TT(50,0,20,50,0,20)","id==1")
t_DY.Draw("NNOut_DY:MEM_DY>>COMP_DY_DY(50,0,20,50,0,20)","id==2")
COMP_TT_sig = gROOT.FindObject("COMP_TT_sig")
COMP_TT_TT = gROOT.FindObject("COMP_TT_TT")
COMP_TT_DY = gROOT.FindObject("COMP_TT_DY")
COMP_DY_sig = gROOT.FindObject("COMP_DY_sig")
COMP_DY_TT = gROOT.FindObject("COMP_DY_TT")
COMP_DY_DY = gROOT.FindObject("COMP_DY_DY")

MEM_TT_sig.SetLineColor(kGreen+2)
MEM_TT_TT.SetLineColor(kGreen+2)
MEM_TT_DY.SetLineColor(kGreen+2)
MEM_DY_sig.SetLineColor(kGreen+2)
MEM_DY_TT.SetLineColor(kGreen+2)
MEM_DY_DY.SetLineColor(kGreen+2)
MEM_TT_sig.SetLineWidth(2)
MEM_TT_TT.SetLineWidth(2)
MEM_TT_DY.SetLineWidth(2)
MEM_DY_sig.SetLineWidth(2)
MEM_DY_TT.SetLineWidth(2)
MEM_DY_DY.SetLineWidth(2)

NNOut_TT_sig.SetLineColor(kBlue+3)
NNOut_TT_TT.SetLineColor(kBlue+3)
NNOut_TT_DY.SetLineColor(kBlue+3)
NNOut_DY_sig.SetLineColor(kBlue+3)
NNOut_DY_TT.SetLineColor(kBlue+3)
NNOut_DY_DY.SetLineColor(kBlue+3)
NNOut_TT_sig.SetLineWidth(2)
NNOut_TT_TT.SetLineWidth(2)
NNOut_TT_DY.SetLineWidth(2)
NNOut_DY_sig.SetLineWidth(2)
NNOut_DY_TT.SetLineWidth(2)
NNOut_DY_DY.SetLineWidth(2)

legend_TT_sig = TLegend(0.50,0.70,0.9,0.9)
legend_TT_sig.SetHeader("Legend","C")
legend_TT_sig.AddEntry(MEM_TT_sig,"MoMEMta")
legend_TT_sig.AddEntry(NNOut_TT_sig,"Neural Network")

legend_TT_TT = TLegend(0.50,0.70,0.9,0.9)
legend_TT_TT.SetHeader("Legend","C")
legend_TT_TT.AddEntry(MEM_TT_TT,"MoMEMta")
legend_TT_TT.AddEntry(NNOut_TT_TT,"Neural Network")

legend_TT_DY = TLegend(0.50,0.70,0.9,0.9)
legend_TT_DY.SetHeader("Legend","C")
legend_TT_DY.AddEntry(MEM_TT_DY,"MoMEMta")
legend_TT_DY.AddEntry(NNOut_TT_DY,"Neural Network")

legend_DY_sig = TLegend(0.50,0.70,0.9,0.9)
legend_DY_sig.SetHeader("Legend","C")
legend_DY_sig.AddEntry(MEM_TT_sig,"MoMEMta")
legend_DY_sig.AddEntry(NNOut_TT_sig,"Neural Network")

legend_DY_TT = TLegend(0.50,0.70,0.9,0.9)
legend_DY_TT.SetHeader("Legend","C")
legend_DY_TT.AddEntry(MEM_TT_TT,"MoMEMta")
legend_DY_TT.AddEntry(NNOut_TT_TT,"Neural Network")

legend_DY_DY = TLegend(0.50,0.70,0.9,0.9)
legend_DY_DY.SetHeader("Legend","C")
legend_DY_DY.AddEntry(MEM_TT_DY,"MoMEMta")
legend_DY_DY.AddEntry(NNOut_TT_DY,"Neural Network")


###############################################################################
# Output Comparison  #
###############################################################################
gROOT.SetBatch(True)
c1 = TCanvas( 'c1', 'Output comparison', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("MoMEMta and NN output plots")
title.SetFillColor(0)
title.Draw()

TT_weight = TPaveText(0.01,0.05,0.04,0.45,"blNDC")
text_TT = TT_weight.AddText("TT weight")
text_TT.SetTextAngle(90.)
text_TT.SetTextFont(43)
text_TT.SetTextSize(30)
text_TT.SetTextAlign(22)
TT_weight.SetFillColor(0)
TT_weight.SetBorderSize(1)
TT_weight.Draw()

DY_weight = TPaveText(0.01,0.5,0.04,0.9,"blNDC")
text_DY = DY_weight.AddText("DY weight")
text_DY.SetTextFont(43)
text_DY.SetTextSize(30)
text_DY.SetTextAngle(90.)
text_DY.SetTextAlign(22)
DY_weight.SetFillColor(0)
DY_weight.SetBorderSize(1)
DY_weight.Draw()

pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.01, 0.37, 0.46, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0.01, 0.67, 0.46, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.65, 0.01, 0.97, 0.46, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.46, 0.37, 0.9, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.46, 0.67, 0.9, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.65, 0.46, 0.97, 0.9, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()


pad1.cd()
MEM_TT_sig.Sumw2()
MEM_TT_sig.SetTitle("H->ZA sample;-log_{10}(Weight) [Normalized]")
rp_TT_sig = TRatioPlot(MEM_TT_sig,NNOut_TT_sig)
rp_TT_sig.Draw()
rp_TT_sig.GetLowerRefGraph().SetMinimum(0)
rp_TT_sig.GetLowerRefGraph().SetMaximum(2)
rp_TT_sig.GetLowerRefYaxis().SetTitle("Ratio")
rp_TT_sig.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_TT_sig.GetLowerRefYaxis().SetTitleSize(.06)
rp_TT_sig.GetUpperRefYaxis().SetTitle("Occurences")
rp_TT_sig.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_TT_sig.GetUpperRefYaxis().SetTitleSize(.06)
rp_TT_sig.GetLowYaxis().SetNdivisions(505)
rp_TT_sig.SetUpTopMargin(0.2)
rp_TT_sig.SetLeftMargin(0.15)
rp_TT_sig.SetUpBottomMargin(0.5)
rp_TT_sig.SetLowBottomMargin(0.5)
rp_TT_sig.SetLowTopMargin(0)
rp_TT_sig.SetSeparationMargin(0.01)


pad2.cd()
MEM_TT_TT.Sumw2()
MEM_TT_TT.SetTitle("TT sample;-log_{10}(Weight) [Normalized]")
rp_TT_TT = TRatioPlot(MEM_TT_TT,NNOut_TT_TT)
rp_TT_TT.Draw()
rp_TT_TT.GetLowerRefGraph().SetMinimum(0)
rp_TT_TT.GetLowerRefGraph().SetMaximum(2)
rp_TT_TT.GetLowerRefYaxis().SetTitle("Ratio")
rp_TT_TT.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_TT_TT.GetLowerRefYaxis().SetTitleSize(.06)
rp_TT_TT.GetUpperRefYaxis().SetTitle("Occurences")
rp_TT_TT.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_TT_TT.GetUpperRefYaxis().SetTitleSize(.06)
rp_TT_TT.GetLowYaxis().SetNdivisions(505)
rp_TT_TT.SetUpTopMargin(0.2)
rp_TT_TT.SetLeftMargin(0.15)
rp_TT_TT.SetUpBottomMargin(0.5)
rp_TT_TT.SetLowBottomMargin(0.5)
rp_TT_TT.SetLowTopMargin(0)
rp_TT_TT.SetSeparationMargin(0.01)


pad3.cd()
MEM_TT_DY.Sumw2()
MEM_TT_DY.SetTitle("DY sample;-log_{10}(Weight) [Normalized]")
rp_TT_DY = TRatioPlot(MEM_TT_DY,NNOut_TT_DY)
rp_TT_DY.Draw()
rp_TT_DY.GetLowerRefGraph().SetMinimum(0)
rp_TT_DY.GetLowerRefGraph().SetMaximum(2)
rp_TT_DY.GetLowerRefYaxis().SetTitle("Ratio")
rp_TT_DY.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_TT_DY.GetLowerRefYaxis().SetTitleSize(.06)
rp_TT_DY.GetUpperRefYaxis().SetTitle("Occurences")
rp_TT_DY.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_TT_DY.GetUpperRefYaxis().SetTitleSize(.06)
rp_TT_DY.GetLowYaxis().SetNdivisions(505)
rp_TT_DY.SetUpTopMargin(0.2)
rp_TT_DY.SetLeftMargin(0.15)
rp_TT_DY.SetUpBottomMargin(0.5)
rp_TT_DY.SetLowBottomMargin(0.5)
rp_TT_DY.SetLowTopMargin(0)
rp_TT_DY.SetSeparationMargin(0.01)



pad4.cd()
MEM_DY_sig.Sumw2()
MEM_DY_sig.SetTitle("H->ZA sample;-log_{10}(Weight) [Normalized]")
rp_DY_sig = TRatioPlot(MEM_DY_sig,NNOut_DY_sig)
rp_DY_sig.Draw()
rp_DY_sig.GetLowerRefGraph().SetMinimum(0)
rp_DY_sig.GetLowerRefGraph().SetMaximum(2)
rp_DY_sig.GetLowerRefYaxis().SetTitle("Ratio")
rp_DY_sig.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_DY_sig.GetLowerRefYaxis().SetTitleSize(.06)
rp_DY_sig.GetUpperRefYaxis().SetTitle("Occurences")
rp_DY_sig.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_DY_sig.GetUpperRefYaxis().SetTitleSize(.06)
rp_DY_sig.GetLowYaxis().SetNdivisions(505)
rp_DY_sig.SetUpTopMargin(0.2)
rp_DY_sig.SetLeftMargin(0.15)
rp_DY_sig.SetUpBottomMargin(0.5)
rp_DY_sig.SetLowBottomMargin(0.5)
rp_DY_sig.SetLowTopMargin(0)
rp_DY_sig.SetSeparationMargin(0.01)



pad5.cd()
MEM_DY_TT.Sumw2()
MEM_DY_TT.SetTitle("TT sample;-log_{10}(Weight) [Normalized]")
rp_DY_TT = TRatioPlot(MEM_DY_TT,NNOut_DY_TT)
rp_DY_TT.Draw()
rp_DY_TT.GetLowerRefGraph().SetMinimum(0)
rp_DY_TT.GetLowerRefGraph().SetMaximum(2)
rp_DY_TT.GetLowerRefYaxis().SetTitle("Ratio")
rp_DY_TT.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_DY_TT.GetLowerRefYaxis().SetTitleSize(.06)
rp_DY_TT.GetUpperRefYaxis().SetTitle("Occurences")
rp_DY_TT.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_DY_TT.GetUpperRefYaxis().SetTitleSize(.06)
rp_DY_TT.GetLowYaxis().SetNdivisions(505)
rp_DY_TT.SetUpTopMargin(0.2)
rp_DY_TT.SetLeftMargin(0.15)
rp_DY_TT.SetUpBottomMargin(0.5)
rp_DY_TT.SetLowBottomMargin(0.5)
rp_DY_TT.SetLowTopMargin(0)
rp_DY_TT.SetSeparationMargin(0.01)



pad6.cd()
MEM_DY_DY.Sumw2()
MEM_DY_DY.SetTitle("DY sample;-log_{10}(Weight) [Normalized]")
rp_DY_DY = TRatioPlot(MEM_DY_DY,NNOut_DY_DY)
rp_DY_DY.Draw()
rp_DY_DY.GetLowerRefGraph().SetMinimum(0)
rp_DY_DY.GetLowerRefGraph().SetMaximum(2)
rp_DY_DY.GetLowerRefYaxis().SetTitle("Ratio")
rp_DY_DY.GetLowerRefYaxis().SetTitleOffset(1.35)
rp_DY_DY.GetLowerRefYaxis().SetTitleSize(.06)
rp_DY_DY.GetUpperRefYaxis().SetTitle("Occurences")
rp_DY_DY.GetUpperRefYaxis().SetTitleOffset(1.35)
rp_DY_DY.GetUpperRefYaxis().SetTitleSize(.06)
rp_DY_DY.GetLowYaxis().SetNdivisions(505)
rp_DY_DY.SetUpTopMargin(0.2)
rp_DY_DY.SetLeftMargin(0.15)
rp_DY_DY.SetUpBottomMargin(0.5)
rp_DY_DY.SetLowBottomMargin(0.5)
rp_DY_DY.SetLowTopMargin(0)
rp_DY_DY.SetSeparationMargin(0.01)



gPad.Update()

c1.Print(path+"Output.png")
#input ("Press any key to end")


# Canvas for comparison plots #
gROOT.SetBatch(True)
c2 = TCanvas( 'c2', 'Output comparison', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Weight comparison plots")
title.SetFillColor(0)
title.Draw()

TT_weight.Draw()
DY_weight.Draw()

pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.01, 0.35, 0.46, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0.01, 0.66, 0.46, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.66, 0.01, 0.97, 0.46, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.46, 0.35, 0.91, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.46, 0.66, 0.91, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.66, 0.46, 0.97, 0.91, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

# Get RMS Errors #
array_TT_sig_MEM = tree2array(t_TT,branches='MEM_TT',selection='id==0')
array_TT_sig_NNOut = tree2array(t_TT,branches='NNOut_TT',selection='id==0')
array_TT_TT_MEM = tree2array(t_TT,branches='MEM_TT',selection='id==1')
array_TT_TT_NNOut = tree2array(t_TT,branches='NNOut_TT',selection='id==1')
array_TT_DY_MEM = tree2array(t_TT,branches='MEM_TT',selection='id==2')
array_TT_DY_NNOut = tree2array(t_TT,branches='NNOut_TT',selection='id==2')
array_DY_sig_MEM = tree2array(t_DY,branches='MEM_DY',selection='id==0')
array_DY_sig_NNOut = tree2array(t_DY,branches='NNOut_DY',selection='id==0')
array_DY_TT_MEM = tree2array(t_DY,branches='MEM_DY',selection='id==1')
array_DY_TT_NNOut = tree2array(t_DY,branches='NNOut_DY',selection='id==1')
array_DY_DY_MEM = tree2array(t_DY,branches='MEM_DY',selection='id==2')
array_DY_DY_NNOut = tree2array(t_DY,branches='NNOut_DY',selection='id==2')
err_TT_sig = MSE(array_TT_sig_MEM,array_TT_sig_NNOut)
err_TT_TT = MSE(array_TT_TT_MEM,array_TT_TT_NNOut)
err_TT_DY = MSE(array_TT_DY_MEM,array_TT_DY_NNOut)
err_DY_sig = MSE(array_DY_sig_MEM,array_DY_sig_NNOut)
err_DY_TT = MSE(array_DY_TT_MEM,array_DY_TT_NNOut)
err_DY_DY = MSE(array_DY_DY_MEM,array_DY_DY_NNOut)

# Plot the clouds #
pad1.cd()
pad1.SetRightMargin(0.25)
pad1.SetTopMargin(0.16)
COMP_TT_sig.Draw("COLZ")
COMP_TT_sig.SetTitle("H->ZA sample ;MoMEMta output;NN output;Occurences")
COMP_TT_sig.GetZaxis().SetTitleOffset(1.2)
COMP_TT_sig.GetXaxis().SetTitleSize(.06)
COMP_TT_sig.GetYaxis().SetTitleSize(.06)
COMP_TT_sig.GetZaxis().SetTitleSize(.06)
pad1_mse = TPaveText(.1,25,12,30,"NB")
pad1_mse.AddText("MSE = %0.5f"%err_TT_sig)
pad1_mse.SetBorderSize(1)
pad1_mse.SetFillColor(0)
pad1_mse.Draw()

pad2.cd()
pad2.SetRightMargin(0.25)
pad2.SetTopMargin(0.16)
COMP_TT_TT.Draw("COLZ")
COMP_TT_TT.SetTitle("TT sample;MoMEMta output;NN output;Occurences")
COMP_TT_TT.GetZaxis().SetTitleOffset(1.2)
COMP_TT_TT.GetXaxis().SetTitleSize(.06)
COMP_TT_TT.GetYaxis().SetTitleSize(.06)
COMP_TT_TT.GetZaxis().SetTitleSize(.06)
pad2_mse = TPaveText(.1,25,12,30,"NB")
pad2_mse.AddText("MSE = %0.5f"%err_TT_TT)
pad2_mse.SetBorderSize(1)
pad2_mse.SetFillColor(0)
pad2_mse.Draw()

pad3.cd()
pad3.SetRightMargin(0.25)
pad3.SetTopMargin(0.16)
COMP_TT_DY.Draw("COLZ")
COMP_TT_DY.SetTitle("DY sample;MoMEMta output;NN output;Occurences")
COMP_TT_DY.GetZaxis().SetTitleOffset(1.2)
COMP_TT_DY.GetXaxis().SetTitleSize(.06)
COMP_TT_DY.GetYaxis().SetTitleSize(.06)
COMP_TT_DY.GetZaxis().SetTitleSize(.06)
pad3_mse = TPaveText(.1,25,12,30,"NB")
pad3_mse.AddText("MSE = %0.5f"%err_TT_DY)
pad3_mse.SetFillColor(0)
pad3_mse.SetBorderSize(1)
pad3_mse.Draw()

pad4.cd()
pad4.SetRightMargin(0.25)
pad4.SetTopMargin(0.16)
COMP_DY_sig.Draw("COLZ")
COMP_DY_sig.SetTitle("H->ZA sample;MoMEMta output;NN output;Occurences")
COMP_DY_sig.GetZaxis().SetTitleOffset(1.2)
COMP_DY_sig.GetXaxis().SetTitleSize(.06)
COMP_DY_sig.GetYaxis().SetTitleSize(.06)
COMP_DY_sig.GetZaxis().SetTitleSize(.06)
pad4_mse = TPaveText(.1,17,8,20,"NB")
pad4_mse.AddText("MSE = %0.5f"%err_DY_sig)
pad4_mse.SetBorderSize(1)
pad4_mse.SetFillColor(0)
pad4_mse.Draw()

pad5.cd()
pad5.SetRightMargin(0.25)
pad5.SetTopMargin(0.16)
COMP_DY_TT.Draw("COLZ")
COMP_DY_TT.SetTitle("TT sample;MoMEMta output;NN output;Occurences")
COMP_DY_TT.GetZaxis().SetTitleOffset(1.2)
COMP_DY_TT.GetXaxis().SetTitleSize(.06)
COMP_DY_TT.GetYaxis().SetTitleSize(.06)
COMP_DY_TT.GetZaxis().SetTitleSize(.06)
pad5_mse = TPaveText(.1,17,8,20,"NB")
pad5_mse.AddText("MSE = %0.5f"%err_DY_TT)
pad5_mse.SetFillColor(0)
pad5_mse.SetBorderSize(1)
pad5_mse.Draw()

pad6.cd()
pad6.SetRightMargin(0.25)
pad6.SetTopMargin(0.16)
COMP_DY_DY.Draw("COLZ")
COMP_DY_DY.SetTitle("DY sample;MoMEMta output;NN output;Occurences")
COMP_DY_DY.GetZaxis().SetTitleOffset(1.2)
COMP_DY_DY.GetXaxis().SetTitleSize(.06)
COMP_DY_DY.GetYaxis().SetTitleSize(.06)
COMP_DY_DY.GetZaxis().SetTitleSize(.06)
pad6_mse = TPaveText(.1,17,8,20,"NB")
pad6_mse.AddText("MSE = %0.5f"%err_DY_DY)
pad6_mse.SetFillColor(0)
pad6_mse.SetBorderSize(1)
pad6_mse.Draw()


gPad.Update()
c2.Print(path+"Comparison.png")

#input ("Press any key to end")

###############################################################################
# Area plots #
###############################################################################

array_sig_MEM = np.c_[array_TT_sig_MEM,array_DY_sig_MEM]
array_sig_MEM.dtype = [('TT','float64'),('DY','float64')]
array_sig_MEM.dtype.names = ['TT','DY']
array_TT_MEM = np.c_[array_TT_TT_MEM,array_DY_TT_MEM]
array_TT_MEM.dtype = [('TT','float64'),('DY','float64')]
array_TT_MEM.dtype.names = ['TT','DY']
array_DY_MEM = np.c_[array_TT_DY_MEM,array_DY_DY_MEM]
array_DY_MEM.dtype = [('TT','float64'),('DY','float64')]
array_DY_MEM.dtype.names = ['TT','DY']
array_sig_NNOut = np.c_[array_TT_sig_NNOut,array_DY_sig_NNOut]
array_sig_NNOut.dtype = [('TT','float64'),('DY','float64')]
array_sig_NNOut.dtype.names = ['TT','DY']
array_TT_NNOut = np.c_[array_TT_TT_NNOut,array_DY_TT_NNOut]
array_TT_NNOut.dtype = [('TT','float64'),('DY','float64')]
array_TT_NNOut.dtype.names = ['TT','DY']
array_DY_NNOut = np.c_[array_TT_DY_NNOut,array_DY_DY_NNOut]
array_DY_NNOut.dtype = [('TT','float64'),('DY','float64')]
array_DY_NNOut.dtype.names = ['TT','DY']


tree_sig_MEM = array2tree(array_sig_MEM)
tree_TT_MEM = array2tree(array_TT_MEM)
tree_DY_MEM = array2tree(array_DY_MEM)
tree_sig_NNOut = array2tree(array_sig_NNOut)
tree_TT_NNOut = array2tree(array_TT_NNOut)
tree_DY_NNOut = array2tree(array_DY_NNOut)

tree_sig_MEM.Draw("TT:DY>>hist_sig_MEM(50,0,30,50,0,30)")
hist_sig_MEM = gROOT.FindObject("hist_sig_MEM")
tree_TT_MEM.Draw("TT:DY>>hist_TT_MEM(50,0,30,50,0,30)")
hist_TT_MEM = gROOT.FindObject("hist_TT_MEM")
tree_DY_MEM.Draw("TT:DY>>hist_DY_MEM(50,0,30,50,0,30)")
hist_DY_MEM = gROOT.FindObject("hist_DY_MEM")
tree_sig_NNOut.Draw("TT:DY>>hist_sig_NNOut(50,0,30,50,0,30)")
hist_sig_NNOut = gROOT.FindObject("hist_sig_NNOut")
tree_TT_NNOut.Draw("TT:DY>>hist_TT_NNOut(50,0,30,50,0,30)")
hist_TT_NNOut = gROOT.FindObject("hist_TT_NNOut")
tree_DY_NNOut.Draw("TT:DY>>hist_DY_NNOut(50,0,30,50,0,30)")
hist_DY_NNOut = gROOT.FindObject("hist_DY_NNOut")


gROOT.SetBatch(True)


c3 = TCanvas( 'c3', 'TT Area plot', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Area plots")
title.SetFillColor(0)
title.Draw()

NN_area = TPaveText(0.01,0.05,0.04,0.45)
text_NN = NN_area.AddText("NN Area")
text_NN.SetTextAngle(90.)
text_NN.SetTextFont(43)
text_NN.SetTextSize(30)
text_NN.SetTextAlign(22)
NN_area.SetFillColor(0)
NN_area.Draw()

MEM_area = TPaveText(0.01,0.5,0.04,0.9)
text_MEM = MEM_area.AddText("MEM area")
text_MEM.SetTextFont(43)
text_MEM.SetTextSize(30)
text_MEM.SetTextAngle(90.)
text_MEM.SetTextAlign(22)
MEM_area.SetFillColor(0)
MEM_area.Draw()


pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.01, 0.35, 0.46, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0.01, 0.66, 0.46, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.66, 0.01, 0.97, 0.46, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.46, 0.35, 0.91, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.46, 0.66, 0.91, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.66, 0.46, 0.97, 0.91, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

pad1.cd()
pad1.SetRightMargin(0.25)
pad1.SetTopMargin(0.16)
hist_sig_NNOut.Draw("COLZ")
hist_sig_NNOut.SetTitle("H-ZA sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized];  Occurences")
hist_sig_NNOut.GetZaxis().SetTitleOffset(1.2)
hist_sig_NNOut.GetXaxis().SetTitleSize(.05)
hist_sig_NNOut.GetYaxis().SetTitleSize(.05)
hist_sig_NNOut.GetZaxis().SetTitleSize(.05)

pad2.cd()
pad2.SetRightMargin(0.25)
pad2.SetTopMargin(0.16)
hist_TT_NNOut.Draw("COLZ")
hist_TT_NNOut.SetTitle("TT sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized]; Occurences")
hist_TT_NNOut.GetZaxis().SetTitleOffset(1.2)
hist_TT_NNOut.GetXaxis().SetTitleSize(.05)
hist_TT_NNOut.GetYaxis().SetTitleSize(.05)
hist_TT_NNOut.GetZaxis().SetTitleSize(.05)

pad3.cd()
pad3.SetRightMargin(0.25)
pad3.SetTopMargin(0.16)
hist_DY_NNOut.Draw("COLZ")
hist_DY_NNOut.SetTitle("DY sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized]; Occurences")
hist_DY_NNOut.GetZaxis().SetTitleOffset(1.2)
hist_DY_NNOut.GetXaxis().SetTitleSize(.05)
hist_DY_NNOut.GetYaxis().SetTitleSize(.05)
hist_DY_NNOut.GetZaxis().SetTitleSize(.05)

pad4.cd()
pad4.SetRightMargin(0.25)
pad4.SetTopMargin(0.16)
hist_sig_MEM.Draw("COLZ")
hist_sig_MEM.SetTitle("H-ZA sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized]; Occurences")
hist_sig_MEM.GetZaxis().SetTitleOffset(1.2)
hist_sig_MEM.GetXaxis().SetTitleSize(.05)
hist_sig_MEM.GetYaxis().SetTitleSize(.05)
hist_sig_MEM.GetZaxis().SetTitleSize(.05)

pad5.cd()
pad5.SetRightMargin(0.25)
pad5.SetTopMargin(0.16)
hist_TT_MEM.Draw("COLZ")
hist_TT_MEM.SetTitle("TT sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized]; Occurences")
hist_TT_MEM.GetZaxis().SetTitleOffset(1.2)
hist_TT_MEM.GetXaxis().SetTitleSize(.05)
hist_TT_MEM.GetYaxis().SetTitleSize(.05)
hist_TT_MEM.GetZaxis().SetTitleSize(.05)

pad6.cd()
pad6.SetRightMargin(0.25)
pad6.SetTopMargin(0.16)
hist_DY_MEM.Draw("COLZ")
hist_DY_MEM.SetTitle("DY sample; -log_{10}(DY Weight) [Normalized];-log_{10}(TT Weight) [Normalized]; Occurences")
hist_DY_MEM.GetZaxis().SetTitleOffset(1.2)
hist_DY_MEM.GetXaxis().SetTitleSize(.05)
hist_DY_MEM.GetYaxis().SetTitleSize(.05)
hist_DY_MEM.GetZaxis().SetTitleSize(.05)

gPad.Update()

c3.Print(path+"Area.png")

#input ("Press any key to end")

###############################################################################
# Invalid plots #
###############################################################################
gROOT.SetBatch(True)
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_sig_inv_norm(50,0,30)","id==0","norm")
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_TT_inv_norm(50,0,30)","id==1","norm")
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_DY_inv_norm(50,0,30)","id==2","norm")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_sig_inv_norm(50,0,30)","id==0","norm")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_TT_inv_norm(50,0,30)","id==1","norm")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_DY_inv_norm(50,0,30)","id==2","norm")
NNOut_TT_sig_inv_norm = gROOT.FindObject("NNOut_TT_sig_inv_norm")
NNOut_TT_TT_inv_norm = gROOT.FindObject("NNOut_TT_TT_inv_norm")
NNOut_TT_DY_inv_norm = gROOT.FindObject("NNOut_TT_DY_inv_norm")
NNOut_DY_sig_inv_norm = gROOT.FindObject("NNOut_DY_sig_inv_norm")
NNOut_DY_TT_inv_norm = gROOT.FindObject("NNOut_DY_TT_inv_norm")
NNOut_DY_DY_inv_norm = gROOT.FindObject("NNOut_DY_DY_inv_norm")

t_TT.Draw("NNOut_TT>>NNOut_TT_sig_norm(100,0,30)","id==0","norm")
t_TT.Draw("NNOut_TT>>NNOut_TT_TT_norm(100,0,30)","id==1","norm")
t_TT.Draw("NNOut_TT>>NNOut_TT_DY_norm(100,0,30)","id==2","norm")
t_DY.Draw("NNOut_DY>>NNOut_DY_sig_norm(100,0,30)","id==0","norm")
t_DY.Draw("NNOut_DY>>NNOut_DY_TT_norm(100,0,30)","id==1","norm")
t_DY.Draw("NNOut_DY>>NNOut_DY_DY_norm(100,0,30)","id==2","norm")
NNOut_TT_sig_norm = gROOT.FindObject("NNOut_TT_sig_norm")
NNOut_TT_TT_norm = gROOT.FindObject("NNOut_TT_TT_norm")
NNOut_TT_DY_norm = gROOT.FindObject("NNOut_TT_DY_norm")
NNOut_DY_sig_norm = gROOT.FindObject("NNOut_DY_sig_norm")
NNOut_DY_TT_norm = gROOT.FindObject("NNOut_DY_TT_norm")
NNOut_DY_DY_norm = gROOT.FindObject("NNOut_DY_DY_norm")


NNOut_TT_sig_inv_norm.SetLineColor(kRed+2)
NNOut_TT_sig_inv_norm.SetLineWidth(2)
NNOut_TT_TT_inv_norm.SetLineColor(kRed+2)
NNOut_TT_TT_inv_norm.SetLineWidth(2)
NNOut_TT_DY_inv_norm.SetLineColor(kRed+2)
NNOut_TT_DY_inv_norm.SetLineWidth(2)
NNOut_DY_sig_inv_norm.SetLineColor(kRed+2)
NNOut_DY_sig_inv_norm.SetLineWidth(2)
NNOut_DY_TT_inv_norm.SetLineColor(kRed+2)
NNOut_DY_TT_inv_norm.SetLineWidth(2)
NNOut_DY_DY_inv_norm.SetLineColor(kRed+2)
NNOut_DY_DY_inv_norm.SetLineWidth(2)

NNOut_TT_sig_norm.SetLineColor(kBlue+3)
NNOut_TT_sig_norm.SetLineWidth(2)
NNOut_TT_TT_norm.SetLineColor(kBlue+3)
NNOut_TT_TT_norm.SetLineWidth(2)
NNOut_TT_DY_norm.SetLineColor(kBlue+3)
NNOut_TT_DY_norm.SetLineWidth(2)
NNOut_DY_sig_norm.SetLineColor(kBlue+3)
NNOut_DY_sig_norm.SetLineWidth(2)
NNOut_DY_TT_norm.SetLineColor(kBlue+3)
NNOut_DY_TT_norm.SetLineWidth(2)
NNOut_DY_DY_norm.SetLineColor(kBlue+3)
NNOut_DY_DY_norm.SetLineWidth(2)

legend_TT_sig_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_TT_sig_inv_norm.SetHeader("Legend","C")
legend_TT_sig_inv_norm.AddEntry(NNOut_TT_sig_norm,"Valid weights : %0.f entries"%(NNOut_TT_sig_norm.GetEntries()))
legend_TT_sig_inv_norm.AddEntry(NNOut_TT_sig_inv_norm,"Invalid weights : %0.f entries"%(NNOut_TT_sig_inv_norm.GetEntries()))

legend_TT_TT_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_TT_TT_inv_norm.SetHeader("Legend","C")
legend_TT_TT_inv_norm.AddEntry(NNOut_TT_TT_norm,"Valid weights : %0.f entries"%(NNOut_TT_TT_norm.GetEntries()))
legend_TT_TT_inv_norm.AddEntry(NNOut_TT_TT_inv_norm,"Invalid weights : %0.f entries"%(NNOut_TT_TT_inv_norm.GetEntries()))

legend_TT_DY_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_TT_DY_inv_norm.SetHeader("Legend","C")
legend_TT_DY_inv_norm.AddEntry(NNOut_TT_DY_norm,"Valid weights : %0.f entries"%(NNOut_TT_DY_norm.GetEntries()))
legend_TT_DY_inv_norm.AddEntry(NNOut_TT_DY_inv_norm,"Invalid weights : %0.f entries"%(NNOut_TT_DY_inv_norm.GetEntries()))

legend_DY_sig_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_DY_sig_inv_norm.SetHeader("Legend","C")
legend_DY_sig_inv_norm.AddEntry(NNOut_DY_sig_norm,"Valid weights : %0.f entries"%(NNOut_DY_sig_norm.GetEntries()))
legend_DY_sig_inv_norm.AddEntry(NNOut_DY_sig_inv_norm,"Invalid weights : %0.f entries"%(NNOut_DY_sig_inv_norm.GetEntries()))

legend_DY_TT_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_DY_TT_inv_norm.SetHeader("Legend","C")
legend_DY_TT_inv_norm.AddEntry(NNOut_DY_TT_norm,"Valid weights : %0.f entries"%(NNOut_DY_TT_norm.GetEntries()))
legend_DY_TT_inv_norm.AddEntry(NNOut_DY_TT_inv_norm,"Invalid weights : %0.f entries"%(NNOut_DY_TT_inv_norm.GetEntries()))

legend_DY_DY_inv_norm = TLegend(0.35,0.65,0.85,0.85)
legend_DY_DY_inv_norm.SetHeader("Legend","C")
legend_DY_DY_inv_norm.AddEntry(NNOut_DY_DY_norm,"Valid weights : %0.f entries"%(NNOut_DY_DY_norm.GetEntries()))
legend_DY_DY_inv_norm.AddEntry(NNOut_DY_DY_inv_norm,"Invalid weights : %0.f entries"%(NNOut_DY_DY_inv_norm.GetEntries()))


c4 = TCanvas( 'c4', 'Invalid plot', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Comparison of valid and invalid weights")
title.SetFillColor(0)
title.Draw()

TT_weight.Draw()
DY_weight.Draw()

pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.01, 0.35, 0.46, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0.01, 0.66, 0.46, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.66, 0.01, 0.97, 0.46, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.46, 0.35, 0.91, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.46, 0.66, 0.91, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.66, 0.46, 0.97, 0.91, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

pad1.cd()
pad1.SetLeftMargin(0.15)
NNOut_TT_sig_norm.Draw()
NNOut_TT_sig_inv_norm.Draw("same")
NNOut_TT_sig_norm.SetTitle("H->ZA sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_sig_norm.SetMaximum(max(NNOut_TT_sig_norm.GetMaximum(),NNOut_TT_sig_inv_norm.GetMaximum()))
NNOut_TT_sig_norm.GetXaxis().SetTitleSize(.05)
NNOut_TT_sig_norm.GetYaxis().SetTitleSize(.05)
NNOut_TT_sig_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_TT_sig_norm.GetYaxis().SetTitleOffset(1.3)
legend_TT_sig_inv_norm.Draw()

gPad.Update()

pad2.cd()
pad2.SetLeftMargin(0.15)
NNOut_TT_TT_norm.Draw()
NNOut_TT_TT_inv_norm.Draw("same")
NNOut_TT_TT_norm.SetTitle("TT sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_TT_norm.SetMaximum(max(NNOut_TT_TT_norm.GetMaximum(),NNOut_TT_TT_inv_norm.GetMaximum()))
NNOut_TT_TT_norm.GetXaxis().SetTitleSize(.05)
NNOut_TT_TT_norm.GetYaxis().SetTitleSize(.05)
NNOut_TT_TT_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_TT_TT_norm.GetYaxis().SetTitleOffset(1.3)
legend_TT_TT_inv_norm.Draw()

gPad.Update()

pad3.cd()
pad3.SetLeftMargin(0.15)
NNOut_TT_DY_norm.Draw()
NNOut_TT_DY_inv_norm.Draw("same")
NNOut_TT_DY_norm.SetTitle("DY sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_DY_norm.SetMaximum(max(NNOut_TT_DY_norm.GetMaximum(),NNOut_TT_DY_inv_norm.GetMaximum()))
legend_TT_DY_inv_norm.Draw()

gPad.Update()

pad4.cd()
pad4.SetLeftMargin(0.15)
NNOut_DY_sig_norm.Draw()
NNOut_DY_sig_inv_norm.Draw("same")
NNOut_DY_sig_norm.SetTitle("H->ZA sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_sig_norm.SetMaximum(max(NNOut_DY_sig_norm.GetMaximum(),NNOut_DY_sig_inv_norm.GetMaximum()))
NNOut_DY_sig_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_sig_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_sig_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_sig_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_sig_inv_norm.Draw()

gPad.Update()

pad5.cd()
pad5.SetLeftMargin(0.15)
NNOut_DY_TT_norm.Draw()
NNOut_DY_TT_inv_norm.Draw("same")
NNOut_DY_TT_norm.SetTitle("TT sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_TT_norm.SetMaximum(max(NNOut_DY_TT_norm.GetMaximum(),NNOut_DY_TT_inv_norm.GetMaximum()))
NNOut_DY_TT_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_TT_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_TT_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_TT_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_TT_inv_norm.Draw()

gPad.Update()

pad6.cd()
pad6.SetLeftMargin(0.15)
NNOut_DY_DY_norm.Draw()
NNOut_DY_DY_inv_norm.Draw("same")
NNOut_DY_DY_norm.SetTitle("DY sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_DY_norm.SetMaximum(max(NNOut_DY_DY_norm.GetMaximum(),NNOut_DY_DY_inv_norm.GetMaximum()))
NNOut_DY_DY_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_DY_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_DY_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_DY_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_DY_inv_norm.Draw()

gPad.Update()

c4.Print(path+"Invalid_comp.png")
#input ("Press any key to end")

###############################################################################
# Invalid sum #
###############################################################################
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_sig_inv(100,0,30)","id==0")
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_TT_inv(100,0,30)","id==1")
t_TT_inv.Draw("NNOut_TT>>NNOut_TT_DY_inv(100,0,30)","id==2")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_sig_inv(100,0,30)","id==0")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_TT_inv(100,0,30)","id==1")
t_DY_inv.Draw("NNOut_DY>>NNOut_DY_DY_inv(100,0,30)","id==2")
NNOut_TT_sig_inv = gROOT.FindObject("NNOut_TT_sig_inv")
NNOut_TT_TT_inv = gROOT.FindObject("NNOut_TT_TT_inv")
NNOut_TT_DY_inv = gROOT.FindObject("NNOut_TT_DY_inv")
NNOut_DY_sig_inv = gROOT.FindObject("NNOut_DY_sig_inv")
NNOut_DY_TT_inv = gROOT.FindObject("NNOut_DY_TT_inv")
NNOut_DY_DY_inv = gROOT.FindObject("NNOut_DY_DY_inv")

NNOut_TT_sig_inv.SetFillColor(kRed+2)
NNOut_TT_TT_inv.SetFillColor(kRed+2)
NNOut_TT_DY_inv.SetFillColor(kRed+2)
NNOut_DY_sig_inv.SetFillColor(kRed+2)
NNOut_DY_TT_inv.SetFillColor(kRed+2)
NNOut_DY_DY_inv.SetFillColor(kRed+2)

NNOut_TT_sig.SetFillColor(kBlue+3)
NNOut_TT_TT.SetFillColor(kBlue+3)
NNOut_TT_DY.SetFillColor(kBlue+3)
NNOut_DY_sig.SetFillColor(kBlue+3)
NNOut_DY_TT.SetFillColor(kBlue+3)
NNOut_DY_DY.SetFillColor(kBlue+3)

NNOut_TT_sig.SetLineColor(0)
NNOut_TT_sig.SetLineWidth(0)
NNOut_TT_sig_inv.SetLineColor(0)
NNOut_TT_sig_inv.SetLineWidth(0)
NNOut_TT_TT.SetLineColor(0)
NNOut_TT_TT.SetLineWidth(0)
NNOut_TT_TT_inv.SetLineColor(0)
NNOut_TT_TT_inv.SetLineWidth(0)
NNOut_TT_DY.SetLineColor(0)
NNOut_TT_DY.SetLineWidth(0)
NNOut_TT_DY_inv.SetLineColor(0)
NNOut_TT_DY_inv.SetLineWidth(0)
NNOut_DY_sig.SetLineColor(0)
NNOut_DY_sig.SetLineWidth(0)
NNOut_DY_sig_inv.SetLineColor(0)
NNOut_DY_sig_inv.SetLineWidth(0)
NNOut_DY_TT.SetLineColor(0)
NNOut_DY_TT.SetLineWidth(0)
NNOut_DY_TT_inv.SetLineColor(0)
NNOut_DY_TT_inv.SetLineWidth(0)
NNOut_DY_DY.SetLineColor(0)
NNOut_DY_DY.SetLineWidth(0)
NNOut_DY_DY_inv.SetLineColor(0)
NNOut_DY_DY_inv.SetLineWidth(0)

legend_TT_sig_inv = TLegend(0.35,0.65,0.85,0.85)
legend_TT_sig_inv.SetHeader("Legend","C")
legend_TT_sig_inv.AddEntry(NNOut_TT_sig,"Valid weights : %0.f entries"%(NNOut_TT_sig.GetEntries()))
legend_TT_sig_inv.AddEntry(NNOut_TT_sig_inv,"Invalid weights : %0.f entries"%(NNOut_TT_sig_inv.GetEntries()))
legend_TT_TT_inv = TLegend(0.35,0.65,0.85,0.85)
legend_TT_TT_inv.SetHeader("Legend","C")
legend_TT_TT_inv.AddEntry(NNOut_TT_TT,"Valid weights : %0.f entries"%(NNOut_TT_TT.GetEntries()))
legend_TT_TT_inv.AddEntry(NNOut_TT_TT_inv,"Invalid weights : %0.f entries"%(NNOut_TT_TT_inv.GetEntries()))
legend_TT_DY_inv = TLegend(0.35,0.65,0.85,0.85)
legend_TT_DY_inv.SetHeader("Legend","C")
legend_TT_DY_inv.AddEntry(NNOut_TT_DY,"Valid weights : %0.f entries"%(NNOut_TT_DY.GetEntries()))
legend_TT_DY_inv.AddEntry(NNOut_TT_DY_inv,"Invalid weights : %0.f entries"%(NNOut_TT_DY_inv.GetEntries()))
legend_DY_sig_inv = TLegend(0.35,0.65,0.85,0.85)
legend_DY_sig_inv.SetHeader("Legend","C")
legend_DY_sig_inv.AddEntry(NNOut_DY_sig,"Valid weights : %0.f entries"%(NNOut_DY_sig.GetEntries()))
legend_DY_sig_inv.AddEntry(NNOut_DY_sig_inv,"Invalid weights : %0.f entries"%(NNOut_DY_sig_inv.GetEntries()))
legend_DY_TT_inv = TLegend(0.35,0.65,0.85,0.85)
legend_DY_TT_inv.SetHeader("Legend","C")
legend_DY_TT_inv.AddEntry(NNOut_DY_TT,"Valid weights : %0.f entries"%(NNOut_DY_TT.GetEntries()))
legend_DY_TT_inv.AddEntry(NNOut_DY_TT_inv,"Invalid weights : %0.f entries"%(NNOut_DY_TT_inv.GetEntries()))
legend_DY_DY_inv = TLegend(0.35,0.65,0.85,0.85)
legend_DY_DY_inv.SetHeader("Legend","C")
legend_DY_DY_inv.AddEntry(NNOut_DY_DY,"Valid weights : %0.f entries"%(NNOut_DY_DY.GetEntries()))
legend_DY_DY_inv.AddEntry(NNOut_DY_DY_inv,"Invalid weights : %0.f entries"%(NNOut_DY_DY_inv.GetEntries()))


NNOut_TT_sig_stack = THStack("stack_TT_sig","")
NNOut_TT_sig_stack.Add(NNOut_TT_sig)
NNOut_TT_sig_stack.Add(NNOut_TT_sig_inv)
NNOut_TT_TT_stack = THStack("stack_TT_TT","")
NNOut_TT_TT_stack.Add(NNOut_TT_TT)
NNOut_TT_TT_stack.Add(NNOut_TT_TT_inv)
NNOut_TT_DY_stack = THStack("stack_TT_DY","")
NNOut_TT_DY_stack.Add(NNOut_TT_DY)
NNOut_TT_DY_stack.Add(NNOut_TT_DY_inv)
NNOut_DY_sig_stack = THStack("stack_DY_sig","")
NNOut_DY_sig_stack.Add(NNOut_DY_sig)
NNOut_DY_sig_stack.Add(NNOut_DY_sig_inv)
NNOut_DY_TT_stack = THStack("stack_DY_TT","")
NNOut_DY_TT_stack.Add(NNOut_DY_TT)
NNOut_DY_TT_stack.Add(NNOut_DY_TT_inv)
NNOut_DY_DY_stack = THStack("stack_DY_DY","")
NNOut_DY_DY_stack.Add(NNOut_DY_DY)
NNOut_DY_DY_stack.Add(NNOut_DY_DY_inv)



gROOT.SetBatch(True)
c5 = TCanvas( 'c5', 'Invalid Sum', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Sum of valid and invalid weights")
title.SetFillColor(0)
title.Draw()

TT_weight.Draw()
DY_weight.Draw()


pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.01, 0.35, 0.46, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0.01, 0.66, 0.46, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.66, 0.01, 0.97, 0.46, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.46, 0.35, 0.91, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.46, 0.66, 0.91, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.66, 0.46, 0.97, 0.91, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

pad1.cd()
pad1.SetLeftMargin(0.15)
NNOut_TT_sig_stack.Draw()
legend_TT_sig_inv.Draw()
NNOut_TT_sig_stack.SetTitle("H->ZA sample; -log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_sig_stack.GetXaxis().SetTitleSize(.05)
NNOut_TT_sig_stack.GetYaxis().SetTitleSize(.05)
NNOut_TT_sig_stack.GetXaxis().SetTitleOffset(1.3)
NNOut_TT_sig_stack.GetYaxis().SetTitleOffset(1.3)
#pad1.SetLogy()

pad2.cd()
NNOut_TT_TT_stack.Draw()
legend_TT_TT_inv.Draw()
NNOut_TT_TT_stack.SetTitle("TT sample; -log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_TT_stack.GetXaxis().SetTitleSize(.05)
NNOut_TT_TT_stack.GetYaxis().SetTitleSize(.05)
NNOut_TT_TT_stack.GetXaxis().SetTitleOffset(1.3)
NNOut_TT_TT_stack.GetYaxis().SetTitleOffset(1.3)
pad2.SetLogy()


pad3.cd()
NNOut_TT_DY_stack.Draw()
legend_TT_DY_inv.Draw()
NNOut_TT_DY_stack.SetTitle("DY sample; -log_{10}(Weight) [Normalized];Occurences")
NNOut_TT_DY_stack.GetXaxis().SetTitleSize(.05)
NNOut_TT_DY_stack.GetYaxis().SetTitleSize(.05)
NNOut_TT_DY_stack.GetXaxis().SetTitleOffset(1.3)
NNOut_TT_DY_stack.GetYaxis().SetTitleOffset(1.3)
pad3.SetLogy()

pad4.cd()
pad5.SetLeftMargin(0.15)
NNOut_DY_sig_norm.Draw()
NNOut_DY_sig_inv_norm.Draw("hist same")
NNOut_DY_sig_norm.SetTitle("H->ZA sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_sig_norm.SetMaximum(1.1*max(NNOut_DY_sig_norm.GetMaximum(),NNOut_DY_sig_inv_norm.GetMaximum()))
NNOut_DY_sig_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_sig_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_sig_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_sig_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_sig_inv_norm.Draw()

pad5.cd()
pad5.SetLeftMargin(0.15)
NNOut_DY_TT_norm.Draw()
NNOut_DY_TT_inv_norm.Draw("hist same")
NNOut_DY_TT_norm.SetTitle("TT sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_TT_norm.SetMaximum(1.1*max(NNOut_DY_TT_norm.GetMaximum(),NNOut_DY_TT_inv_norm.GetMaximum()))
NNOut_DY_TT_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_TT_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_TT_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_TT_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_TT_inv_norm.Draw()

pad6.cd()
pad6.SetLeftMargin(0.15)
NNOut_DY_DY_norm.Draw()
NNOut_DY_DY_inv_norm.Draw("hist same")
NNOut_DY_DY_norm.SetTitle("DY sample;-log_{10}(Weight) [Normalized];Occurences")
NNOut_DY_DY_norm.SetMaximum(1.1*max(NNOut_DY_DY_norm.GetMaximum(),NNOut_DY_DY_inv_norm.GetMaximum()))
NNOut_DY_DY_norm.GetXaxis().SetTitleSize(.05)
NNOut_DY_DY_norm.GetYaxis().SetTitleSize(.05)
NNOut_DY_DY_norm.GetXaxis().SetTitleOffset(1.3)
NNOut_DY_DY_norm.GetYaxis().SetTitleOffset(1.3)
legend_DY_DY_inv_norm.Draw()

#pad4.cd()
#NNOut_DY_sig_stack.Draw()
#legend_DY_sig_inv.Draw()
#NNOut_DY_sig_stack.SetTitle("H->ZA sample; -log_{10}(Weight) [Normalized];Occurences")
#NNOut_DY_sig_stack.GetXaxis().SetTitleSize(.05)
#NNOut_DY_sig_stack.GetYaxis().SetTitleSize(.05)
#NNOut_DY_sig_stack.GetXaxis().SetTitleOffset(1.3)
#NNOut_DY_sig_stack.GetYaxis().SetTitleOffset(1.3)
#pad4.SetLogy()

#pad5.cd()
#NNOut_DY_TT_stack.Draw()
#legend_DY_TT_inv.Draw()
#NNOut_DY_TT_stack.SetTitle("TT sample; -log_{10}(Weight) [Normalized];Occurences")
#NNOut_DY_TT_stack.GetXaxis().SetTitleSize(.05)
#NNOut_DY_TT_stack.GetYaxis().SetTitleSize(.05)
#NNOut_DY_TT_stack.GetXaxis().SetTitleOffset(1.3)
#NNOut_DY_TT_stack.GetYaxis().SetTitleOffset(1.3)
#pad5.SetLogy()

#pad6.cd()
#NNOut_DY_DY_stack.Draw()
#legend_DY_DY_inv.Draw()
#NNOut_DY_DY_stack.SetTitle("DY sample; -log_{10}(Weight) [Normalized];Occurences")
#NNOut_DY_DY_stack.GetXaxis().SetTitleSize(.05)
#NNOut_DY_DY_stack.GetYaxis().SetTitleSize(.05)
#NNOut_DY_DY_stack.GetXaxis().SetTitleOffset(1.3)
#NNOut_DY_DY_stack.GetYaxis().SetTitleOffset(1.3)
#pad6.SetLogy()

gPad.Update()

c5.Print(path+"Invalid_sum.png")
#input ("Press any key to end")



###############################################################################
# Event ratio plots #
###############################################################################
#gROOT.SetBatch(True)
c6 = TCanvas( 'c6','Ratio plot', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Ratio Plot event by event")
title.SetFillColor(0)
title.Draw()

TT_weight.Draw()
DY_weight.Draw()

# Recover normalization #
norm = np.genfromtxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/'+args.label+'/normalization.txt')

# Recover original weights #
#array_weight_TT_sig_MEM = np.power(10,-array_TT_sig_MEM)*norm[0]
#array_weight_TT_sig_NNOut = np.power(10,-array_TT_sig_NNOut)*norm[0]
#array_weight_TT_TT_MEM = np.power(10,-array_TT_TT_MEM)*norm[0]
#array_weight_TT_TT_NNOut = np.power(10,-array_TT_TT_NNOut)*norm[0]
#array_weight_TT_DY_MEM = np.power(10,-array_TT_DY_MEM)*norm[0]
#array_weight_TT_DY_NNOut = np.power(10,-array_TT_DY_NNOut)*norm[0]
#array_weight_DY_sig_MEM = np.power(10,-array_DY_sig_MEM)*norm[1]
#array_weight_DY_sig_NNOut = np.power(10,-array_DY_sig_NNOut)*norm[1]
#array_weight_DY_TT_MEM = np.power(10,-array_DY_TT_MEM)*norm[1]
#array_weight_DY_TT_NNOut = np.power(10,-array_DY_TT_NNOut)*norm[1]
#array_weight_DY_DY_MEM = np.power(10,-array_DY_DY_MEM)*norm[1]
#array_weight_DY_DY_NNOut = np.power(10,-array_DY_DY_NNOut)*norm[1]


ratio_TT_sig = np.divide(np.subtract(array_TT_sig_MEM,array_TT_sig_NNOut),array_TT_sig_MEM)
ratio_TT_TT = np.divide(np.subtract(array_TT_TT_MEM,array_TT_TT_NNOut),array_TT_TT_MEM)
ratio_TT_DY = np.divide(np.subtract(array_TT_DY_MEM,array_TT_DY_NNOut),array_TT_DY_MEM)
ratio_DY_sig = np.divide(np.subtract(array_DY_sig_MEM,array_DY_sig_NNOut),array_DY_sig_MEM)
ratio_DY_TT = np.divide(np.subtract(array_DY_TT_MEM,array_DY_TT_NNOut),array_DY_TT_MEM)
ratio_DY_DY = np.divide(np.subtract(array_DY_DY_MEM,array_DY_DY_NNOut),array_DY_DY_MEM)


ratio_TT_sig.dtype = [('ratio','float64')]
ratio_TT_sig.dtype.names = ['ratio']
ratio_TT_TT.dtype = [('ratio','float64')]
ratio_TT_TT.dtype.names = ['ratio']
ratio_TT_DY.dtype = [('ratio','float64')]
ratio_TT_DY.dtype.names = ['ratio']
ratio_DY_sig.dtype = [('ratio','float64')]
ratio_DY_sig.dtype.names = ['ratio']
ratio_DY_TT.dtype = [('ratio','float64')]
ratio_DY_TT.dtype.names = ['ratio']
ratio_DY_DY.dtype = [('ratio','float64')]
ratio_DY_DY.dtype.names = ['ratio']

tree_ratio_TT_sig = array2tree(ratio_TT_sig)
tree_ratio_TT_TT = array2tree(ratio_TT_TT)
tree_ratio_TT_DY = array2tree(ratio_TT_DY)
tree_ratio_DY_sig = array2tree(ratio_DY_sig)
tree_ratio_DY_TT = array2tree(ratio_DY_TT)
tree_ratio_DY_DY = array2tree(ratio_DY_DY)

tree_ratio_TT_sig.SetLineWidth(2)
tree_ratio_TT_sig.SetLineColor(4)
tree_ratio_TT_TT.SetLineWidth(2)
tree_ratio_TT_TT.SetLineColor(4)
tree_ratio_TT_DY.SetLineWidth(2)
tree_ratio_TT_DY.SetLineColor(4)
tree_ratio_DY_sig.SetLineWidth(2)
tree_ratio_DY_sig.SetLineColor(4)
tree_ratio_DY_TT.SetLineWidth(2)
tree_ratio_DY_TT.SetLineColor(4)
tree_ratio_DY_DY.SetLineWidth(2)
tree_ratio_DY_DY.SetLineColor(4)

gStyle.SetOptFit(1)
gStyle.SetStatW(0.22)
gStyle.SetStatH(0.25)
gStyle.SetStatY(1)
gStyle.SetStatX(1)

pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0.05, 0.345, 0.45, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.365, 0.05, 0.66, 0.45, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.68, 0.05, 0.975, 0.45, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.50, 0.345, 0.9, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.365, 0.50, 0.66, 0.9, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.68, 0.50, 0.975, 0.9, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

pad1.cd()
pad1.SetBottomMargin(0.2)
tree_ratio_TT_sig.Draw("ratio>>TF1_ratio_TT_sig(100,-2,2)")
TF1_ratio_TT_sig = gROOT.FindObject("TF1_ratio_TT_sig")
TF1_ratio_TT_sig.SetTitle('H->ZA sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_TT_sig.Fit("gaus","R","",-0.4,0.4)
fit1 = TF1_ratio_TT_sig.GetFunction("gaus")
fit1.Draw("same")
TF1_ratio_TT_sig.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_TT_sig.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_TT_sig.GetXaxis().SetTitleSize(.06)
TF1_ratio_TT_sig.GetYaxis().SetTitleSize(.06)
legend_ratio_TT_sig = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_TT_sig.SetHeader("Legend","C")
legend_ratio_TT_sig.AddEntry(TF1_ratio_TT_sig,"Distribution")
legend_ratio_TT_sig.AddEntry(fit1,"Gaussian Fit")
legend_ratio_TT_sig.Draw()

pad2.cd()
pad2.SetBottomMargin(0.2)
tree_ratio_TT_TT.Draw("ratio>>TF1_ratio_TT_TT(100,-2,2)")
TF1_ratio_TT_TT = gROOT.FindObject("TF1_ratio_TT_TT")
TF1_ratio_TT_TT.SetTitle('TT sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_TT_TT.Fit("gaus","R","",-0.8,0.5)
fit2 = TF1_ratio_TT_TT.GetFunction("gaus")
fit2.Draw("same")
TF1_ratio_TT_TT.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_TT_TT.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_TT_TT.GetXaxis().SetTitleSize(.06)
TF1_ratio_TT_TT.GetYaxis().SetTitleSize(.06)
legend_ratio_TT_TT = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_TT_TT.SetHeader("Legend","C")
legend_ratio_TT_TT.AddEntry(TF1_ratio_TT_TT,"Distribution")
legend_ratio_TT_TT.AddEntry(fit2,"Gaussian Fit")
legend_ratio_TT_TT.Draw()

pad3.cd()
pad3.SetBottomMargin(0.2)
tree_ratio_TT_DY.Draw("ratio>>TF1_ratio_TT_DY(100,-2,2)")
TF1_ratio_TT_DY = gROOT.FindObject("TF1_ratio_TT_DY")
TF1_ratio_TT_DY.SetTitle('DY sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_TT_DY.Fit("gaus","R","",-0.45,0.45)
fit3 = TF1_ratio_TT_DY.GetFunction("gaus")
fit3.Draw("same")
TF1_ratio_TT_DY.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_TT_DY.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_TT_DY.GetXaxis().SetTitleSize(.06)
TF1_ratio_TT_DY.GetYaxis().SetTitleSize(.06)
legend_ratio_TT_DY = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_TT_DY.SetHeader("Legend","C")
legend_ratio_TT_DY.AddEntry(TF1_ratio_TT_DY,"Distribution")
legend_ratio_TT_DY.AddEntry(fit3,"Gaussian Fit")
legend_ratio_TT_DY.Draw()


pad4.cd()
pad4.SetBottomMargin(0.2)
tree_ratio_DY_sig.Draw("ratio>>TF1_ratio_DY_sig(100,-2,2)")
TF1_ratio_DY_sig = gROOT.FindObject("TF1_ratio_DY_sig")
TF1_ratio_DY_sig.SetTitle('H->ZA sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_DY_sig.Fit("gaus","R","",-0.2,0.2)
fit4 = TF1_ratio_DY_sig.GetFunction("gaus")
fit4.Draw("same")
TF1_ratio_DY_sig.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_DY_sig.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_DY_sig.GetXaxis().SetTitleSize(.06)
TF1_ratio_DY_sig.GetYaxis().SetTitleSize(.06)
legend_ratio_DY_sig = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_DY_sig.SetHeader("Legend","C")
legend_ratio_DY_sig.AddEntry(TF1_ratio_DY_sig,"Distribution")
legend_ratio_DY_sig.AddEntry(fit4,"Gaussian Fit")
legend_ratio_DY_sig.Draw()


pad5.cd()
pad5.SetBottomMargin(0.2)
tree_ratio_DY_TT.Draw("ratio>>TF1_ratio_DY_TT(100,-2,2)")
TF1_ratio_DY_TT = gROOT.FindObject("TF1_ratio_DY_TT")
TF1_ratio_DY_TT.SetTitle('TT sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_DY_TT.Fit("gaus","R","",-0.2,0.2)
fit5 = TF1_ratio_DY_TT.GetFunction("gaus")
fit5.Draw("same")
TF1_ratio_DY_TT.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_DY_TT.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_DY_TT.GetXaxis().SetTitleSize(.06)
TF1_ratio_DY_TT.GetYaxis().SetTitleSize(.06)
legend_ratio_DY_TT = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_DY_TT.SetHeader("Legend","C")
legend_ratio_DY_TT.AddEntry(TF1_ratio_DY_TT,"Distribution")
legend_ratio_DY_TT.AddEntry(fit5,"Gaussian Fit")
legend_ratio_DY_TT.Draw()


pad6.cd()
pad6.SetBottomMargin(0.2)
tree_ratio_DY_DY.Draw("ratio>>TF1_ratio_DY_DY(100,-2,2)")
TF1_ratio_DY_DY = gROOT.FindObject("TF1_ratio_DY_DY")
TF1_ratio_DY_DY.SetTitle('DY sample; #frac{Y_{MEM}-Y_{NN}}{Y_{MEM}}; Occurences')
TF1_ratio_DY_DY.Fit("gaus","R","",-0.2,0.2)
fit6 = TF1_ratio_DY_DY.GetFunction("gaus")
fit6.Draw("same")
TF1_ratio_DY_DY.GetYaxis().SetTitleOffset(1.3)
TF1_ratio_DY_DY.GetXaxis().SetTitleOffset(1.3)
TF1_ratio_DY_DY.GetXaxis().SetTitleSize(.06)
TF1_ratio_DY_DY.GetYaxis().SetTitleSize(.06)
legend_ratio_DY_DY = TLegend(0.15,0.6,0.4,0.85)
legend_ratio_DY_DY.SetHeader("Legend","C")
legend_ratio_DY_DY.AddEntry(TF1_ratio_DY_DY,"Distribution")
legend_ratio_DY_DY.AddEntry(fit6,"Gaussian Fit")
legend_ratio_DY_DY.Draw()

gPad.Update()

c6.Print(path+"Ratio.png")

# Relative error #
c6.Clear()

title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("MoMEMta relative error")
title.SetFillColor(0)
title.Draw()

TT_weight.Draw()
DY_weight.Draw()

pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0, 0.4, 0.45, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0, 0.7, 0.45, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.65, 0, 1, 0.45, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.45, 0.4, 0.9, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.45, 0.7, 0.9, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.65, 0.45, 1, 0.9, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

bin_TT = "(100,0,0.04)"
bin_DY = "(100,0,0.003)"

str_TT = "(1/2.30258)*(original_MEM_TT_err/original_MEM_TT)/MEM_TT"
str_DY = "(1/2.30258)*(original_MEM_DY_err/original_MEM_DY)/MEM_DY"

array_err_TT_sig = tree2array(t_TT, branches=str_TT,selection='id==0')
array_err_TT_TT = tree2array(t_TT, branches=str_TT,selection='id==1')
array_err_TT_DY = tree2array(t_TT, branches=str_TT,selection='id==2')
array_err_DY_sig = tree2array(t_DY, branches=str_DY,selection='id==0')
array_err_DY_TT = tree2array(t_DY, branches=str_DY,selection='id==1')
array_err_DY_DY = tree2array(t_DY, branches=str_DY,selection='id==2')

quant_TT_sig = np.percentile(array_err_TT_sig,q=95)
quant_TT_TT = np.percentile(array_err_TT_TT,q=95)
quant_TT_DY = np.percentile(array_err_TT_DY,q=95)
quant_DY_sig = np.percentile(array_err_DY_sig,q=95)
quant_DY_TT = np.percentile(array_err_DY_TT,q=95)
quant_DY_DY = np.percentile(array_err_DY_DY,q=95)

pad1.cd()
pad1.SetBottomMargin(0.25)
pad1.SetRightMargin(0.2)
t_TT.Draw(str_TT+">>diff_TT_sig"+bin_TT,"id==0")
#t_TT.Draw("original_MEM_TT_err/original_MEM_TT>>diff_TT_sig"+bin_TT,"id==0")
diff_TT_sig= gROOT.FindObject("diff_TT_sig")
diff_TT_sig.SetLineWidth(2)
diff_TT_sig.SetTitle('H->ZA sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_TT_sig.GetYaxis().SetTitleOffset(1.1)
diff_TT_sig.GetXaxis().SetTitleOffset(1.35)
diff_TT_sig.GetXaxis().SetTitleSize(.06)
diff_TT_sig.GetYaxis().SetTitleSize(.06)
diff_TT_sig.GetXaxis().SetNdivisions(505)
pad1_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad1_quant.AddText("95%% quantile : %0.5f"%quant_TT_sig)
pad1_quant.SetBorderSize(1)
pad1_quant.SetFillColor(0)
pad1_quant.Draw()



pad2.cd()
pad2.SetBottomMargin(0.25)
pad2.SetRightMargin(0.2)
t_TT.Draw(str_TT+">>diff_TT_TT"+bin_TT,"id==1")
#t_TT.Draw("original_MEM_TT_err/original_MEM_TT>>diff_TT_TT"+bin_TT,"id==1")
diff_TT_TT= gROOT.FindObject("diff_TT_TT")
diff_TT_TT.SetLineWidth(2)
diff_TT_TT.SetTitle('TT sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_TT_TT.GetYaxis().SetTitleOffset(1.1)
diff_TT_TT.GetXaxis().SetTitleOffset(1.35)
diff_TT_TT.GetXaxis().SetTitleSize(.06)
diff_TT_TT.GetYaxis().SetTitleSize(.06)
diff_TT_TT.GetXaxis().SetNdivisions(505)
pad2_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad2_quant.AddText("95%% quantile : %0.5f"%quant_TT_TT)
pad2_quant.SetBorderSize(1)
pad2_quant.SetFillColor(0)
pad2_quant.Draw()


pad3.cd()
pad3.SetBottomMargin(0.25)
pad3.SetRightMargin(0.2)
t_TT.Draw(str_TT+">>diff_TT_DY"+bin_TT,"id==2")
#t_TT.Draw("original_MEM_TT_err/original_MEM_TT>>diff_TT_DY"+bin_TT,"id==2")
diff_TT_DY= gROOT.FindObject("diff_TT_DY")
diff_TT_DY.SetLineWidth(2)
diff_TT_DY.SetTitle('DY sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_TT_DY.GetYaxis().SetTitleOffset(1.1)
diff_TT_DY.GetXaxis().SetTitleOffset(1.35)
diff_TT_DY.GetXaxis().SetTitleSize(.06)
diff_TT_DY.GetYaxis().SetTitleSize(.06)
diff_TT_DY.GetXaxis().SetNdivisions(505)
pad3_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad3_quant.AddText("95%% quantile : %0.5f"%quant_TT_DY)
pad3_quant.SetBorderSize(1)
pad3_quant.SetFillColor(0)
pad3_quant.Draw()

pad4.cd()
pad4.SetBottomMargin(0.25)
pad4.SetRightMargin(0.2)
t_DY.Draw(str_DY+">>diff_DY_sig"+bin_DY,"id==0")
diff_DY_sig= gROOT.FindObject("diff_DY_sig")
diff_DY_sig.SetLineWidth(2)
diff_DY_sig.SetTitle('H->ZA sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_DY_sig.GetYaxis().SetTitleOffset(1.1)
diff_DY_sig.GetXaxis().SetTitleOffset(1.35)
diff_DY_sig.GetXaxis().SetTitleSize(.06)
diff_DY_sig.GetYaxis().SetTitleSize(.06)
diff_DY_sig.GetXaxis().SetNdivisions(505)
pad4_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad4_quant.AddText("95%% quantile : %0.5f"%quant_DY_sig)
pad4_quant.SetBorderSize(1)
pad4_quant.SetFillColor(0)
pad4_quant.Draw()

pad5.cd()
pad5.SetBottomMargin(0.25)
pad5.SetRightMargin(0.2)
t_DY.Draw(str_DY+">>diff_DY_TT"+bin_DY,"id==1")
diff_DY_TT = gROOT.FindObject("diff_DY_TT")
diff_DY_TT.SetLineWidth(2)
diff_DY_TT.SetTitle('TT sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_DY_TT.GetYaxis().SetTitleOffset(1.1)
diff_DY_TT.GetXaxis().SetTitleOffset(1.35)
diff_DY_TT.GetXaxis().SetTitleSize(.06)
diff_DY_TT.GetYaxis().SetTitleSize(.06)
diff_DY_TT.GetXaxis().SetNdivisions(505)
pad5_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad5_quant.AddText("95%% quantile : %0.5f"%quant_DY_TT)
pad5_quant.SetBorderSize(1)
pad5_quant.SetFillColor(0)
pad5_quant.Draw()


pad6.cd()
pad6.SetBottomMargin(0.25)
pad6.SetRightMargin(0.2)
t_DY.Draw(str_DY+">>diff_DY_DY"+bin_DY,"id==2")
diff_DY_DY= gROOT.FindObject("diff_DY_DY")
diff_DY_DY.SetLineWidth(2)
diff_DY_DY.SetTitle('DY sample; #frac{Y_{Error}^{MoMEMta}}{Y_{MEM}}; Occurences')
diff_DY_DY.GetYaxis().SetTitleOffset(1.1)
diff_DY_DY.GetXaxis().SetTitleOffset(1.35)
diff_DY_DY.GetXaxis().SetTitleSize(.06)
diff_DY_DY.GetYaxis().SetTitleSize(.06)
diff_DY_DY.GetXaxis().SetNdivisions(505)
pad6_quant = TPaveText(0.45,0.7,0.8,0.85,"brNDC")
pad6_quant.AddText("95%% quantile : %0.5f"%quant_DY_DY)
pad6_quant.SetBorderSize(1)
pad6_quant.SetFillColor(0)
pad6_quant.Draw('same')


gPad.Update()

c6.Print(path+"MoMEMtaError.png")
#input ("Press any key to end")
sys.exit()

###############################################################################
# Discriminant plots #
###############################################################################
gROOT.SetBatch(True)

# Recover normalization #
norm = np.genfromtxt('/home/ucl/cp3/fbury/storage/MoMEMtaModelNN/'+args.label+'/normalization.txt')
# Invalid tree -> Array #
array_TT_sig_inv_original = tree2array(t_TT_inv,branches=['visible_cross_section','NNOut_TT','original_MEM_TT','original_MEM_DY'],selection='id==0')
array_TT_TT_inv_original = tree2array(t_TT_inv,branches=['visible_cross_section','NNOut_TT','original_MEM_TT','original_MEM_DY'],selection='id==1')
array_TT_DY_inv_original = tree2array(t_TT_inv,branches=['visible_cross_section','NNOut_TT','original_MEM_TT','original_MEM_DY'],selection='id==2')
array_DY_sig_inv_original = tree2array(t_DY_inv,branches=['visible_cross_section','NNOut_DY','original_MEM_TT','original_MEM_DY'],selection='id==0')
array_DY_TT_inv_original = tree2array(t_DY_inv,branches=['visible_cross_section','NNOut_DY','original_MEM_TT','original_MEM_DY'],selection='id==1')
array_DY_DY_inv_original = tree2array(t_DY_inv,branches=['visible_cross_section','NNOut_DY','original_MEM_TT','original_MEM_DY'],selection='id==2')

# Rescaling to original weights #       
one_sig_MEM = np.ones(array_TT_sig_MEM.shape[0])
one_TT_MEM = np.ones(array_TT_TT_MEM.shape[0])
one_DY_MEM = np.ones(array_TT_DY_MEM.shape[0])
one_sig_NNOut = np.ones(array_TT_sig_NNOut.shape[0])
one_TT_NNOut = np.ones(array_TT_TT_NNOut.shape[0])
one_DY_NNOut = np.ones(array_TT_DY_NNOut.shape[0])

one_TT_sig_inv = np.ones(array_TT_sig_inv_original['original_MEM_TT'].shape[0])
one_TT_TT_inv = np.ones(array_TT_TT_inv_original['original_MEM_TT'].shape[0])
one_TT_DY_inv = np.ones(array_TT_DY_inv_original['original_MEM_TT'].shape[0])
one_DY_sig_inv = np.ones(array_DY_sig_inv_original['original_MEM_DY'].shape[0])
one_DY_TT_inv = np.ones(array_DY_TT_inv_original['original_MEM_DY'].shape[0])
one_DY_DY_inv = np.ones(array_DY_DY_inv_original['original_MEM_DY'].shape[0])

vis_xsec_TT_sig = tree2array(t_TT,branches='visible_cross_section',selection='id==0')
vis_xsec_TT_TT = tree2array(t_TT,branches='visible_cross_section',selection='id==1')
vis_xsec_TT_DY = tree2array(t_TT,branches='visible_cross_section',selection='id==2')
vis_xsec_DY_sig = tree2array(t_DY,branches='visible_cross_section',selection='id==0')
vis_xsec_DY_TT = tree2array(t_DY,branches='visible_cross_section',selection='id==1')
vis_xsec_DY_DY = tree2array(t_DY,branches='visible_cross_section',selection='id==2')

array_TT_sig_MEM_original = np.divide(np.power(one_sig_MEM*10,-array_TT_sig_MEM)*norm[0],vis_xsec_TT_sig)
array_TT_TT_MEM_original = np.divide(np.power(one_TT_MEM*10,-array_TT_TT_MEM)*norm[0],vis_xsec_TT_TT)
array_TT_DY_MEM_original = np.divide(np.power(one_DY_MEM*10,-array_TT_DY_MEM)*norm[0],vis_xsec_TT_DY)
array_DY_sig_MEM_original = np.divide(np.power(one_sig_MEM*10,-array_DY_sig_MEM)*norm[1],vis_xsec_DY_sig)
array_DY_TT_MEM_original = np.divide(np.power(one_TT_MEM*10,-array_DY_TT_MEM)*norm[1],vis_xsec_DY_TT)
array_DY_DY_MEM_original = np.divide(np.power(one_DY_MEM*10,-array_DY_DY_MEM)*norm[1],vis_xsec_DY_DY)

array_TT_sig_NNOut_original = np.divide(np.power(one_sig_NNOut*10,-array_TT_sig_NNOut)*norm[0],vis_xsec_TT_sig)
array_TT_TT_NNOut_original = np.divide(np.power(one_TT_NNOut*10,-array_TT_TT_NNOut)*norm[0],vis_xsec_TT_TT)
array_TT_DY_NNOut_original = np.divide(np.power(one_DY_NNOut*10,-array_TT_DY_NNOut)*norm[0],vis_xsec_TT_DY)
array_DY_sig_NNOut_original = np.divide(np.power(one_sig_NNOut*10,-array_DY_sig_NNOut)*norm[1],vis_xsec_DY_sig)
array_DY_TT_NNOut_original = np.divide(np.power(one_TT_NNOut*10,-array_DY_TT_NNOut)*norm[1],vis_xsec_DY_TT)
array_DY_DY_NNOut_original = np.divide(np.power(one_DY_NNOut*10,-array_DY_DY_NNOut)*norm[1],vis_xsec_DY_DY)

array_TT_sig_inv_original['NNOut_TT'] = np.divide(np.power(one_TT_sig_inv*10,-array_TT_sig_inv_original['NNOut_TT'])*norm[0],array_TT_sig_inv_original['visible_cross_section'])
array_TT_TT_inv_original['NNOut_TT'] = np.divide(np.power(one_TT_TT_inv*10,-array_TT_TT_inv_original['NNOut_TT'])*norm[0],array_TT_TT_inv_original['visible_cross_section'])
array_TT_DY_inv_original['NNOut_TT'] = np.divide(np.power(one_TT_DY_inv*10,-array_TT_DY_inv_original['NNOut_TT'])*norm[0],array_TT_DY_inv_original['visible_cross_section'])
array_DY_sig_inv_original['NNOut_DY'] = np.divide(np.power(one_DY_sig_inv*10,-array_DY_sig_inv_original['NNOut_DY'])*norm[1],array_DY_sig_inv_original['visible_cross_section'])
array_DY_TT_inv_original['NNOut_DY'] = np.divide(np.power(one_DY_TT_inv*10,-array_DY_TT_inv_original['NNOut_DY'])*norm[1],array_DY_TT_inv_original['visible_cross_section'])
array_DY_DY_inv_original['NNOut_DY'] = np.divide(np.power(one_DY_DY_inv*10,-array_DY_DY_inv_original['NNOut_DY'])*norm[1],array_DY_DY_inv_original['visible_cross_section'])

array_TT_sig_inv_original['original_MEM_TT'] = np.divide(array_TT_sig_inv_original['original_MEM_TT'],array_TT_sig_inv_original['visible_cross_section'])
array_TT_TT_inv_original['original_MEM_TT'] = np.divide(array_TT_TT_inv_original['original_MEM_TT'],array_TT_TT_inv_original['visible_cross_section'])
array_TT_DY_inv_original['original_MEM_TT'] = np.divide(array_TT_DY_inv_original['original_MEM_TT'],array_TT_DY_inv_original['visible_cross_section'])
array_DY_sig_inv_original['original_MEM_TT'] = np.divide(array_DY_sig_inv_original['original_MEM_TT'],array_DY_sig_inv_original['visible_cross_section'])
array_DY_TT_inv_original['original_MEM_TT'] = np.divide(array_DY_TT_inv_original['original_MEM_TT'],array_DY_TT_inv_original['visible_cross_section'])
array_DY_DY_inv_original['original_MEM_TT'] = np.divide(array_DY_DY_inv_original['original_MEM_TT'],array_DY_DY_inv_original['visible_cross_section'])

array_TT_sig_inv_original['original_MEM_DY'] = np.divide(array_TT_sig_inv_original['original_MEM_DY'],array_TT_sig_inv_original['visible_cross_section'])
array_TT_TT_inv_original['original_MEM_DY'] = np.divide(array_TT_TT_inv_original['original_MEM_DY'],array_TT_TT_inv_original['visible_cross_section'])
array_TT_DY_inv_original['original_MEM_DY'] = np.divide(array_TT_DY_inv_original['original_MEM_DY'],array_TT_DY_inv_original['visible_cross_section'])
array_DY_sig_inv_original['original_MEM_DY'] = np.divide(array_DY_sig_inv_original['original_MEM_DY'],array_DY_sig_inv_original['visible_cross_section'])
array_DY_TT_inv_original['original_MEM_DY'] = np.divide(array_DY_TT_inv_original['original_MEM_DY'],array_DY_TT_inv_original['visible_cross_section'])
array_DY_DY_inv_original['original_MEM_DY'] = np.divide(array_DY_DY_inv_original['original_MEM_DY'],array_DY_DY_inv_original['visible_cross_section'])

# Computing disciminant #
alpha = 200

dis_sig_MEM = np.divide(array_TT_sig_MEM_original,np.add(array_DY_sig_MEM_original*alpha,array_TT_sig_MEM_original))
dis_TT_MEM = np.divide(array_TT_TT_MEM_original,np.add(array_DY_TT_MEM_original*alpha,array_TT_TT_MEM_original))
dis_DY_MEM = np.divide(array_TT_DY_MEM_original,np.add(array_DY_DY_MEM_original*alpha,array_TT_DY_MEM_original))
dis_sig_NNOut = np.divide(array_TT_sig_NNOut_original,np.add(array_DY_sig_NNOut_original*alpha,array_TT_sig_NNOut_original))
dis_TT_NNOut = np.divide(array_TT_TT_NNOut_original,np.add(array_DY_TT_NNOut_original*alpha,array_TT_TT_NNOut_original))
dis_DY_NNOut = np.divide(array_TT_DY_NNOut_original,np.add(array_DY_DY_NNOut_original*alpha,array_TT_DY_NNOut_original))


dis_TT_sig_MEM_inv = np.divide(array_TT_sig_inv_original['original_MEM_TT'],np.add(array_TT_sig_inv_original['original_MEM_TT'],array_TT_sig_inv_original['original_MEM_DY']*alpha))
dis_TT_TT_MEM_inv = np.divide(array_TT_TT_inv_original['original_MEM_TT'],np.add(array_TT_TT_inv_original['original_MEM_TT'],array_TT_TT_inv_original['original_MEM_DY']*alpha))
dis_TT_DY_MEM_inv = np.divide(array_TT_DY_inv_original['original_MEM_TT'],np.add(array_TT_DY_inv_original['original_MEM_TT'],array_TT_DY_inv_original['original_MEM_DY']*alpha))
dis_DY_sig_MEM_inv = np.divide(array_DY_sig_inv_original['original_MEM_TT'],np.add(array_DY_sig_inv_original['original_MEM_TT'],array_DY_sig_inv_original['original_MEM_DY']*alpha))
dis_DY_TT_MEM_inv = np.divide(array_DY_TT_inv_original['original_MEM_TT'],np.add(array_DY_TT_inv_original['original_MEM_TT'],array_DY_TT_inv_original['original_MEM_DY']*alpha))
dis_DY_DY_MEM_inv = np.divide(array_DY_DY_inv_original['original_MEM_TT'],np.add(array_DY_DY_inv_original['original_MEM_TT'],array_DY_DY_inv_original['original_MEM_DY']*alpha))

dis_TT_sig_NNOut_inv = np.divide(array_TT_sig_inv_original['NNOut_TT'],np.add(array_TT_sig_inv_original['NNOut_TT'],array_TT_sig_inv_original['original_MEM_DY']*alpha))
dis_TT_TT_NNOut_inv = np.divide(array_TT_TT_inv_original['NNOut_TT'],np.add(array_TT_TT_inv_original['NNOut_TT'],array_TT_TT_inv_original['original_MEM_DY']*alpha))
dis_TT_DY_NNOut_inv = np.divide(array_TT_DY_inv_original['NNOut_TT'],np.add(array_TT_DY_inv_original['NNOut_TT'],array_TT_DY_inv_original['original_MEM_DY']*alpha))
dis_DY_sig_NNOut_inv = np.divide(array_DY_sig_inv_original['original_MEM_TT'],np.add(array_DY_sig_inv_original['original_MEM_TT'],array_DY_sig_inv_original['NNOut_DY']*alpha))
dis_DY_TT_NNOut_inv = np.divide(array_DY_TT_inv_original['original_MEM_TT'],np.add(array_DY_TT_inv_original['original_MEM_TT'],array_DY_TT_inv_original['NNOut_DY']*alpha))
dis_DY_DY_NNOut_inv = np.divide(array_DY_DY_inv_original['original_MEM_TT'],np.add(array_DY_DY_inv_original['original_MEM_TT'],array_DY_DY_inv_original['NNOut_DY']*alpha))

# Array -> Tree #
dis_sig_MEM.dtype = [('discriminant','float64')]
dis_sig_MEM.dtype.names = ['discriminant']
tree_dis_sig_MEM = array2tree(dis_sig_MEM)
dis_TT_MEM.dtype = [('discriminant','float64')]
dis_TT_MEM.dtype.names = ['discriminant']
tree_dis_TT_MEM = array2tree(dis_TT_MEM)
dis_DY_MEM.dtype = [('discriminant','float64')]
dis_DY_MEM.dtype.names = ['discriminant']
tree_dis_DY_MEM = array2tree(dis_DY_MEM)
dis_sig_NNOut.dtype = [('discriminant','float64')]
dis_sig_NNOut.dtype.names = ['discriminant']
tree_dis_sig_NNOut = array2tree(dis_sig_NNOut)
dis_TT_NNOut.dtype = [('discriminant','float64')]
dis_TT_NNOut.dtype.names = ['discriminant']
tree_dis_TT_NNOut = array2tree(dis_TT_NNOut)
dis_DY_NNOut.dtype = [('discriminant','float64')]
dis_DY_NNOut.dtype.names = ['discriminant']
tree_dis_DY_NNOut = array2tree(dis_DY_NNOut)

dis_TT_sig_MEM_inv.dtype = [('discriminant','float64')]
dis_TT_sig_MEM_inv.dtype.names = ['discriminant']
tree_dis_TT_sig_MEM_inv = array2tree(dis_TT_sig_MEM_inv)
dis_TT_TT_MEM_inv.dtype = [('discriminant','float64')]
dis_TT_TT_MEM_inv.dtype.names = ['discriminant']
tree_dis_TT_TT_MEM_inv = array2tree(dis_TT_TT_MEM_inv)
dis_TT_DY_MEM_inv.dtype = [('discriminant','float64')]
dis_TT_DY_MEM_inv.dtype.names = ['discriminant']
tree_dis_TT_DY_MEM_inv = array2tree(dis_TT_DY_MEM_inv)
dis_DY_sig_MEM_inv.dtype = [('discriminant','float64')]
dis_DY_sig_MEM_inv.dtype.names = ['discriminant']
tree_dis_DY_sig_MEM_inv = array2tree(dis_DY_sig_MEM_inv)
dis_DY_TT_MEM_inv.dtype = [('discriminant','float64')]
dis_DY_TT_MEM_inv.dtype.names = ['discriminant']
tree_dis_DY_TT_MEM_inv = array2tree(dis_DY_TT_MEM_inv)
dis_DY_DY_MEM_inv.dtype = [('discriminant','float64')]
dis_DY_DY_MEM_inv.dtype.names = ['discriminant']
tree_dis_DY_DY_MEM_inv = array2tree(dis_DY_DY_MEM_inv)

dis_TT_sig_NNOut_inv.dtype = [('discriminant','float64')]
dis_TT_sig_NNOut_inv.dtype.names = ['discriminant']
tree_dis_TT_sig_NNOut_inv = array2tree(dis_TT_sig_NNOut_inv)
dis_TT_TT_NNOut_inv.dtype = [('discriminant','float64')]
dis_TT_TT_NNOut_inv.dtype.names = ['discriminant']
tree_dis_TT_TT_NNOut_inv = array2tree(dis_TT_TT_NNOut_inv)
dis_TT_DY_NNOut_inv.dtype = [('discriminant','float64')]
dis_TT_DY_NNOut_inv.dtype.names = ['discriminant']
tree_dis_TT_DY_NNOut_inv = array2tree(dis_TT_DY_NNOut_inv)
dis_DY_sig_NNOut_inv.dtype = [('discriminant','float64')]
dis_DY_sig_NNOut_inv.dtype.names = ['discriminant']
tree_dis_DY_sig_NNOut_inv = array2tree(dis_DY_sig_NNOut_inv)
dis_DY_TT_NNOut_inv.dtype = [('discriminant','float64')]
dis_DY_TT_NNOut_inv.dtype.names = ['discriminant']
tree_dis_DY_TT_NNOut_inv = array2tree(dis_DY_TT_NNOut_inv)
dis_DY_DY_NNOut_inv.dtype = [('discriminant','float64')]
dis_DY_DY_NNOut_inv.dtype.names = ['discriminant']
tree_dis_DY_DY_NNOut_inv = array2tree(dis_DY_DY_NNOut_inv)


# Tree -> TF1 #
tree_dis_sig_MEM.Draw("discriminant>>TF1_dis_sig_MEM(50,0,1)")
TF1_dis_sig_MEM = gROOT.FindObject("TF1_dis_sig_MEM")
tree_dis_TT_MEM.Draw("discriminant>>TF1_dis_TT_MEM(50,0,1)")
TF1_dis_TT_MEM = gROOT.FindObject("TF1_dis_TT_MEM")
tree_dis_DY_MEM.Draw("discriminant>>TF1_dis_DY_MEM(50,0,1)")
TF1_dis_DY_MEM = gROOT.FindObject("TF1_dis_DY_MEM")
tree_dis_sig_NNOut.Draw("discriminant>>TF1_dis_sig_NNOut(50,0,1)")
TF1_dis_sig_NNOut = gROOT.FindObject("TF1_dis_sig_NNOut")
tree_dis_TT_NNOut.Draw("discriminant>>TF1_dis_TT_NNOut(50,0,1)")
TF1_dis_TT_NNOut = gROOT.FindObject("TF1_dis_TT_NNOut")
tree_dis_DY_NNOut.Draw("discriminant>>TF1_dis_DY_NNOut(50,0,1)")
TF1_dis_DY_NNOut = gROOT.FindObject("TF1_dis_DY_NNOut")

tree_dis_TT_sig_MEM_inv.Draw("discriminant>>TF1_dis_TT_sig_MEM_inv(50,0,1)")
TF1_dis_TT_sig_MEM_inv = gROOT.FindObject("TF1_dis_TT_sig_MEM_inv")
tree_dis_TT_TT_MEM_inv.Draw("discriminant>>TF1_dis_TT_TT_MEM_inv(50,0,1)")
TF1_dis_TT_TT_MEM_inv = gROOT.FindObject("TF1_dis_TT_TT_MEM_inv")
tree_dis_TT_DY_MEM_inv.Draw("discriminant>>TF1_dis_TT_DY_MEM_inv(50,0,1)")
TF1_dis_TT_DY_MEM_inv = gROOT.FindObject("TF1_dis_TT_DY_MEM_inv")
tree_dis_DY_sig_MEM_inv.Draw("discriminant>>TF1_dis_DY_sig_MEM_inv(50,0,1)")
TF1_dis_DY_sig_MEM_inv = gROOT.FindObject("TF1_dis_DY_sig_MEM_inv")
tree_dis_DY_TT_MEM_inv.Draw("discriminant>>TF1_dis_DY_TT_MEM_inv(50,0,1)")
TF1_dis_DY_TT_MEM_inv = gROOT.FindObject("TF1_dis_DY_TT_MEM_inv")
tree_dis_DY_DY_MEM_inv.Draw("discriminant>>TF1_dis_DY_DY_MEM_inv(50,0,1)")
TF1_dis_DY_DY_MEM_inv = gROOT.FindObject("TF1_dis_DY_DY_MEM_inv")

tree_dis_TT_sig_NNOut_inv.Draw("discriminant>>TF1_dis_TT_sig_NNOut_inv(50,0,1)")
TF1_dis_TT_sig_NNOut_inv = gROOT.FindObject("TF1_dis_TT_sig_NNOut_inv")
tree_dis_TT_TT_NNOut_inv.Draw("discriminant>>TF1_dis_TT_TT_NNOut_inv(50,0,1)")
TF1_dis_TT_TT_NNOut_inv = gROOT.FindObject("TF1_dis_TT_TT_NNOut_inv")
tree_dis_TT_DY_NNOut_inv.Draw("discriminant>>TF1_dis_TT_DY_NNOut_inv(50,0,1)")
TF1_dis_TT_DY_NNOut_inv = gROOT.FindObject("TF1_dis_TT_DY_NNOut_inv")
tree_dis_DY_sig_NNOut_inv.Draw("discriminant>>TF1_dis_DY_sig_NNOut_inv(50,0,1)")
TF1_dis_DY_sig_NNOut_inv = gROOT.FindObject("TF1_dis_DY_sig_NNOut_inv")
tree_dis_DY_TT_NNOut_inv.Draw("discriminant>>TF1_dis_DY_TT_NNOut_inv(50,0,1)")
TF1_dis_DY_TT_NNOut_inv = gROOT.FindObject("TF1_dis_DY_TT_NNOut_inv")
tree_dis_DY_DY_NNOut_inv.Draw("discriminant>>TF1_dis_DY_DY_NNOut_inv(50,0,1)")
TF1_dis_DY_DY_NNOut_inv = gROOT.FindObject("TF1_dis_DY_DY_NNOut_inv")

# Tree properties #

TF1_dis_sig_MEM.SetFillColor(kBlue+3)
TF1_dis_TT_sig_MEM_inv.SetFillColor(kRed+2)
TF1_dis_DY_sig_MEM_inv.SetFillColor(kGreen+3)
TF1_dis_TT_MEM.SetFillColor(kBlue+3)
TF1_dis_TT_TT_MEM_inv.SetFillColor(kRed+2)
TF1_dis_DY_TT_MEM_inv.SetFillColor(kGreen+3)
TF1_dis_DY_MEM.SetFillColor(kBlue+3)
TF1_dis_TT_DY_MEM_inv.SetFillColor(kRed+2)
TF1_dis_DY_DY_MEM_inv.SetFillColor(kGreen+3)

TF1_dis_sig_NNOut.SetFillColor(kBlue+3)
TF1_dis_TT_sig_NNOut_inv.SetFillColor(kRed+2)
TF1_dis_DY_sig_NNOut_inv.SetFillColor(kGreen+3)
TF1_dis_TT_NNOut.SetFillColor(kBlue+3)
TF1_dis_TT_TT_NNOut_inv.SetFillColor(kRed+2)
TF1_dis_DY_TT_NNOut_inv.SetFillColor(kGreen+3)
TF1_dis_DY_NNOut.SetFillColor(kBlue+3)
TF1_dis_TT_DY_NNOut_inv.SetFillColor(kRed+2)
TF1_dis_DY_DY_NNOut_inv.SetFillColor(kGreen+3)

TF1_dis_sig_MEM.SetLineWidth(0)
TF1_dis_TT_sig_MEM_inv.SetLineWidth(0)
TF1_dis_DY_sig_MEM_inv.SetLineWidth(0)
TF1_dis_TT_MEM.SetLineWidth(0)
TF1_dis_TT_TT_MEM_inv.SetLineWidth(0)
TF1_dis_DY_TT_MEM_inv.SetLineWidth(0)
TF1_dis_DY_MEM.SetLineWidth(0)
TF1_dis_TT_DY_MEM_inv.SetLineWidth(0)
TF1_dis_DY_DY_MEM_inv.SetLineWidth(0)

TF1_dis_sig_NNOut.SetLineWidth(0)
TF1_dis_TT_sig_NNOut_inv.SetLineWidth(0)
TF1_dis_DY_sig_NNOut_inv.SetLineWidth(0)
TF1_dis_TT_NNOut.SetLineWidth(0)
TF1_dis_TT_TT_NNOut_inv.SetLineWidth(0)
TF1_dis_DY_TT_NNOut_inv.SetLineWidth(0)
TF1_dis_DY_NNOut.SetLineWidth(0)
TF1_dis_TT_DY_NNOut_inv.SetLineWidth(0)
TF1_dis_DY_DY_NNOut_inv.SetLineWidth(0)

# Stack plots #

Stack_dis_sig_MEM = THStack("Stack_dis_sig_MEM","")
Stack_dis_sig_MEM.Add(TF1_dis_sig_MEM)
Stack_dis_sig_MEM.Add(TF1_dis_TT_sig_MEM_inv)
Stack_dis_sig_MEM.Add(TF1_dis_DY_sig_MEM_inv)

Stack_dis_TT_MEM = THStack("Stack_dis_TT_MEM","")
Stack_dis_TT_MEM.Add(TF1_dis_TT_MEM)
Stack_dis_TT_MEM.Add(TF1_dis_TT_TT_MEM_inv)
Stack_dis_TT_MEM.Add(TF1_dis_DY_TT_MEM_inv)

Stack_dis_DY_MEM = THStack("Stack_dis_DY_MEM","")
Stack_dis_DY_MEM.Add(TF1_dis_DY_MEM)
Stack_dis_DY_MEM.Add(TF1_dis_TT_DY_MEM_inv)
Stack_dis_DY_MEM.Add(TF1_dis_DY_DY_MEM_inv)

Stack_dis_sig_NNOut = THStack("Stack_dis_sig_NNOut","")
Stack_dis_sig_NNOut.Add(TF1_dis_sig_NNOut)
Stack_dis_sig_NNOut.Add(TF1_dis_TT_sig_NNOut_inv)
Stack_dis_sig_NNOut.Add(TF1_dis_DY_sig_NNOut_inv)

Stack_dis_TT_NNOut = THStack("Stack_dis_TT_NNOut","")
Stack_dis_TT_NNOut.Add(TF1_dis_TT_NNOut)
Stack_dis_TT_NNOut.Add(TF1_dis_TT_TT_NNOut_inv)
Stack_dis_TT_NNOut.Add(TF1_dis_DY_TT_NNOut_inv)

Stack_dis_DY_NNOut = THStack("Stack_dis_DY_NNOut","")
Stack_dis_DY_NNOut.Add(TF1_dis_DY_NNOut)
Stack_dis_DY_NNOut.Add(TF1_dis_TT_DY_NNOut_inv)
Stack_dis_DY_NNOut.Add(TF1_dis_DY_DY_NNOut_inv)


# Legend #
legend_dis_sig_MEM = TLegend(0.23,0.55,0.75,0.85)
legend_dis_sig_MEM.SetHeader("Legend","C")
legend_dis_sig_MEM.AddEntry(TF1_dis_sig_MEM,"Valid weights : %0.f entries"%(TF1_dis_sig_MEM.GetEntries()))
legend_dis_sig_MEM.AddEntry(TF1_dis_TT_sig_MEM_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_sig_MEM_inv.GetEntries()))
legend_dis_sig_MEM.AddEntry(TF1_dis_DY_sig_MEM_inv,"Invalid DY weights : %0.f entries"%(TF1_dis_DY_sig_MEM_inv.GetEntries()))

legend_dis_TT_MEM = TLegend(0.23,0.55,0.75,0.85)
legend_dis_TT_MEM.SetHeader("Legend","C")
legend_dis_TT_MEM.AddEntry(TF1_dis_TT_MEM,"Valid weights : %0.f entries"%(TF1_dis_TT_MEM.GetEntries()))
legend_dis_TT_MEM.AddEntry(TF1_dis_TT_TT_MEM_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_TT_MEM_inv.GetEntries()))
legend_dis_TT_MEM.AddEntry(TF1_dis_DY_TT_MEM_inv,"Invalid DY weights  : %0.f entries"%(TF1_dis_DY_TT_MEM_inv.GetEntries()))

legend_dis_DY_MEM = TLegend(0.23,0.55,0.75,0.85)
legend_dis_DY_MEM.SetHeader("Legend","C")
legend_dis_DY_MEM.AddEntry(TF1_dis_DY_MEM,"Valid weights : %0.f entries"%(TF1_dis_DY_MEM.GetEntries()))
legend_dis_DY_MEM.AddEntry(TF1_dis_TT_DY_MEM_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_DY_MEM_inv.GetEntries()))
legend_dis_DY_MEM.AddEntry(TF1_dis_DY_DY_MEM_inv,"Invalid DY weights  : %0.f entries"%(TF1_dis_DY_DY_MEM_inv.GetEntries()))

legend_dis_sig_NNOut = TLegend(0.23,0.55,0.75,0.85)
legend_dis_sig_NNOut.SetHeader("Legend","C")
legend_dis_sig_NNOut.AddEntry(TF1_dis_sig_NNOut,"Valid weights : %0.f entries"%(TF1_dis_sig_NNOut.GetEntries()))
legend_dis_sig_NNOut.AddEntry(TF1_dis_TT_sig_NNOut_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_sig_NNOut_inv.GetEntries()))
legend_dis_sig_NNOut.AddEntry(TF1_dis_DY_sig_NNOut_inv,"Invalid DY weights  : %0.f entries"%(TF1_dis_DY_sig_NNOut_inv.GetEntries()))

legend_dis_TT_NNOut = TLegend(0.23,0.55,0.75,0.85)
legend_dis_TT_NNOut.SetHeader("Legend","C")
legend_dis_TT_NNOut.AddEntry(TF1_dis_TT_NNOut,"Valid weights : %0.f entries"%(TF1_dis_TT_NNOut.GetEntries()))
legend_dis_TT_NNOut.AddEntry(TF1_dis_TT_TT_NNOut_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_TT_NNOut_inv.GetEntries()))
legend_dis_TT_NNOut.AddEntry(TF1_dis_DY_TT_NNOut_inv,"Invalid DY weights  : %0.f entries"%(TF1_dis_DY_TT_NNOut_inv.GetEntries()))

legend_dis_DY_NNOut = TLegend(0.23,0.55,0.75,0.85)
legend_dis_DY_NNOut.SetHeader("Legend","C")
legend_dis_DY_NNOut.AddEntry(TF1_dis_DY_NNOut,"Valid weights : %0.f entries"%(TF1_dis_DY_NNOut.GetEntries()))
legend_dis_DY_NNOut.AddEntry(TF1_dis_TT_DY_NNOut_inv,"Invalid TT weights : %0.f entries"%(TF1_dis_TT_DY_NNOut_inv.GetEntries()))
legend_dis_DY_NNOut.AddEntry(TF1_dis_DY_DY_NNOut_inv,"Invalid DY weights  : %0.f entries"%(TF1_dis_DY_DY_NNOut_inv.GetEntries()))


# Plots #
c7 = TCanvas( 'c7','Discriminant plot', 200, 10, 1200, 700 ) 
title = TPaveText( .3, 0.92, .7, .99 )
title.AddText("Discriminant plot")
title.SetFillColor(0)
title.Draw()

NN_discriminant = TPaveText(0.01,0.05,0.04,0.45)
text_NN = NN_discriminant.AddText("NN Discriminant")
text_NN.SetTextAngle(90.)
text_NN.SetTextFont(43)
text_NN.SetTextSize(30)
text_NN.SetTextAlign(22)
NN_discriminant.SetFillColor(0)
NN_discriminant.Draw()

MEM_discriminant = TPaveText(0.01,0.5,0.04,0.9)
text_MEM = MEM_discriminant.AddText("MEM Discriminant")
text_MEM.SetTextFont(43)
text_MEM.SetTextSize(30)
text_MEM.SetTextAngle(90.)
text_MEM.SetTextAlign(22)
MEM_discriminant.SetFillColor(0)
MEM_discriminant.Draw()


pad1 = TPad( 'pad1', 'TT weight : Signal', 0.05, 0, 0.4, 0.45, -1 )
pad2 = TPad( 'pad2', 'TT weight : TT', 0.35, 0, 0.7, 0.45, -1 )
pad3 = TPad( 'pad3', 'TT weight : DY', 0.65, 0, 1, 0.45, -1 )
pad4 = TPad( 'pad4', 'DY weight : Signal', 0.05, 0.45, 0.4, 0.9, -1 )
pad5 = TPad( 'pad5', 'DY weight : TT', 0.35, 0.45, 0.7, 0.9, -1 )
pad6 = TPad( 'pad6', 'DY weight : DY', 0.65, 0.45, 1, 0.9, -1 )

pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()
pad5.Draw()
pad6.Draw()

pad1.cd()
pad1.SetLeftMargin(0.13)
pad1.SetTopMargin(0.15)
pad1.SetLogy()
Stack_dis_sig_NNOut.Draw()
legend_dis_sig_NNOut.Draw()
Stack_dis_sig_NNOut.SetTitle("H-ZA sample;Discriminant;Occurences")
Stack_dis_sig_NNOut.GetYaxis().SetTitleOffset(1.1)
Stack_dis_sig_NNOut.GetXaxis().SetTitleSize(.06)
Stack_dis_sig_NNOut.GetYaxis().SetTitleSize(.06)

pad2.cd()
pad2.SetLeftMargin(0.13)
pad2.SetTopMargin(0.15)
pad2.SetLogy()
Stack_dis_TT_NNOut.Draw()
legend_dis_TT_NNOut.Draw()
Stack_dis_TT_NNOut.SetTitle("TT sample;Discriminant;Occurences")
Stack_dis_TT_NNOut.GetYaxis().SetTitleOffset(1.1)
Stack_dis_TT_NNOut.GetXaxis().SetTitleSize(.06)
Stack_dis_TT_NNOut.GetYaxis().SetTitleSize(.06)

pad3.cd()
pad3.SetLeftMargin(0.13)
pad3.SetTopMargin(0.15)
pad3.SetLogy()
Stack_dis_DY_NNOut.Draw()
legend_dis_DY_NNOut.Draw()
Stack_dis_DY_NNOut.SetTitle("DY sample;Discriminant;Occurences")
Stack_dis_DY_NNOut.GetYaxis().SetTitleOffset(1.1)
Stack_dis_DY_NNOut.GetXaxis().SetTitleSize(.06)
Stack_dis_DY_NNOut.GetYaxis().SetTitleSize(.06)

pad4.cd()
pad4.SetTopMargin(0.15)
pad4.SetLeftMargin(0.13)
pad4.SetLogy()
Stack_dis_sig_MEM.Draw()
legend_dis_sig_MEM.Draw()
Stack_dis_sig_MEM.SetTitle("H-ZA sample;Discriminant;Occurences")
Stack_dis_sig_MEM.GetYaxis().SetTitleOffset(1.1)
Stack_dis_sig_MEM.GetXaxis().SetTitleSize(.06)
Stack_dis_sig_MEM.GetYaxis().SetTitleSize(.06)

pad5.cd()
pad5.SetLeftMargin(0.13)
pad5.SetTopMargin(0.15)
pad5.SetLogy()
Stack_dis_TT_MEM.Draw()
legend_dis_TT_MEM.Draw()
Stack_dis_TT_MEM.SetTitle("TT sample;Discriminant;Occurences")
Stack_dis_TT_MEM.GetYaxis().SetTitleOffset(1.1)
Stack_dis_TT_MEM.GetXaxis().SetTitleSize(.06)
Stack_dis_TT_MEM.GetYaxis().SetTitleSize(.06)

pad6.cd()
pad6.SetLeftMargin(0.13)
pad6.SetTopMargin(0.15)
pad6.SetLogy()
Stack_dis_DY_MEM.Draw()
legend_dis_DY_MEM.Draw()
Stack_dis_DY_MEM.SetTitle("DY sample;Discriminant;Occurences")
Stack_dis_DY_MEM.GetYaxis().SetTitleOffset(1.1)
Stack_dis_DY_MEM.GetXaxis().SetTitleSize(.06)
Stack_dis_DY_MEM.GetYaxis().SetTitleSize(.06)

gPad.Update()
c7.Print(path+"Discriminant.png")

#input ("Press any key to end")

###############################################################################
# Discriminant ROC curves #
###############################################################################
dis_sig_MEM = np.concatenate((dis_sig_MEM,dis_DY_sig_MEM_inv,dis_TT_sig_MEM_inv),axis=0) 
dis_TT_MEM = np.concatenate((dis_TT_MEM,dis_DY_TT_MEM_inv,dis_TT_TT_MEM_inv),axis=0) 
dis_DY_MEM = np.concatenate((dis_DY_MEM,dis_DY_DY_MEM_inv,dis_TT_DY_MEM_inv),axis=0) 
dis_sig_NNOut = np.concatenate((dis_sig_NNOut,dis_DY_sig_NNOut_inv,dis_TT_sig_NNOut_inv),axis=0) 
dis_TT_NNOut = np.concatenate((dis_TT_NNOut,dis_DY_TT_NNOut_inv,dis_TT_TT_NNOut_inv),axis=0) 
dis_DY_NNOut = np.concatenate((dis_DY_NNOut,dis_DY_DY_NNOut_inv,dis_TT_DY_NNOut_inv),axis=0) 
dis_sig_MEM = dis_sig_MEM.astype('Float64')
dis_TT_MEM = dis_TT_MEM.astype('Float64')
dis_DY_MEM = dis_DY_MEM.astype('Float64')
dis_sig_NNOut = dis_sig_NNOut.astype('Float64')
dis_TT_NNOut = dis_TT_NNOut.astype('Float64')
dis_DY_NNOut = dis_DY_NNOut.astype('Float64')


# Discriminants #
bins = np.linspace(0,1,50)

fig1 = plt.figure(1,figsize=(10,5))
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)
ax1.hist(dis_sig_MEM,bins=bins)
ax2.hist(dis_TT_MEM,bins=bins)
ax3.hist(dis_DY_MEM,bins=bins)
ax4.hist(dis_sig_NNOut,bins=bins)
ax5.hist(dis_TT_NNOut,bins=bins)
ax6.hist(dis_DY_NNOut,bins=bins)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax1.set_title('H->ZA sample : MEM')
ax2.set_title('TT sample : MEM')
ax3.set_title('DY sample : MEM')
ax4.set_title('H->ZA sample : NN')
ax5.set_title('TT sample : NN')
ax6.set_title('DY sample : NN')
#plt.show()
plt.close()

# TT and DY disciminant comparison #

fig2 = plt.figure(2)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.hist(dis_TT_MEM,bins=bins,alpha=0.7,color='g',label='TT')
ax1.hist(dis_DY_MEM,bins=bins,alpha=0.7,color='r',label='DY')
ax2.hist(dis_TT_NNOut,bins=bins,alpha=0.7,color='g',label='TT')
ax2.hist(dis_DY_NNOut,bins=bins,alpha=0.7,color='r',label='DY')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('MEM')
ax2.set_title('NNOut')
#plt.show()
plt.close()

# ROC curve #
dis_roc_MEM = np.concatenate((dis_TT_MEM,dis_DY_MEM),axis=0)
target_roc_MEM = np.concatenate((np.ones(dis_TT_MEM.shape[0]),np.zeros(dis_DY_MEM.shape[0])),axis=0)
dis_roc_NNOut = np.concatenate((dis_TT_NNOut,dis_DY_NNOut),axis=0)
target_roc_NNOut = np.concatenate((np.ones(dis_TT_NNOut.shape[0]),np.zeros(dis_DY_NNOut.shape[0])),axis=0)
fpr_MEM, tpr_MEM,tresholds = roc_curve(target_roc_MEM,dis_roc_MEM)
fpr_NNOut, tpr_NNOut,tresholds = roc_curve(target_roc_NNOut,dis_roc_NNOut)
auc_MEM = roc_auc_score(target_roc_MEM,dis_roc_MEM)
auc_NNOut = roc_auc_score(target_roc_NNOut,dis_roc_NNOut)

fig3 = plt.figure(3)
plt.plot(tpr_MEM,fpr_MEM,color='g',label=('MEM : AUC score = %0.5f'%(auc_MEM)))
plt.plot(tpr_NNOut,fpr_NNOut,color='r',label=('NN : AUC score = %0.5f'%(auc_NNOut)))
plt.title('ROC curve of the discriminant',fontsize=18)
plt.xlabel('True Positive Rate',fontsize=16)
plt.ylabel('False Positive Rate',fontsize=16)
plt.legend(loc='upper left',fontsize=14)
plt.ylim([0,1])
plt.ylim([0,1])
plt.xlim([0,1])
plt.grid()
#plt.show()
fig3.savefig(path+'ROC_discriminant.png')
print ('[INFO] Plot saved as '+path+'ROC_discriminant.png')
plt.close()


###############################################################################
# Separate plot for different masses #
###############################################################################
gROOT.SetBatch(True)
c8 = TCanvas( 'c8','Specific Mass Plot', 200, 10, 1200, 700 ) 

mHmA = tree2array(t_TT,branches=['mH','mA'],selection='id==0')
mHmA = np.unique(mHmA)

path_mass = path+'Masses/'
if not os.path.exists(path_mass):
    os.makedirs(path_mass)

for i in range(0,mHmA.shape[0]):
    print ('\tmH = '+str(mHmA[i][0])+', mA = '+str(mHmA[i][1]))
    # Selecting correct dataset #
    selection_sig = 'id==0 && mH=='+str(mHmA[i][0])+' && mA=='+str(mHmA[i][1])
    selection_back = 'id!=0'
    if mHmA[i][0]>=3000:
        binning = '(100,0,45)'
    elif mHmA[i][0]>=2000:
        binning = '(100,0,40)'
    elif mHmA[i][0]>=1000:
        binning = '(100,0,30)'
    elif mHmA[i][0]>=650:
        binning = '(100,0,25)'
    elif mHmA[i][0]>=500:
        binning = '(100,0,20)'
    else :
        binning = '(100,0,10)'

    t_TT.Draw("MEM_TT>>MEM_TT"+binning,selection_sig)
    MEM_TT = gROOT.FindObject("MEM_TT")
    t_TT.Draw("NNOut_TT>>NNOut_TT"+binning,selection_sig)
    NNOut_TT = gROOT.FindObject("NNOut_TT")

    t_DY.Draw("MEM_DY>>MEM_DY"+binning,selection_sig)
    MEM_DY = gROOT.FindObject("MEM_DY")
    t_DY.Draw("NNOut_DY>>NNOut_DY"+binning,selection_sig)
    NNOut_DY = gROOT.FindObject("NNOut_DY")

    t_TT_inv.Draw("NNOut_TT>>NNOut_TT_inv"+binning,selection_sig)
    NNOut_TT_inv = gROOT.FindObject("NNOut_TT_inv")

    t_DY_inv.Draw("NNOut_DY>>NNOut_DY_inv"+binning,selection_sig)
    NNOut_DY_inv = gROOT.FindObject("NNOut_DY_inv")

    bin_dis = "(40,0,1)"
    dis_str = "(original_MEM_TT/visible_cross_section)/((original_MEM_TT/visible_cross_section)+200*(original_MEM_DY/visible_cross_section))"

    t_TT.Draw(dis_str+">>sig_dis_MEM"+bin_dis,selection_sig)
    sig_dis_MEM = gROOT.FindObject("sig_dis_MEM")

    t_TT_inv.Draw(dis_str+">>sig_dis_MEM_inv_TT"+bin_dis,selection_sig)
    sig_dis_MEM_inv_TT = gROOT.FindObject("sig_dis_MEM_inv_TT")

    t_DY_inv.Draw(dis_str+">>sig_dis_MEM_inv_DY"+bin_dis,selection_sig)
    sig_dis_MEM_inv_DY = gROOT.FindObject("sig_dis_MEM_inv_DY")

    t_TT.Draw(dis_str+">>back_dis_MEM"+bin_dis,selection_back)
    back_dis_MEM = gROOT.FindObject("back_dis_MEM")

    array_sig_MEM = tree2array(t_TT,branches=dis_str,selection=selection_sig)
    array_sig_MEM_inv_TT = tree2array(t_TT_inv,branches=dis_str,selection=selection_sig)
    array_sig_MEM_inv_DY = tree2array(t_DY_inv,branches=dis_str,selection=selection_sig)
    array_back_MEM = tree2array(t_TT,branches=dis_str,selection=selection_back)
    array_sig_tot = np.concatenate((array_sig_MEM,array_sig_MEM_inv_TT,array_sig_MEM_inv_DY),axis=0)

    # Histograms properties #
    MEM_TT.SetLineWidth(2)
    MEM_TT.SetLineColor(kGreen+2)
    NNOut_TT.SetLineWidth(1)
    NNOut_TT.SetLineColor(kBlue+3)
    MEM_DY.SetLineWidth(2)
    MEM_DY.SetLineColor(kGreen+2)
    NNOut_DY.SetLineWidth(1)
    NNOut_DY.SetLineColor(kBlue+3)
    
    NNOut_TT_inv.SetLineWidth(0)
    NNOut_TT_inv.SetLineColor(kRed+2)
    NNOut_DY_inv.SetLineWidth(0)
    NNOut_DY_inv.SetLineColor(kRed+2)

    sig_dis_MEM.SetFillColor(kBlue+3)
    sig_dis_MEM_inv_TT.SetFillColor(kRed+2)
    sig_dis_MEM_inv_DY.SetFillColor(kGreen+2)
    sig_dis_MEM.SetLineWidth(0)
    sig_dis_MEM_inv_TT.SetLineWidth(0)
    sig_dis_MEM_inv_DY.SetLineWidth(0)

    back_dis_MEM.SetLineColor(kOrange+10)
    back_dis_MEM.SetLineWidth(3)

    # Clearing canvas and printing Titles#
    c8.Clear()

    title = TPaveText( .1, 0.92, .9, .99, "blNDC") 
    title.AddText('Specific Mass Plot : m_{H} = %0.f GeV, m_{A} = %0.f GeV'%(mHmA[i][0],mHmA[i][1]))
    title.SetFillColor(0)
    title.SetBorderSize(4)
    title.SetTextFont(43)
    title.SetTextSize(30)
    title.SetTextAlign(22)
    title.Draw()

    TT_weight.Draw()
    DY_weight.Draw()

    # Creating Pads #
    pad1 = TPad( 'pad1', 'Bottom left', 0.05, 0.01, 0.37, 0.46, -1 )
    pad2 = TPad( 'pad2', 'Bottom center',0.35, 0.01, 0.67, 0.46, -1 )
    pad3 = TPad( 'pad3', 'Bottom right',0.63, 0, .99, 0.5, -1 )
    pad4 = TPad( 'pad4', 'Top left', 0.05, 0.46, 0.37, 0.9, -1 )
    pad5 = TPad( 'pad5', 'Top center',0.35, 0.46, 0.67, 0.9, -1 )
    pad6 = TPad( 'pad6', 'Top right',0.65, 0.46, 0.97, 0.9, -1 )


    pad1.Draw()
    pad2.Draw()
    pad3.Draw()
    pad4.Draw()
    pad5.Draw()
    pad6.Draw()

    # TT weights #
    pad1.cd()
    MEM_TT.Sumw2()
    MEM_TT.SetTitle("MEM and NN Output")
    MEM_TT.GetXaxis().SetTitle("-log_{10}(Weight) [Normalized]")
    MEM_TT.GetXaxis().SetTitleSize(40)
    MEM_TT.GetXaxis().SetTitleOffset(1.5)
    rp_TT = TRatioPlot(MEM_TT,NNOut_TT)
    rp_TT.Draw()
    rp_TT.GetLowerRefGraph().SetMinimum(0)
    rp_TT.GetLowerRefGraph().SetMaximum(2)
    rp_TT.GetLowerRefYaxis().SetTitle("Ratio")
    rp_TT.GetLowerRefYaxis().SetTitleOffset(0.9)
    rp_TT.GetLowerRefYaxis().SetTitleSize(.06)
    rp_TT.GetUpperRefYaxis().SetTitle("Occurences")
    rp_TT.GetUpperRefYaxis().SetTitleOffset(0.9)
    rp_TT.GetUpperRefYaxis().SetTitleSize(.06)
    rp_TT.GetLowYaxis().SetNdivisions(505)
    rp_TT.SetUpTopMargin(0.2)
    rp_TT.SetLeftMargin(0.1)
    rp_TT.SetUpBottomMargin(0.5)
    rp_TT.SetLowBottomMargin(0.5)
    rp_TT.SetLowTopMargin(0)
    rp_TT.SetSeparationMargin(0.01)

    # DY weights #
    pad4.cd()
    MEM_DY.Sumw2()
    MEM_DY.SetTitle("MEM and NN Output")
    MEM_DY.GetXaxis().SetTitle("-log_{10}(Weight) [Normalized]")
    MEM_DY.GetXaxis().SetTitleSize(40)
    MEM_DY.GetXaxis().SetTitleOffset(1.5)
    rp_DY = TRatioPlot(MEM_DY,NNOut_DY)
    rp_DY.Draw()
    rp_DY.GetLowerRefGraph().SetMinimum(0)
    rp_DY.GetLowerRefGraph().SetMaximum(2)
    rp_DY.GetLowerRefYaxis().SetTitle("Ratio")
    rp_DY.GetLowerRefYaxis().SetTitleOffset(0.9)
    rp_DY.GetLowerRefYaxis().SetTitleSize(.06)
    rp_DY.GetUpperRefYaxis().SetTitle("Occurences")
    rp_DY.GetUpperRefYaxis().SetTitleOffset(0.9)
    rp_DY.GetUpperRefYaxis().SetTitleSize(.06)
    rp_DY.GetLowYaxis().SetNdivisions(505)
    rp_DY.SetUpTopMargin(0.2)
    rp_DY.SetLeftMargin(0.1)
    rp_DY.SetUpBottomMargin(0.5)
    rp_DY.SetLowBottomMargin(0.5)
    rp_DY.SetLowTopMargin(0)
    rp_DY.SetSeparationMargin(0.01)


   # TT invalid weights #
    NNOut_TT_copy = NNOut_TT.Clone()

    legend_NNOut_TT = TLegend(0.4,0.65,0.85,0.82)
    legend_NNOut_TT.SetHeader("Legend","C")
    legend_NNOut_TT.AddEntry(NNOut_TT_copy,"Valid weights : %0.f entries"%(NNOut_TT.GetEntries()))
    legend_NNOut_TT.AddEntry(NNOut_TT_inv,"Invalid weights : %0.f entries"%(NNOut_TT_inv.GetEntries()))

    NNOut_TT_stack = THStack("stack_TT","")
    NNOut_TT_stack.Add(NNOut_TT_copy)
    NNOut_TT_stack.Add(NNOut_TT_inv)

    NNOut_TT_inv.SetFillColor(kRed+2)
    NNOut_TT_copy.SetFillColor(kBlue+3)

    max_stack = NNOut_TT_stack.GetMaximum()*1.4

    pad2.cd()
    pad2.SetTopMargin(0.18)
    NNOut_TT_copy.Draw()
    NNOut_TT_copy.SetMaximum(max_stack)
    NNOut_TT_copy.GetXaxis().SetTitleSize(.06)
    NNOut_TT_copy.GetYaxis().SetTitleSize(.06)
    NNOut_TT_stack.Draw("same")
    NNOut_TT_copy.SetTitle("NN Output Comparison;-log_{10}(Weight) [Normalized];Occurences")
    legend_NNOut_TT.Draw()

    # DY invalid weights #
    NNOut_DY_copy = NNOut_DY.Clone()

    NNOut_DY_stack = THStack("stack_DY","")
    NNOut_DY_stack.Add(NNOut_DY_copy)
    NNOut_DY_stack.Add(NNOut_DY_inv)

    NNOut_DY_inv.SetFillColor(kRed+2)
    NNOut_DY_copy.SetFillColor(kBlue+3)

    max_stack = NNOut_DY_stack.GetMaximum()*1.2

    legend_NNOut_DY = TLegend(0.4,0.65,0.85,0.82)
    legend_NNOut_DY.SetHeader("Legend","C")
    legend_NNOut_DY.AddEntry(NNOut_DY_copy,"Valid weights : %0.f entries"%(NNOut_DY.GetEntries()))
    legend_NNOut_DY.AddEntry(NNOut_DY_inv,"Invalid weights : %0.f entries"%(NNOut_DY_inv.GetEntries()))

    pad5.cd()
    pad5.SetTopMargin(0.18)
    NNOut_DY_copy.Draw()
    NNOut_DY_copy.SetMaximum(max_stack)
    NNOut_DY_copy.GetXaxis().SetTitleSize(.06)
    NNOut_DY_copy.GetYaxis().SetTitleSize(.06)
    NNOut_DY_stack.Draw("same")
    NNOut_DY_copy.SetTitle("NN Output Comparison;-log_{10}(Weight) [Normalized];Occurences")
    legend_NNOut_DY.Draw()

    # MEM discriminant #
    
    stack_sig_dis = THStack("stack_sig_dis","")
    stack_sig_dis.Add(sig_dis_MEM)
    stack_sig_dis.Add(sig_dis_MEM_inv_TT)
    stack_sig_dis.Add(sig_dis_MEM_inv_DY)

    legend_sig_dis = TLegend(0.25,0.45,0.75,0.82)
    legend_sig_dis.SetHeader("Legend","C")
    legend_sig_dis.AddEntry(sig_dis_MEM,"Signal valid weights : %0.f entries"%(sig_dis_MEM.GetEntries()))
    legend_sig_dis.AddEntry(sig_dis_MEM_inv_TT,"Signal invalid TT weights : %0.f entries"%(sig_dis_MEM_inv_TT.GetEntries()))
    legend_sig_dis.AddEntry(sig_dis_MEM_inv_DY,"Signal invalid DY weights : %0.f entries"%(sig_dis_MEM_inv_DY.GetEntries()))
    legend_sig_dis.AddEntry(back_dis_MEM,"Background [Normalized]: %0.f entries"%(back_dis_MEM.GetEntries()))

    pad6.cd()
    pad6.SetTopMargin(0.18)
    back_dis_MEM.Scale((sig_dis_MEM.GetEntries()+sig_dis_MEM_inv_TT.GetEntries()+sig_dis_MEM_inv_DY.GetEntries())/back_dis_MEM.GetEntries())
    stack_sig_dis.Draw()
    back_dis_MEM.Draw("same")
    legend_sig_dis.Draw() 
    stack_sig_dis.SetTitle("MoMEMta Output discriminant;Discriminant;Occurences")
    stack_sig_dis.GetYaxis().SetTitleOffset(1.3)
    stack_sig_dis.GetXaxis().SetTitleSize(.06)
    stack_sig_dis.GetYaxis().SetTitleSize(.06)

    
    # Roc curve #
    array_tot = np.concatenate((array_sig_tot,array_back_MEM),axis=0)
    target = np.concatenate((np.zeros(array_sig_tot.shape[0]),np.ones(array_back_MEM.shape[0])),axis=0) 
    back_eff,sig_eff,tresholds = roc_curve(target,array_tot)
    auc_sigback = roc_auc_score(target,array_tot)

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    fig = plt.figure(1)
    plt.plot(sig_eff,back_eff,'b',label=('AUC = %0.5f'%(auc_sigback)))
    plt.title('ROC curve : Signal vs Background')
    plt.legend(loc='upper left')
    plt.xlabel('Signal selection efficiency')
    plt.ylabel('Background selection efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid()
    #plt.show()
    fig.savefig(path+'ROC_sigVSback.png',bbox_inches='tight')
    plt.close()

    pad3.cd()
    ROC_image = TImage.Open(path+'ROC_sigVSback.png')
    ROC_image.Draw()

    c8.cd()
    line = TLine(0.64,0.02,0.64,0.9)
    line.SetLineColor(kBlack)
    line.SetLineWidth(5)
    line.Draw()

    gPad.Update()
    #input ("Press any key to end")
    # Save Canvas #
    c8.Print(path_mass+'mH_'+str(int(mHmA[i][0]))+'_mA_'+str(int(mHmA[i][1]))+'.png')



