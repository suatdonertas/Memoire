# Libraries #
import sys  
import glob 
import os
import re
import argparse
import math 
import numpy as np
import ROOT 
from ROOT import TFile, TTree, TCanvas, TPaveText, TPad, gPad, gStyle, TLegend, TH1F, gROOT, TLegend, TRatioPlot, TStyle, TF1, THStack, TColor, TImage, TLine, TPaveStats 
from ROOT import kBlack, kBlue, kRed, kOrange, kYellow, kGreen

gROOT.SetBatch(True)

gStyle.SetOptStat("")
gStyle.SetTitleFontSize(.11)
gStyle.SetLabelSize(.03, "XY")

f = TFile.Open("outputDY.root")
t = f.Get("tree")

t.Draw("jj_M:NN_jj_M>>jj(100,0,2200,100,0,2200)","","colz")
jj = gROOT.FindObject("jj")

t.Draw("lljj_M:NN_lljj_M>>lljj(100,0,2800,100,0,2800)","","colz")
lljj = gROOT.FindObject("lljj")



c1 = TCanvas( 'c1', 'rec', 200, 10, 1400, 600 )


pad1 = TPad( 'pad1', 'mbb', 0.02, 0.05, 0.49, 1, -1 )
pad2 = TPad( 'pad2', 'mllbb', 0.51, 0.05, 0.98, 1, -1 )

pad1.Draw()
pad2.Draw()

pad1.cd()
pad1.SetTopMargin(0.15)
pad1.SetLeftMargin(0.2)
pad1.SetRightMargin(0.2)
pad1.SetBottomMargin(0.15)
jj.Draw("colz")
jj.SetTitle("M_{bb} reconstruction;M_{bb} from input [GeV];M_{bb} from network [GeV]; Occurences")
jj.GetXaxis().SetTitleSize(.06)
jj.GetYaxis().SetTitleSize(.06)
jj.GetZaxis().SetTitleSize(.06)
jj.GetXaxis().SetTitleOffset(1.0)
jj.GetYaxis().SetTitleOffset(1.3)
jj.GetZaxis().SetTitleOffset(1.3)

pad2.cd()
pad2.SetTopMargin(0.15)
pad2.SetLeftMargin(0.2)
pad2.SetRightMargin(0.2)
pad2.SetBottomMargin(0.15)
lljj.Draw("colz")
lljj.SetTitle("M_{llbb} reconstruction;M_{llbb} from input [GeV];M_{llbb} from network [GeV]; Occurences")
lljj.GetXaxis().SetTitleSize(.06)
lljj.GetYaxis().SetTitleSize(.06)
lljj.GetZaxis().SetTitleSize(.06)
lljj.GetXaxis().SetTitleOffset(1.0)
lljj.GetYaxis().SetTitleOffset(1.3)
lljj.GetZaxis().SetTitleOffset(1.3)


gPad.Update()

c1.Print("masses.png")
