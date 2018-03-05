#include "TTree.h"
#include "TFile.h"
#include "TSystem.h"
#include "TH2F.h"
#include <string>
#include <sstream>
#include <fstream>


using namespace std;
int MassPlane(){
    TCanvas * c1 = new TCanvas("c1", "Comp", 800, 600);
    TH2F* mass_plane = new TH2F("mass_plane","mass_plane",1000,0,1000,1000,1000);
   
    TFile *f = TFile::Open("/home/ucl/cp3/fbury/storage/invmass.root");
    TTree *t = (TTree*)f->Get("t");     

    t->Draw("lljj_M:jj_M","lljj_M<1000 && jj_M<1000","COLZ");

    return 0;
}
