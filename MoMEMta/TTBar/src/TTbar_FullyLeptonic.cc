/*
 *  MoMEMta: a modular implementation of the Matrix Element Method
 *  Copyright (C) 2016  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <signal.h>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include <momemta/ConfigurationReader.h>
#include <momemta/Logging.h>
#include <momemta/MoMEMta.h>
#include <momemta/Unused.h>

#include <chrono>
#include <memory>

// ROOT
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <Math/PtEtaPhiM4D.h>
#include <Math/LorentzVector.h>


using namespace std;
using namespace std::chrono;

using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;

/*
 * Example executable file loading an input sample of events,
 * computing weights using MoMEMta in the fully-leptonic ttbar hypothesis,
 * and saving these weights along with a copy of the event content in an output file.
 */

void normalizeInput(LorentzVector& p4) {
    if (p4.M() > 0)
        return;

    // Increase the energy until M is positive
    p4.SetE(p4.P());
    while (p4.M2() < 0) {
        double delta = p4.E() * 1e-5;
        p4.SetE(p4.E() + delta);
    };
}

// Command line
DEFINE_string(output, "output.root", "Name of the output file containing the weights");
DEFINE_uint64(from, 0, "First entry to process");
DEFINE_uint64(to, 0, "Last entry to process. If 0, process all the inputs");
DEFINE_bool(verbose, false, "Enable verbose mode");
DEFINE_string(confs_dir, CONFS_DIRECTORY, "Directory containing configurations");
DEFINE_string(input, "input.root", "Name of the input file of the tree");

int main(int argc, char** argv) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_verbose)
        logging::set_level(logging::level::debug);
    else
        logging::set_level(logging::level::info);

    using std::swap;

    /*
     * Load events from input file, retrieve reconstructed particles and MET
     */
    TChain chain("t");
    string INPUT_DIR = "/nfs/scratch/fynu/asaggio/CMSSW_8_0_30/src/cp3_llbb/ZATools/factories_ZA/fourVectors_for_Florian/slurm/output/";
    string file = INPUT_DIR+FLAGS_input;
    LOG(info)<<"Directory : "+INPUT_DIR;
    LOG(info)<<"Using file : "+FLAGS_input; 

    chain.Add(file.c_str());
    TTreeReader myReader(&chain);

    TTreeReaderValue<LorentzVectorE> lep_plus_p4E(myReader, "lep1_p4");
    TTreeReaderValue<LorentzVectorE> lep_minus_p4E(myReader, "lep2_p4");
    TTreeReaderValue<LorentzVectorE> jet1_p4E(myReader, "jet1_p4");
    TTreeReaderValue<LorentzVectorE> jet2_p4E(myReader, "jet2_p4");
    //TTreeReaderValue<float> MET_met(myReader, "MET_met");
    //TTreeReaderValue<float> MET_phi(myReader, "MET_phi");
    //TTreeReaderValue<int> leading_lep_PID(myReader, "leadLepPID");

    /*
     * Define output TTree, which will be a clone of the input tree,
     * with the addition of the weights we're computing (including uncertainty and computation time)
     */
    std::unique_ptr<TTree> out_tree(chain.CloneTree(0));
    double weight_TT, weight_TT_err, weight_TT_time;
    out_tree->Branch("weight_TT", &weight_TT);
    out_tree->Branch("weight_TT_err", &weight_TT_err);
    out_tree->Branch("weight_TT_time", &weight_TT_time);

    /*
     * Prepare MoMEMta to compute the weights
     */

    // Construct the ConfigurationReader from the Lua file
    LOG(info) << "Reading configuration files from '" << FLAGS_confs_dir << "'";
//#ConfigurationReader configuration("../confs/TTbar_FullyLeptonic.lua");
    ConfigurationReader configuration(FLAGS_confs_dir + "TTbar_FullyLeptonic.lua");


    // Instantiate MoMEMta using a **frozen** configuration
    MoMEMta weight(configuration.freeze());
    
    // To and From parameters
    size_t to = 0;
    if (FLAGS_to > 0)
        to = FLAGS_to;
    else
        to = chain.GetEntries();

    size_t from = 0;
    if (FLAGS_from > 0)
        from = std::max(from, FLAGS_from);

    if (from >= to) {
        LOG(fatal) << "First entry to process is greater than the total number of entries (" << from << " >= " << to << ")";
        abort();
    }

    myReader.SetEntriesRange(from, to);

    size_t selected = 0;
    LOG(info) << "Processing " << to - from << " entries, from " << from << " to " << to;

    /*
     * Loop over all input events
     */
    while (myReader.Next()) {
        LOG(info) << "Processing entry " << myReader.GetCurrentEntry() << " up to " << to;
        selected++;
        /*
         * Prepare the LorentzVectors passed to MoMEMta:
         * In the input file they are written in the PtEtaPhiM<float> basis,
         * while MoMEMta expects PxPyPzE<double>, so we have to perform this change of basis:
         *
         * We define here Particles, allowing MoMEMta to correctly map the inputs to the configuration file.
         * The string identifier used here must be the same as used to declare the inputs in the config file
         */
        momemta::Particle lep_plus("lepton1",  LorentzVector { lep_plus_p4E->Px(), lep_plus_p4E->Py(), lep_plus_p4E->Pz(), lep_plus_p4E->E() });
        momemta::Particle lep_minus("lepton2", LorentzVector { lep_minus_p4E->Px(), lep_minus_p4E->Py(), lep_minus_p4E->Pz(), lep_minus_p4E->E() });
        momemta::Particle bjet1("bjet1", LorentzVector { jet1_p4E->Px(), jet1_p4E->Py(), jet1_p4E->Pz(), jet1_p4E->E() });
        momemta::Particle bjet2("bjet2", LorentzVector { jet2_p4E->Px(), jet2_p4E->Py(), jet2_p4E->Pz(), jet2_p4E->E() });
       // momemta::Particle lep_plus("lepton1", lep_plus_p4E);
        //momemta::Particle lep_minus("lepton2", lep_minus_p4E);
        //momemta::Particle jet1("jet1", jet1_p4E);
        //momemta::Particle jet2("jet2", jet2_p4E);

        // Due to numerical instability, the mass can sometimes be negative. If it's the case, change the energy in order to be mass-positive
        normalizeInput(lep_plus.p4);
        normalizeInput(lep_minus.p4);
        normalizeInput(bjet1.p4);
        normalizeInput(bjet2.p4);

        //LorentzVectorE met_p4E { *MET_met, 0, *MET_phi, 0 };
        //LorentzVector met_p4 { met_p4E.Px(), met_p4E.Py(), met_p4E.Pz(), met_p4E.E() };
        
        // Ensure the leptons are given in the correct order w.r.t their charge 
        //if (*leading_lep_PID < 0)
        //    swap(lep_plus, lep_minus);

        auto start_time = system_clock::now();
        // Compute the weights!
        //std::vector<std::pair<double, double>> weights = weight.computeWeights({lep_minus, bjet1, lep_plus, bjet2}, met_p4);
        std::vector<std::pair<double, double>> weights = weight.computeWeights({lep_minus, bjet1, lep_plus, bjet2});
        auto end_time = system_clock::now();

        // Retrieve the weight and uncertainty
        weight_TT = weights.back().first;
        weight_TT_err = weights.back().second;
        weight_TT_time = std::chrono::duration_cast<milliseconds>(end_time - start_time).count();

        LOG(debug) << "Event " << myReader.GetCurrentEntry() << " result: " << weight_TT << " +- " << weight_TT_err;
        LOG(info) << "Weight computed in " << weight_TT_time << "ms";
    
        LOG(debug) << "Filling tree...";
        out_tree->Fill();
        LOG(debug) << "Done, next event";
    }

    // Save our output TTree
    out_tree->SaveAs(FLAGS_output.c_str());

    LOG(info) << "Processing done: " << selected << " selected events";

    out_tree->Write();

    return 0;
}
