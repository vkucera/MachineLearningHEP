#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
main script for doing final stage analysis
"""
import os
from ROOT import TFile, TH1F, TCanvas  # pylint: disable=import-error, no-name-in-module
import pickle
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile

from machine_learning_hep.utilities import merge_method
class Analyzer: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "analyzer"
    def __init__(self, datap, run_param, mcordata):
        self.datap = datap
        self.run_param = run_param
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][self.mcordata]["unmerged_tree_dir"])
        self.p_period = datap["multi"][self.mcordata]["period"]
        self.p_nparall = datap["multi"][self.mcordata]["nprocessesparallel"]
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.p_dofullevtmerge = datap["dofullevtmerge"]
        self.p_modelname = datap["analysis"]["modelname"]

        #directories

        #namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.dlper_reco = datap["analysis"][self.mcordata]["pkl_skimmed_decmerged"]
        self.d_recomergedper = datap["analysis"][self.mcordata]["pkl_skimmed_decmergedallp"]
        self.d_results = datap["analysis"][self.mcordata]["results"]
        self.d_resultsallp = datap["analysis"][self.mcordata]["resultsallp"]
        self.lpt_probcutfin = datap["analysis"]["probcutoptimal"]
        self.lpt_probcutpre = datap["analysis"]["probcutpresel"][self.mcordata]
        self.lpt_reco = [self.n_reco.replace(".pkl", "_%s%d_%d_%.2f.pkl" % \
                           (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                            self.lpt_probcutpre[i])) for i in range(self.p_nptbins)]
        self.lpt_gen = [self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lptper_gen = [[os.path.join(direc, self.lpt_gen[ipt]) \
                              for direc in self.dlper_reco] \
                              for ipt in range(self.p_nptbins)]
        self.lptper_reco = [[os.path.join(direc, self.lpt_reco[ipt]) \
                              for direc in self.dlper_reco] \
                              for ipt in range(self.p_nptbins)]
        self.lpt_genmergedp = [os.path.join(self.d_recomergedper, self.lpt_gen[ipt]) \
                              for ipt in range(self.p_nptbins)]
        self.lpt_recomergedp = [os.path.join(self.d_recomergedper, self.lpt_reco[ipt]) \
                              for ipt in range(self.p_nptbins)]

        self.n_filemass = datap["files_names"]["histofilename"]
        self.lpt_filemass = [self.n_filemass.replace(".root", "%d_%d_%.2f.root" % \
                (self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt], \
                 self.lpt_probcutfin[ipt])) for ipt in range(self.p_nptbins)]
        self.mperpt_filemass = [[os.path.join(mydir, self.lpt_filemass[ipt]) \
                                 for mydir in self.d_results] \
                                 for ipt in range(self.p_nptbins)]
        self.lptmerged_filemass = [os.path.join(self.d_resultsallp, filen)
                                  for filen in self.lpt_filemass]
        self.l_selml = ["y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[ipt]) \
                       for ipt in range(self.p_nptbins)]

        self.p_mass_fit_lim = datap["analysis"]['mass_fit_lim']
        self.p_bin_width = datap["analysis"]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))

        for ipt in range(self.p_nptbins):
            merge_method(self.lptper_gen[ipt], self.lpt_genmergedp[ipt])
            merge_method(self.lptper_reco[ipt], self.lpt_recomergedp[ipt])

    def extractmass(self, filedf, seldf, histoname, nbins, minv, maxv):
        df = pickle.load(openfile(filedf, "rb"))
        df = df.query(seldf)
        h_invmass = TH1F("hmass", "", nbins, minv, maxv)
        fill_hist(h_invmass, df.inv_mass)
        return h_invmass

    def histomass(self):
        for ipt in range(self.p_nptbins):
            for indexp in range(self.prodnumber):
                myfile = TFile.Open(self.mperpt_filemass[ipt][indexp], "recreate")
                h_invmass = self.extractmass(self.lptper_reco[ipt][indexp],
                                             self.l_selml[ipt],
                                             "hmass", self.p_num_bins,
                                             self.p_mass_fit_lim[0],
                                             self.p_mass_fit_lim[1])

                myfile.cd()
                h_invmass.Write()

            myfile = TFile.Open(self.lptmerged_filemass[ipt], "recreate")
            h_invmass = self.extractmass(self.lpt_recomergedp[ipt],
                                        self.l_selml[ipt],
                                        "hmass", self.p_num_bins,
                                        self.p_mass_fit_lim[0],
                                        self.p_mass_fit_lim[1])

            myfile.cd()
            h_invmass.Write()
