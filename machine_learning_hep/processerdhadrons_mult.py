#############################################################################
##  © Copyright CERN 2024. All rights not expressly granted are reserved.  ##
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

#pylint: disable=import-error, no-name-in-module, consider-using-f-string, too-many-statements, too-many-branches, too-many-arguments, too-many-instance-attributes, too-many-locals

"""
main script for doing data processing, machine learning and analysis
"""
import math
import array
import os
import numpy as np
from ROOT import TFile, TH1F
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities_files import create_folder_struc
from machine_learning_hep.utilities import seldf_singlevar, seldf_singlevar_inclusive
from machine_learning_hep.utilities import mergerootfiles, read_df
from machine_learning_hep.utilities import get_timestamp_string
from machine_learning_hep.utils.hist import fill_hist
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.processer import Processer
from machine_learning_hep.bitwise import tag_bit_df

# pylint: disable=invalid-name
class ProcesserDhadrons_mult(Processer):
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes

    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                         d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                         p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                         p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                         d_results, typean, runlisttrigger, d_mcreweights)

        self.v_invmass = datap["variables"].get("var_inv_mass", "fM")
        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.s_presel_gen_eff = datap["analysis"][self.typean]['presel_gen_eff']
        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"]
        self.corr_eff_mult = datap["analysis"][self.typean]["corrEffMult"]
        self.mc_cut_on_binning2 = datap["analysis"][self.typean].get("mc_cut_on_binning2", True)

        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #self.sel_final_fineptbins = datap["analysis"][self.typean]["sel_final_fineptbins"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger
        self.event_cand_validation = datap["analysis"][self.typean].get("event_cand_validation", "")
        if "event_cand_validation" not in datap["analysis"][self.typean]:
            self.event_cand_validation = False
        self.usetriggcorrfunc = \
                datap["analysis"][self.typean]["triggersel"].get("usetriggcorrfunc", None)
        self.weightfunc = None
        self.weighthist = None
        if self.usetriggcorrfunc is not None and self.mcordata == "data":
            filename = os.path.join(self.d_mcreweights, "trigger%s.root" % self.typean)
            if os.path.exists(filename):
                weight_file = TFile.Open(filename, "read")
                self.weightfunc = weight_file.Get("func%s_norm" % self.typean)
                self.weighthist = weight_file.Get("hist%s_norm" % self.typean)
                self.weighthist.SetDirectory(0)
                weight_file.Close()
            else:
                print("trigger correction file", filename, "doesnt exist")
        self.nbinshisto = datap["analysis"][self.typean]["nbinshisto"]
        self.minvaluehisto = datap["analysis"][self.typean]["minvaluehisto"]
        self.maxvaluehisto = datap["analysis"][self.typean]["maxvaluehisto"]
        self.mass = datap["mass"]

        # Event re-weighting MC
        self.event_weighting_mc = datap["analysis"][self.typean].get("event_weighting_mc", {})
        self.event_weighting_mc = self.event_weighting_mc.get(self.period, {})

    @staticmethod
    def make_weights(col, func, hist, use_func):
        """Helper function to extract weights

        Args:
            col: np.array
                array to evaluate/run over
            func: ROOT.TF1
                ROOT function to use for evaluation
            hist: TH1
                ROOT histogram used for getting weights
            use_func: bool
                whether or not to use func (otherwise hist)

        Returns:
            iterable
        """

        if use_func:
            return [func.Eval(x) for x in col]
        def reg(value):
            # warning, the histogram has empty bins at high mult.
            # (>125 ntrkl) so a check is needed to avoid a 1/0 division
            # when computing the inverse of the weight
            return value if value != 0. else 1.
        return [reg(hist.GetBinContent(hist.FindBin(iw))) for iw in col]

    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = read_df(self.l_evtorig[index])
        neventsorig = len(dfevtorig)
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        #neventsaftertrigger = len(dfevtorig)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
        #neventsafterrunsel = len(dfevtorig)
        if self.s_evtsel is not None:
            dfevtevtsel = dfevtorig.query(self.s_evtsel)
        else:
            dfevtevtsel = dfevtorig

        #validation plot for event selection
        neventsafterevtsel = len(dfevtevtsel)
        histonorm = TH1F("histonorm", "histonorm", 10, 0, 10)
        histonorm.SetBinContent(1, neventsorig)
        histonorm.GetXaxis().SetBinLabel(1, "tot events")
        histonorm.SetBinContent(2, neventsafterevtsel)
        histonorm.GetXaxis().SetBinLabel(2, "tot events after evt sel")
        for ibin2, _ in enumerate(self.lvar2_binmin):
            binneddf = seldf_singlevar_inclusive(dfevtevtsel, self.v_var2_binning_gen, \
                self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
            histonorm.SetBinContent(3 + ibin2, len(binneddf))
            histonorm.GetXaxis().SetBinLabel(3 + ibin2, \
                        "tot events after mult sel %d - %d" % \
                        (self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2]))
        histonorm.Write()

        myfile.cd()
        hEvents = TH1F('all_events', 'all_events', 1, -0.5, 0.5)
        hSelEvents = TH1F('sel_events', 'sel_events', 1, -0.5, 0.5)
        hEvents.SetBinContent(1, len(dfevtorig))
        hSelEvents.SetBinContent(1, len(dfevtevtsel))

        hEvents.Write()
        hSelEvents.Write()

        list_df_recodtrig = []

        for ipt in range(self.p_nptfinbins): # pylint: disable=too-many-nested-blocks
            bin_id = self.bin_matching[ipt]
            df = read_df(self.mptfiles_recoskmldec[bin_id][index])
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)
            if self.runlistrigger is not None:
                df = selectdfrunlist(df, \
                    self.run_param[self.runlistrigger], "run_number")
            if self.doml is True:
                df = df.query(self.l_selml[ipt])
            list_df_recodtrig.append(df)
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            if self.do_custom_analysis_cuts:
                df = self.apply_cuts_ptbin(df, ipt)

            for ibin2, _ in enumerate(self.lvar2_binmin):

                if self.mltype == "MultiClassification":
                    suffix = "%s%d_%d_%.2f%.2f%s_%.2f_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt][0],
                              self.lpt_probcutfin[ipt][1], self.v_var2_binning,
                              self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                else:
                    suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt],
                              self.v_var2_binning,
                              self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_weight = TH1F("h_invmass_weight" + suffix, "", self.p_num_bins,
                                        self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar_inclusive(df, self.v_var2_binning, \
                                         self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                fill_hist(h_invmass, df_bin[self.v_invmass])
                if self.usetriggcorrfunc is not None and self.mcordata == "data":
                    weights = self.make_weights(df_bin[self.v_var2_binning_gen], self.weightfunc,
                                                self.weighthist, self.usetriggcorrfunc)

                    weightsinv = [1./weight for weight in weights]
                    fill_hist(h_invmass_weight, df_bin[self.v_invmass], weights=weightsinv)
                myfile.cd()
                h_invmass.Write()
                h_invmass_weight.Write()

                if self.mcordata == "mc":
                    df_bin[self.v_ismcrefl] = np.array(tag_bit_df(df_bin, self.v_bitvar,
                                                                  self.b_mcrefl), dtype=int)
                    df_bin_sig = df_bin[df_bin[self.v_ismcsignal] == 1]
                    df_bin_refl = df_bin[df_bin[self.v_ismcrefl] == 1]
                    h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                         self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                          self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    fill_hist(h_invmass_sig, df_bin_sig[self.v_invmass])
                    fill_hist(h_invmass_refl, df_bin_refl[self.v_invmass])
                    myfile.cd()
                    h_invmass_sig.Write()
                    h_invmass_refl.Write()

        if self.event_cand_validation is True:
            label = "h%s" % self.v_var2_binning_gen
            histomult = TH1F(label, label, self.nbinshisto,
                             self.minvaluehisto, self.maxvaluehisto)
            fill_hist(histomult, dfevtevtsel[self.v_var2_binning_gen])
            histomult.Write()

    def get_reweighted_count(self, dfsel, ibin=None):
        """Apply event weights

        Args:
            dfsel: pandas.DataFrame
                dataframe with column to apply weights for
            ibin: int (optional)
                Try to extract ibin'th entry from what is loaded from the database.
                By default, ibin corresponds to the multiplcity bin under study

        Returns:
            float: nominal value,
            float: error

        """

        def no_weights(df_):
            val = len(df_)
            return val, math.sqrt(val)

        event_weighting_mc = {}
        if self.event_weighting_mc and ibin is not None \
                and len(self.event_weighting_mc) - 1 >= ibin:
            # Check is there is a dictionary with desired info
            event_weighting_mc = self.event_weighting_mc[ibin]

        # If there were explicit info in the analysis database, assume that all fields exist
        # If incomplete, there will be a mix-up between these values and default values
        filepath = event_weighting_mc.get("filepath", os.path.join(self.d_mcreweights,
                                                                   self.n_mcreweights))
        if not os.path.exists(filepath):
            print(f"Could not find filepath {filepath} for MC event weighting." \
                    "Compute unweighted values...")
            return no_weights(dfsel)

        weight_file = TFile.Open(filepath, "read")
        histo_name = event_weighting_mc.get("histo_name", "Weights0")
        weights = weight_file.Get(histo_name)

        if not weights:
            print(f"Could not find histogram {histo_name} for MC event weighting." \
                    "Compute unweighted values...")
            return no_weights(dfsel)

        weight_according_to = event_weighting_mc.get("according_to", self.v_var2_binning_gen)

        w = [weights.GetBinContent(weights.FindBin(v)) for v in
             dfsel[weight_according_to]]
        val = sum(w)
        err = math.sqrt(sum(map(lambda i: i * i, w)))
        #print('reweighting sum: {:.1f} +- {:.1f} -> {:.1f} +- {:.1f} (zeroes: {})' \
        #      .format(len(dfsel), math.sqrt(len(dfsel)), val, err, w.count(0.)))
        return val, err

    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        out_file = TFile.Open(self.l_histoeff[index], "recreate")
        h_list = []
        for ibin2, _ in enumerate(self.lvar2_binmin):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen,
                                            self.lvar2_binmin[ibin2],
                                            self.lvar2_binmax[ibin2])
            n_bins = len(self.lpt_finbinmin)
            analysis_bin_lims_temp = self.lpt_finbinmin.copy()
            analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
            analysis_bin_lims = array.array('f', analysis_bin_lims_temp)

            def make_histo(name, title,
                           name_extra=stringbin2,
                           bins=n_bins,
                           binning=analysis_bin_lims):
                histo = TH1F(name + name_extra, title, bins, binning)
                h_list.append(histo)
                return histo

            h_gen_pr = make_histo("h_gen_pr",
                                  "Prompt Generated in acceptance |y|<0.5")
            h_presel_pr = make_histo("h_presel_pr",
                                     "Prompt Reco in acc |#eta|<0.8 and sel")
            h_presel_pr_wotof = make_histo("h_presel_pr_wotof",
                                           "Prompt Reco in acc woTOF |#eta|<0.8 and pre-sel")
            h_presel_pr_wtof = make_histo("h_presel_pr_wtof",
                                          "Prompt Reco in acc wTOF |#eta|<0.8 and pre-sel")
            h_sel_pr = make_histo("h_sel_pr",
                                  "Prompt Reco and sel in acc |#eta|<0.8 and sel")
            h_sel_pr_wotof = make_histo("h_sel_pr_wotof",
                                        "Prompt Reco and sel woTOF in acc |#eta|<0.8")
            h_sel_pr_wtof = make_histo("h_sel_pr_wtof",
                                       "Prompt Reco and sel wTOF in acc |#eta|<0.8")
            h_gen_fd = make_histo("h_gen_fd",
                                  "FD Generated in acceptance |y|<0.5")
            h_presel_fd = make_histo("h_presel_fd",
                                     "FD Reco in acc |#eta|<0.8 and sel")
            h_sel_fd = make_histo("h_sel_fd",
                                  "FD Reco and sel in acc |#eta|<0.8 and sel")

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                df_mc_reco = read_df(self.mptfiles_recoskmldec[bin_id][index])
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                if self.runlistrigger is not None:
                    df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger], "run_number")
                df_mc_gen = read_df(self.mptfiles_gensk[bin_id][index])
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
                if self.s_evtsel is not None:
                    df_mc_gen = df_mc_gen.query(self.s_evtsel)
                if self.runlistrigger is not None:
                    df_mc_gen = selectdfrunlist(df_mc_gen, \
                             self.run_param[self.runlistrigger], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                # Whether or not to cut on the 2nd binning variable
                if self.mc_cut_on_binning2:
                    df_mc_reco = seldf_singlevar_inclusive(df_mc_reco, self.v_var2_binning_gen, \
                                                 self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                    df_mc_gen = seldf_singlevar_inclusive(df_mc_gen, self.v_var2_binning_gen, \
                                                self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_gen_sel_pr = df_mc_gen.loc[(df_mc_gen.ismcprompt == 1) & (df_mc_gen.ismcsignal == 1)]
                df_reco_presel_pr = df_mc_reco.loc[(df_mc_reco.ismcprompt == 1) & (df_mc_reco.ismcsignal == 1)]
                df_reco_sel_pr = None
                if self.doml is True:
                    df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[ipt])
                else:
                    df_reco_sel_pr = df_reco_presel_pr.copy()
                df_gen_sel_fd = df_mc_gen.loc[(df_mc_gen.ismcfd == 1) & (df_mc_gen.ismcsignal == 1)]
                df_reco_presel_fd = df_mc_reco.loc[(df_mc_reco.ismcfd == 1) & (df_mc_reco.ismcsignal == 1)]
                df_reco_sel_fd = None
                if self.doml is True:
                    df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[ipt])
                else:
                    df_reco_sel_fd = df_reco_presel_fd.copy()

                if self.do_custom_analysis_cuts:
                    df_reco_sel_pr = self.apply_cuts_ptbin(df_reco_sel_pr, ipt)
                    df_reco_sel_fd = self.apply_cuts_ptbin(df_reco_sel_fd, ipt)

                def set_content(df_to_use, histogram,
                                i_b=ibin2, b_c=bincounter):
                    if self.corr_eff_mult[i_b] is True:
                        val, err = self.get_reweighted_count(df_to_use, i_b)
                    else:
                        val = len(df_to_use)
                        err = math.sqrt(val)
                    histogram.SetBinContent(b_c + 1, val)
                    histogram.SetBinError(b_c + 1, err)

                set_content(df_gen_sel_pr, h_gen_pr)
                if "nsigTOF_Pr_0" in df_reco_presel_pr:
                    set_content(df_reco_presel_pr[df_reco_presel_pr.nsigTOF_Pr_0 < -998],
                                h_presel_pr_wotof)
                    set_content(df_reco_presel_pr[df_reco_presel_pr.nsigTOF_Pr_0 > -998],
                                h_presel_pr_wtof)
                set_content(df_reco_presel_pr, h_presel_pr)
                set_content(df_reco_sel_pr, h_sel_pr)
                if "nsigTOF_Pr_0" in df_reco_sel_pr:
                    set_content(df_reco_sel_pr[df_reco_sel_pr.nsigTOF_Pr_0 < -998], h_sel_pr_wotof)
                    set_content(df_reco_sel_pr[df_reco_sel_pr.nsigTOF_Pr_0 > -998], h_sel_pr_wtof)
                set_content(df_gen_sel_fd, h_gen_fd)
                set_content(df_reco_presel_fd, h_presel_fd)
                set_content(df_reco_sel_fd, h_sel_fd)

                bincounter = bincounter + 1

            out_file.cd()
            for h in h_list:
                h.Write()
            h_list = []

    def process_efficiency(self):
        print("Doing efficiencies", self.mcordata, self.period)
        print("Using run selection for eff histo", \
               self.runlistrigger, "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        else:
            print("No extra selection needed since we are doing std analysis")
        for ibin2 in range(len(self.lvar2_binmin)):
            if self.corr_eff_mult[ibin2] is True:
                print("Reweighting efficiencies for bin", ibin2)
            else:
                print("Not reweighting efficiencies for bin", ibin2)

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_efficiency_single, arguments, self.p_chunksizeunp)
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/histoeff_{self.period}/{get_timestamp_string()}/"
        mergerootfiles(self.l_histoeff, self.n_fileeff, tmp_merged)
