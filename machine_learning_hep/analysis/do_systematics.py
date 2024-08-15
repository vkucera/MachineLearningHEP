#  © Copyright CERN 2018. All rights not expressly granted are reserved.  #
#                 Author: Gian.Michele.Innocenti@cern.ch                  #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

"""
main script for doing final stage analysis
"""
import argparse

# pylint: disable=too-many-lines, line-too-long
import os
from array import array
from math import sqrt

import yaml

# pylint: disable=import-error, no-name-in-module
from ROOT import TH1F, TCanvas, TFile, TGraphAsymmErrors, TLatex, TLegend, gStyle  # , TLine

from machine_learning_hep.utils.hist import get_axis, print_histogram

from machine_learning_hep.do_variations import (
    format_varlabel,
    format_varname,
    healthy_structure,
)
from machine_learning_hep.logger import get_logger

# HF specific imports
from machine_learning_hep.utilities import (
    combine_graphs,
    draw_latex,
    get_colour,
    get_marker,
    get_plot_range,
    get_y_window_gr,
    get_y_window_his,
    make_message_notfound,
    # make_plot,
    setup_canvas,
    setup_histogram,
    setup_legend,
    setup_tgraph,
)


def shrink_err_x(graph, width=0.1):
    for i in range(graph.GetN()):
        graph.SetPointEXlow(i, width)
        graph.SetPointEXhigh(i, width)


# pylint: disable=too-many-instance-attributes, too-many-statements
class AnalyzerJetSystematics:
    def __init__(self, path_database_analysis: str, typean: str):
        self.logger = get_logger()
        self.typean = typean

        with open(path_database_analysis, "r", encoding="utf-8") as file_in:
            db_analysis = yaml.safe_load(file_in)
        case = list(db_analysis.keys())[0]
        datap = db_analysis[case]

        # plotting
        # LaTeX string
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.v_varshape_latex = datap["analysis"][self.typean]["var_shape_latex"]

        # first variable (hadron pt)
        self.v_var_binning = datap["var_binning"]  # name
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)  # number of bins

        # second variable (jet pt)
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]  # name
        self.lvar2_binmin_reco = datap["analysis"][self.typean]["sel_binmin2_reco"]
        self.lvar2_binmax_reco = datap["analysis"][self.typean]["sel_binmax2_reco"]
        self.lvar2_binmin_gen = datap["analysis"][self.typean]["sel_binmin2_gen"]
        self.lvar2_binmax_gen = datap["analysis"][self.typean]["sel_binmax2_gen"]
        self.p_nbin2_gen = len(self.lvar2_binmin_gen)  # number of gen bins

        # observable (z, shape,...)
        # self.lvarshape_binmin_reco = \
        #     datap["analysis"][self.typean]["sel_binminshape_reco"]
        # self.lvarshape_binmax_reco = \
        #     datap["analysis"][self.typean]["sel_binmaxshape_reco"]
        # self.lvarshape_binmin_gen = datap["analysis"][self.typean]["sel_binminshape_gen"]
        # self.lvarshape_binmax_gen = datap["analysis"][self.typean]["sel_binmaxshape_gen"]
        # self.p_nbinshape_gen = len(self.lvarshape_binmin_gen)  # number of gen bins

        # systematics variations

        # models to compare with
        # POWHEG + PYTHIA 6
        # PYTHIA 8
        # self.pythia8_prompt_variations_legend = datap["analysis"][self.typean]["pythia8_prompt_variations_legend"]

        # unfolding
        self.niter_unfolding = datap["analysis"][self.typean]["unfolding_iterations"]
        self.choice_iter_unfolding = datap["analysis"][self.typean]["unfolding_iterations_sel"]

        # systematics
        # import parameters of variations from the variation database

        path_database_variations = datap["analysis"][self.typean]["variations_db"]
        if not path_database_variations:
            self.logger.critical(make_message_notfound("the variation database"))
        if "/" not in path_database_variations:
            path_database_variations = f"{os.path.dirname(path_database_analysis)}/{path_database_variations}"
        with open(path_database_variations, "r", encoding="utf-8") as file_sys:
            db_variations = yaml.safe_load(file_sys)

        if not healthy_structure(db_variations):
            self.logger.critical("Bad structure of the variation database.")
        db_variations = db_variations["categories"]
        self.systematic_catnames = [catname for catname, val in db_variations.items() if val["activate"]]
        self.n_sys_cat = len(self.systematic_catnames)
        self.systematic_catlabels = [""] * self.n_sys_cat
        self.systematic_catgroups = [""] * self.n_sys_cat
        self.systematic_varnames = [None] * self.n_sys_cat
        self.systematic_varlabels = [None] * self.n_sys_cat
        self.systematic_variations = [0] * self.n_sys_cat
        self.systematic_correlation = [None] * self.n_sys_cat
        self.systematic_rms = [False] * self.n_sys_cat
        self.systematic_symmetrise = [False] * self.n_sys_cat
        self.systematic_rms_both_sides = [False] * self.n_sys_cat
        self.powheg_nonprompt_varnames = []
        self.powheg_nonprompt_varlabels = []
        self.lvar2_binmin_gen_sys = None
        self.lvar2_binmax_gen_sys = None
        self.lvar2_binmin_reco_sys = None
        self.lvar2_binmax_reco_sys = None
        for c, catname in enumerate(self.systematic_catnames):
            self.systematic_catlabels[c] = db_variations[catname]["label"]
            self.systematic_catgroups[c] = db_variations[catname].get("group", self.systematic_catlabels[c])
            self.systematic_varnames[c] = []
            self.systematic_varlabels[c] = []
            for varname, val in db_variations[catname]["variations"].items():
                if catname == "binning" and varname == "pt_jet":
                    self.lvar2_binmin_gen_sys = val["diffs"]["analysis"][self.typean]["sel_binmin2_gen"]
                    self.lvar2_binmax_gen_sys = val["diffs"]["analysis"][self.typean]["sel_binmax2_gen"]
                    for i_var, list_pt in enumerate(self.lvar2_binmin_gen_sys):
                        if list_pt == "#":
                            self.lvar2_binmin_gen_sys[i_var] = self.lvar2_binmin_gen
                    for i_var, list_pt in enumerate(self.lvar2_binmax_gen_sys):
                        if list_pt == "#":
                            self.lvar2_binmax_gen_sys[i_var] = self.lvar2_binmax_gen
                    self.lvar2_binmin_reco_sys = val["diffs"]["analysis"][self.typean]["sel_binmin2_reco"]
                    self.lvar2_binmax_reco_sys = val["diffs"]["analysis"][self.typean]["sel_binmax2_reco"]
                    for i_var, list_pt in enumerate(self.lvar2_binmin_reco_sys):
                        if list_pt == "#":
                            self.lvar2_binmin_reco_sys[i_var] = self.lvar2_binmin_reco
                    for i_var, list_pt in enumerate(self.lvar2_binmax_reco_sys):
                        if list_pt == "#":
                            self.lvar2_binmax_reco_sys[i_var] = self.lvar2_binmax_reco
                n_var = len(val["activate"])
                for a, act in enumerate(val["activate"]):
                    if act:
                        varname_i = format_varname(varname, a, n_var)
                        varlabel_i = format_varlabel(val["label"], a, n_var)
                        self.systematic_varnames[c].append(varname_i)
                        self.systematic_varlabels[c].append(varlabel_i)
                        if catname == "feeddown":
                            self.powheg_nonprompt_varnames.append(varname_i)
                            self.powheg_nonprompt_varlabels.append(varlabel_i)
            self.systematic_variations[c] = len(self.systematic_varnames[c])
            self.systematic_correlation[c] = db_variations[catname]["correlation"]
            self.systematic_rms[c] = db_variations[catname]["rms"]
            self.systematic_symmetrise[c] = db_variations[catname]["symmetrise"]
            self.systematic_rms_both_sides[c] = db_variations[catname]["rms_both_sides"]
        self.systematic_catgroups_list = list(dict.fromkeys(self.systematic_catgroups))
        self.n_sys_gr = len(self.systematic_catgroups_list)
        # self.inclusive_unc = datap["analysis"][self.typean]["inclusive_unc"]
        # self.use_inclusive_systematics = datap["analysis"][self.typean]["use_inclusive_systematics"]
        # print("Use inclusive systematics:", self.use_inclusive_systematics)
        # self.do_check_signif = datap["analysis"][self.typean]["signif_check"]
        # self.signif_threshold = datap["analysis"][self.typean]["signif_thresh"]
        # print("Check if significance >", self.signif_threshold, "for systematic fits:", self.do_check_signif)

        # output directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["resultsallp"]

        # input files

        file_result_name = datap["files_names"]["resultfilename"]
        self.file_unfold = os.path.join(self.d_resultsallpdata, file_result_name)
        file_eff_name = datap["files_names"]["efffilename"]
        self.file_efficiency = os.path.join(self.d_resultsallpmc, file_eff_name)
        self.file_efficiency = self.file_unfold

        # official figures
        self.shape = typean[len("jet_") :]
        self.size_can = [800, 800]
        self.offsets_axes = [0.8, 1.1]
        self.margins_can = [0.1, 0.13, 0.05, 0.03]
        self.fontsize = 0.035
        self.opt_leg_g = "FP"  # for systematic uncertanties in the legend
        self.opt_plot_g = "2"
        self.x_latex = 0.18
        self.y_latex_top = 0.88
        self.y_step = 0.05
        # axes titles
        self.title_x = self.v_varshape_latex
        self.title_y = "(1/#it{N}_{jet}) d#it{N}/d%s" % self.v_varshape_latex
        self.title_full = ";%s;%s" % (self.title_x, self.title_y)
        self.title_full_ratio = ";%s;data/MC: ratio of %s" % (self.title_x, self.title_y)
        # text
        # self.text_alice = "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV"
        self.text_alice = "#bf{ALICE}, pp, #sqrt{#it{s}} = 13 TeV"
        self.text_jets = "%s-tagged charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.p_latexnhadron
        self.text_jets_ratio = "#Lambda_{c}^{+}, D^{0} -tagged charged jets, anti-#it{k}_{T}, #it{R} = 0.4"
        self.text_ptjet = "%g #leq %s < %g GeV/#it{c}, |#it{#eta}_{jet ch}| < 0.5"
        self.text_pth = "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, |#it{y}_{%s}| < 0.8"
        self.text_sd = "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        self.text_acc_h = "|#it{y}| < 0.8"
        self.text_powheg = "POWHEG + PYTHIA 6 + EvtGen"


    def jetsystematics(self):
        string_default = "default/default"
        if string_default not in self.d_resultsallpdata:
            self.logger.critical("Not a default database! Cannot run systematics.")

        debug = True
        if debug:
            print("Categories: ", self.systematic_catnames)
            print("Category labels: ", self.systematic_catlabels)
            print("Category Groups: ", self.systematic_catgroups)
            print("Category Groups unique: ", self.systematic_catgroups_list, self.n_sys_gr)
            print("Numbers of variations: ", self.systematic_variations)
            print("Variations: ", self.systematic_varnames)
            print("Variation labels: ", self.systematic_varlabels)
            print("Correlation: ", self.systematic_correlation)
            print("RMS: ", self.systematic_rms)
            print("Symmetrisation: ", self.systematic_symmetrise)
            print("RMS both sides: ", self.systematic_rms_both_sides)
            print("Feed-down variations: ", self.powheg_nonprompt_varnames)
            print("Jet pT rec min variations: ", self.lvar2_binmin_reco_sys)
            print("Jet pT rec max variations: ", self.lvar2_binmax_reco_sys)
            print("Jet pT gen min variations: ", self.lvar2_binmin_gen_sys)
            print("Jet pT gen max variations: ", self.lvar2_binmax_gen_sys)

        path_def = self.file_unfold
        input_file_default = TFile.Open(path_def)
        if not input_file_default:
            self.logger.critical(make_message_notfound(path_def))
        path_eff = self.file_efficiency
        eff_file_default = TFile.Open(path_eff)
        file_sys_out = TFile.Open("%s/systematics_results.root" % self.d_resultsallpdata, "recreate")

        # get the default (central value) result histograms

        var = "zpar"
        method = "sidesub"
        input_histograms_default = []
        # eff_default = []
        for ibin2 in range(self.p_nbin2_gen):
            name_hist_unfold_2d = f'h_ptjet-{var}_{method}_unfolded_data_0'
            if not (hist_unfold := input_file_default.Get(name_hist_unfold_2d)):
                self.logger.critical(make_message_notfound(name_hist_unfold_2d, eff_file_default))
            axis_jetpt = get_axis(hist_unfold, 0)
            jetptrange = (axis_jetpt.GetBinLowEdge(ibin2+1), axis_jetpt.GetBinUpEdge(ibin2+1))
            name_his = f'h_{var}_{method}_unfolded_data_jetpt-{jetptrange[0]}-{jetptrange[1]}_sel'
            input_histograms_default.append(input_file_default.Get(name_his))
            if not input_histograms_default[ibin2]:
                self.logger.critical(make_message_notfound(name_his, path_def))
            print(f"Default histogram ({jetptrange[0]} to {jetptrange[1]})")
            print_histogram(input_histograms_default[ibin2])
            # name_eff = "eff_mult%d" % ibin2
            # eff_default.append(eff_file_default.Get(name_eff))

        # get the files containing result variations

        input_files_sys = []
        input_files_eff = []
        for sys_cat in range(self.n_sys_cat):
            input_files_sysvar = []
            input_files_sysvar_eff = []
            for sys_var, varname in enumerate(self.systematic_varnames[sys_cat]):
                path = path_def.replace(string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                input_files_sysvar.append(TFile.Open(path))
                eff_file = path_eff.replace(string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                input_files_sysvar_eff.append(TFile.Open(eff_file))
                if not input_files_sysvar[sys_var]:
                    self.logger.critical(make_message_notfound(path))
                if not input_files_sysvar_eff[sys_var]:
                    self.logger.critical(make_message_notfound(eff_file))
            input_files_sys.append(input_files_sysvar)
            input_files_eff.append(input_files_sysvar_eff)

        # get the variation result histograms

        input_histograms_sys = []
        input_histograms_sys_eff = []
        for ibin2 in range(self.p_nbin2_gen):  # pylint: disable=:too-many-nested-blocks
            name_eff = "eff_mult%d" % ibin2
            input_histograms_syscat = []
            input_histograms_syscat_eff = []
            for sys_cat in range(self.n_sys_cat):
                input_histograms_syscatvar = []
                input_histograms_eff = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    print(
                        "Variation: %s, %s"
                        % (self.systematic_catnames[sys_cat], self.systematic_varnames[sys_cat][sys_var])
                    )
                    name_hist_unfold_2d = f'h_ptjet-{var}_{method}_unfolded_data_0'
                    if not (hist_unfold := input_files_sys[sys_cat][sys_var].Get(name_hist_unfold_2d)):
                        self.logger.critical(make_message_notfound(name_hist_unfold_2d, eff_file))
                    axis_jetpt = get_axis(hist_unfold, 0)
                    # signif_check = True
                    string_catvar = self.systematic_catnames[sys_cat] + "/" + self.systematic_varnames[sys_cat][sys_var]
                    if string_catvar.startswith("binning/pt_jet"):
                        suffix_jetpt = "%g_%g" % (
                            self.lvar2_binmin_reco_sys[sys_var][ibin2],
                            self.lvar2_binmax_reco_sys[sys_var][ibin2],
                        )
                    else:
                        suffix_jetpt = "%g_%g" % (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                    # if self.do_check_signif:  # TODO: signifcance check
                    #     for ipt in range(self.p_nptfinbins):
                    #         pass
                    # FIXME exception for different jet pt binning, pylint: disable=fixme
                    if string_catvar.startswith("binning/pt_jet"):
                        name_his = "unfolded_z_sel_%s_%.2f_%.2f" % (
                            self.v_var2_binning,
                            self.lvar2_binmin_gen_sys[sys_var][ibin2],
                            self.lvar2_binmax_gen_sys[sys_var][ibin2],
                        )
                    else:
                        jetptrange = (axis_jetpt.GetBinLowEdge(ibin2+1), axis_jetpt.GetBinUpEdge(ibin2+1))
                        name_his = f'h_{var}_{method}_unfolded_data_jetpt-{jetptrange[0]}-{jetptrange[1]}_sel'
                    sys_var_histo = input_files_sys[sys_cat][sys_var].Get(name_his)
                    # sys_var_histo_eff = input_files_eff[sys_cat][sys_var].Get(name_eff)
                    path_file = path_def.replace(string_default, string_catvar)
                    path_eff_file = path_eff.replace(string_default, string_catvar)
                    # if not signif_check:
                    #     print("BAD FIT in Variation: %s, %s" % (self.systematic_catnames[sys_cat], self.systematic_varnames[sys_cat][sys_var]))
                    #     for idr in range(len(self.lvarshape_binmin_gen)):
                    #         sys_var_histo.SetBinContent(idr+1, 0)
                    #         sys_var_histo_eff.SetBinContent(idr+1, 0)
                    input_histograms_syscatvar.append(sys_var_histo)
                    # input_histograms_eff.append(sys_var_histo_eff)
                    if not input_histograms_syscatvar[sys_var]:
                        self.logger.critical(make_message_notfound(name_his, path_file))
                    print_histogram(sys_var_histo)
                    # if not input_histograms_eff[sys_var]:
                    #     self.logger.critical(make_message_notfound(name_eff, path_eff_file))
                    if debug:
                        print(
                            "Variation: %s, %s: got histogram %s from file %s"
                            % (
                                self.systematic_catnames[sys_cat],
                                self.systematic_varnames[sys_cat][sys_var],
                                name_his,
                                path_file,
                            )
                        )
                        # print(
                        #     "Variation: %s, %s: got efficiency histogram %s from file %s"
                        #     % (
                        #         self.systematic_catnames[sys_cat],
                        #         self.systematic_varnames[sys_cat][sys_var],
                        #         name_eff,
                        #         path_eff_file,
                        #     )
                        # )
                    # input_histograms_syscatvar[sys_var].Scale(1.0, "width") #remove these later and put normalisation directly in systematics
                input_histograms_syscat.append(input_histograms_syscatvar)
                # input_histograms_syscat_eff.append(input_histograms_eff)
            input_histograms_sys.append(input_histograms_syscat)
            # input_histograms_sys_eff.append(input_histograms_syscat_eff)

        # plot the variations

        print("Categories: %d", self.n_sys_cat)
        # self.logger.info("Categories: %d", self.n_sys_cat)

        for ibin2 in range(self.p_nbin2_gen):
            # plot all the variations together
            suffix = "%s_%g_%g" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            nsys = 0
            csysvar = TCanvas("csysvar_%s" % suffix, "systematic variations" + suffix)
            setup_canvas(csysvar)
            leg_sysvar = TLegend(0.75, 0.15, 0.95, 0.85, "variation")
            setup_legend(leg_sysvar)
            leg_sysvar.AddEntry(input_histograms_default[ibin2], "default", "P")
            setup_histogram(input_histograms_default[ibin2])
            l_his_all = []
            for l_cat in input_histograms_sys[ibin2]:
                for his_var in l_cat:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
            l_his_all.append(input_histograms_default[ibin2])
            y_min, y_max = get_y_window_his(l_his_all)
            y_margin_up = 0.15
            y_margin_down = 0.05
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(
                *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            )
            # input_histograms_default[ibin2].GetXaxis().SetRangeUser(
            #     round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2)
            # )
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            print_histogram(input_histograms_default[ibin2])

            self.logger.info("Categories: %d", self.n_sys_cat)
            print("Categories: %d" % self.n_sys_cat)

            for sys_cat in range(self.n_sys_cat):
                self.logger.info("Category: %s", self.systematic_catlabels[sys_cat])
                print("Category: %s" % self.systematic_catlabels[sys_cat])
                for sys_var in range(self.systematic_variations[sys_cat]):
                    self.logger.info("Variation: %s", self.systematic_varlabels[sys_cat][sys_var])
                    print("Variation: %s" % self.systematic_varlabels[sys_cat][sys_var])
                    leg_sysvar.AddEntry(
                        input_histograms_sys[ibin2][sys_cat][sys_var],
                        ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])),
                        "P",
                    )
                    self.logger.info("Adding label %s", ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])))
                    print("Adding label %s" % ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])))
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var], get_colour(nsys + 1))
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1

            latex = TLatex(
                0.15,
                0.82,
                "%g #leq %s < %g GeV/#it{c}"
                % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]),
            )
            draw_latex(latex)
            # leg_sysvar.Draw("same")
            csysvar.SaveAs("%s/sys_var_all_%s.eps" % (self.d_resultsallpdata, suffix))

            continue

            # plot the variations for each category separately

            for sys_cat in range(self.n_sys_cat):
                suffix2 = self.systematic_catnames[sys_cat]
                nsys = 0
                csysvar_each = TCanvas("csysvar_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix)
                setup_canvas(csysvar_each)
                csysvar_each.SetRightMargin(0.25)
                leg_sysvar_each = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(input_histograms_default[ibin2], "default", "P")
                setup_histogram(input_histograms_default[ibin2])
                l_his_all = []
                for his_var in input_histograms_sys[ibin2][sys_cat]:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                l_his_all.append(input_histograms_default[ibin2])
                y_min, y_max = get_y_window_his(l_his_all)
                y_margin_up = 0.15
                y_margin_down = 0.05
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        input_histograms_default[ibin2].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        input_histograms_default[ibin2].GetXaxis().SetRangeUser(
                            round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2)
                        )
                        input_histograms_default[ibin2].SetTitle("")
                        input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
                        input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                        input_histograms_default[ibin2].Draw()
                    leg_sysvar_each.AddEntry(
                        input_histograms_sys[ibin2][sys_cat][sys_var], self.systematic_varlabels[sys_cat][sys_var], "P"
                    )
                    if input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(1) != 0:
                        setup_histogram(
                            input_histograms_sys[ibin2][sys_cat][sys_var], get_colour(nsys + 1), get_marker(nsys + 1)
                        )
                        input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]),
                )
                draw_latex(latex)
                leg_sysvar_each.Draw("same")
                csysvar_each.SaveAs("%s/sys_var_%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))

                nsys = 0
                csysvar_ratio = TCanvas(
                    "csysvar_ratio_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_ratio)
                csysvar_ratio.SetRightMargin(0.25)
                leg_sysvar_ratio = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_ratio)
                histo_ratio = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    default_his = input_histograms_default[ibin2].Clone("default_his")
                    var_his = input_histograms_sys[ibin2][sys_cat][sys_var].Clone("var_his")
                    var_his.Divide(default_his)
                    histo_ratio.append(var_his)
                l_his_all = []
                for his_var in histo_ratio:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                y_min, y_max = get_y_window_his(l_his_all)
                y_margin_up = 0.15
                y_margin_down = 0.05

                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        histo_ratio[sys_var].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        histo_ratio[sys_var].SetTitle("")
                        histo_ratio[sys_var].SetXTitle(self.v_varshape_latex)
                        histo_ratio[sys_var].SetYTitle("variation/default")
                        histo_ratio[sys_var].Draw()
                    if histo_ratio[sys_var].GetBinContent(1) > 0:
                        leg_sysvar_ratio.AddEntry(
                            histo_ratio[sys_var], self.systematic_varlabels[sys_cat][sys_var], "P"
                        )
                        setup_histogram(histo_ratio[sys_var], get_colour(nsys + 1), get_marker(nsys + 1))
                        histo_ratio[sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]),
                )
                draw_latex(latex)
                # line = TLine(self.lvarshape_binmin_reco[0], 1, self.lvarshape_binmax_reco[-1], 1)
                # line.SetLineColor(1)
                # line.Draw()
                leg_sysvar_ratio.Draw("same")
                csysvar_ratio.SaveAs("%s/sys_var_ratio_%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))

                csysvar_eff = TCanvas("csysvar_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix)
                setup_canvas(csysvar_eff)
                csysvar_eff.SetRightMargin(0.25)
                leg_sysvar_eff = TLegend(
                    0.77,
                    0.2,
                    0.95,
                    0.85,
                    self.systematic_catlabels[sys_cat],
                )  # Rg
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(eff_default[ibin2], "default", "P")
                setup_histogram(eff_default[ibin2])
                l_his_all = []
                for his_var in input_histograms_sys_eff[ibin2][sys_cat]:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                l_his_all.append(eff_default[ibin2])
                y_min, y_max = get_y_window_his(l_his_all)
                y_margin_up = 0.15
                y_margin_down = 0.05
                nsys = 0
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        eff_default[ibin2].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        eff_default[ibin2].SetTitle("")
                        eff_default[ibin2].SetXTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.p_latexnhadron)
                        eff_default[ibin2].SetYTitle("prompt %s-jet efficiency" % self.p_latexnhadron)
                        eff_default[ibin2].Draw()
                    if input_histograms_sys_eff[ibin2][sys_cat][sys_var].GetBinContent(1) > 0:
                        leg_sysvar_eff.AddEntry(
                            input_histograms_sys_eff[ibin2][sys_cat][sys_var],
                            self.systematic_varlabels[sys_cat][sys_var],
                            "P",
                        )
                        setup_histogram(
                            input_histograms_sys_eff[ibin2][sys_cat][sys_var],
                            get_colour(nsys + 1),
                            get_marker(nsys + 1),
                        )
                        input_histograms_sys_eff[ibin2][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]),
                )
                draw_latex(latex)
                leg_sysvar_eff.Draw("same")
                csysvar_eff.SaveAs("%s/sys_var_eff_%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))

                csysvar_bin = TCanvas(
                    "csysvar_bin_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_bin)
                csysvar_bin.SetRightMargin(0.25)
                var_histos = []
                labels = []
                sys_array = []
                def_array = []
                for r_val in enumerate(self.lvarshape_binmin_gen):
                    var_histo = TH1F(
                        "varhisto_%s_%s" % (suffix2, suffix),
                        "varhisto",
                        self.systematic_variations[sys_cat],
                        0,
                        self.systematic_variations[sys_cat],
                    )
                    def_bin = input_histograms_default[ibin2].GetBinContent(r_val + 1)
                    def_array.append(def_bin)
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        sys_bin = input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(r_val + 1)
                        if sys_bin != 0:
                            sys_bin = sys_bin / def_bin
                        sys_array.append(sys_bin)
                        var_histo.SetBinContent(sys_var + 1, sys_bin)
                    label_varhisto = "%s = %s #minus %s" % (
                        self.v_varshape_latex,
                        self.lvarshape_binmin_gen[r_val],
                        self.lvarshape_binmax_gen[r_val],
                    )
                    labels.append(label_varhisto)
                    var_histos.append(var_histo)
                leg_varhisto = TLegend(0.77, 0.2, 0.95, 0.85)  # Rg
                setup_legend(leg_varhisto)

                nsys = 0
                csysvar_eff_ratio = TCanvas(
                    "csysvar_eff_ratio_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_eff_ratio)
                csysvar_eff_ratio.SetRightMargin(0.25)
                leg_sysvar_eff_ratio = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_eff_ratio)
                histo_ratio = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    default_his = eff_default[ibin2].Clone("default_his")
                    var_his = input_histograms_sys_eff[ibin2][sys_cat][sys_var].Clone("var_his")
                    var_his.Divide(default_his)
                    histo_ratio.append(var_his)
                l_his_all = []
                for his_var in histo_ratio:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                y_min, y_max = get_y_window_his(l_his_all)
                y_margin_up = 0.05
                y_margin_down = 0.05
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        histo_ratio[sys_var].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        histo_ratio[sys_var].SetTitle("")
                        histo_ratio[sys_var].SetXTitle("#it{p}_{T}")
                        histo_ratio[sys_var].SetYTitle("efficiency (variation/default)")
                        histo_ratio[sys_var].Draw()
                    if histo_ratio[sys_var].GetBinContent(1) > 0:
                        leg_sysvar_eff_ratio.AddEntry(
                            histo_ratio[sys_var], self.systematic_varlabels[sys_cat][sys_var], "P"
                        )
                        setup_histogram(histo_ratio[sys_var], get_colour(nsys + 1), get_marker(nsys + 1))
                        histo_ratio[sys_var].Draw("same")
                    else:
                        histo_ratio[sys_var].Reset()
                        setup_histogram(histo_ratio[sys_var], 0, get_marker(nsys + 1))
                    nsys = nsys + 1
                # line = TLine(self.lvarshape_binmin_reco[0], 1, self.lvarshape_binmax_reco[-1], 1)
                # line.SetLineColor(1)
                # line.Draw()
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]),
                )
                draw_latex(latex)
                leg_sysvar_eff_ratio.Draw("same")
                csysvar_eff_ratio.SaveAs("%s/sys_var_eff_ratio_%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))

        return


        # calculate the systematic uncertainties

        sys_up = []  # list of absolute upward uncertainties for all categories, shape bins, pt_jet bins
        sys_down = []  # list of absolute downward uncertainties for all categories, shape bins, pt_jet bins
        sys_up_full = []  # list of combined absolute upward uncertainties for all shape bins, pt_jet bins
        sys_down_full = []  # list of combined absolute downward uncertainties for all shape bins, pt_jet bins
        for ibin2 in range(self.p_nbin2_gen):  # pylint: disable=too-many-nested-blocks
            sys_up_jetpt = []  # list of absolute upward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_down_jetpt = []  # list of absolute downward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_up_z_full = []  # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            sys_down_z_full = []  # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            for ibinshape in range(self.p_nbinshape_gen):
                sys_up_z = []  # list of absolute upward uncertainties for all categories in a given (pt_jet, shape) bin
                sys_down_z = []  # list of absolute downward uncertainties for all categories in a given (pt_jet, shape) bin
                error_full_up = 0  # combined absolute upward uncertainty in a given (pt_jet, shape) bin
                error_full_down = 0  # combined absolute downward uncertainty in a given (pt_jet, shape) bin
                for sys_cat in range(self.n_sys_cat):
                    error_var_up = 0  # absolute upward uncertainty for a given category in a given (pt_jet, shape) bin
                    error_var_down = (
                        0  # absolute downward uncertainty for a given category in a given (pt_jet, shape) bin
                    )
                    count_sys_up = 0
                    count_sys_down = 0
                    error = 0
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        out_sys = False
                        # FIXME exception for the untagged bin pylint: disable=fixme
                        bin_first = 1
                        # bin_first = 2 if "untagged" in self.systematic_varlabels[sys_cat][sys_var] else 1
                        # FIXME exception for the untagged bin pylint: disable=fixme
                        if input_histograms_sys[ibin2][sys_cat][sys_var].Integral() == 0:
                            error = 0
                            out_sys = True
                        else:
                            error = input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(
                                ibinshape + bin_first
                            ) - input_histograms_default[ibin2].GetBinContent(ibinshape + 1)
                        if error >= 0:
                            if self.systematic_rms[sys_cat] is True:
                                error_var_up += error * error
                                if not out_sys:
                                    count_sys_up = count_sys_up + 1
                            else:
                                if error > error_var_up:
                                    error_var_up = error
                        else:
                            if self.systematic_rms[sys_cat] is True:
                                if self.systematic_rms_both_sides[sys_cat] is True:
                                    error_var_up += error * error
                                    if not out_sys:
                                        count_sys_up = count_sys_up + 1
                                else:
                                    error_var_down += error * error
                                    if not out_sys:
                                        count_sys_down = count_sys_down + 1
                            else:
                                if abs(error) > error_var_down:
                                    error_var_down = abs(error)
                    if self.systematic_rms[sys_cat] is True:
                        if count_sys_up != 0:
                            error_var_up = error_var_up / count_sys_up
                        else:
                            error_var_up = 0.0
                        error_var_up = sqrt(error_var_up)
                        if count_sys_down != 0:
                            error_var_down = error_var_down / count_sys_down
                        else:
                            error_var_down = 0.0
                        if self.systematic_rms_both_sides[sys_cat] is True:
                            error_var_down = error_var_up
                        else:
                            error_var_down = sqrt(error_var_down)
                    if self.systematic_symmetrise[sys_cat] is True:
                        if error_var_up > error_var_down:
                            error_var_down = error_var_up
                        else:
                            error_var_up = error_var_down
                    error_full_up += error_var_up * error_var_up
                    error_full_down += error_var_down * error_var_down
                    sys_up_z.append(error_var_up)
                    sys_down_z.append(error_var_down)
                error_full_up = sqrt(error_full_up)
                sys_up_z_full.append(error_full_up)
                error_full_down = sqrt(error_full_down)
                sys_down_z_full.append(error_full_down)
                sys_up_jetpt.append(sys_up_z)
                sys_down_jetpt.append(sys_down_z)
            sys_up_full.append(sys_up_z_full)
            sys_down_full.append(sys_down_z_full)
            sys_up.append(sys_up_jetpt)
            sys_down.append(sys_down_jetpt)

        # create graphs to plot the uncertainties

        tgsys = []  # list of graphs with combined absolute uncertainties for all pt_jet bins
        tgsys_cat = []  # list of graphs with relative uncertainties for all categories, pt_jet bins
        full_unc_up = []  # list of relative uncertainties for all categories, pt_jet bins
        full_unc_down = []  # list of relative uncertainties for all categories, pt_jet bins
        for ibin2 in range(self.p_nbin2_gen):
            # combined uncertainties

            shapebins_centres = []
            shapebins_contents = []
            shapebins_widths_up = []
            shapebins_widths_down = []
            shapebins_error_up = []
            shapebins_error_down = []
            rel_unc_up = []
            rel_unc_down = []
            unc_rel_min = 100.0
            unc_rel_max = 0.0
            for ibinshape in range(self.p_nbinshape_gen):
                shapebins_centres.append(input_histograms_default[ibin2].GetBinCenter(ibinshape + 1))
                val = input_histograms_default[ibin2].GetBinContent(ibinshape + 1)
                shapebins_contents.append(val)
                shapebins_widths_up.append(input_histograms_default[ibin2].GetBinWidth(ibinshape + 1) * 0.5)
                shapebins_widths_down.append(input_histograms_default[ibin2].GetBinWidth(ibinshape + 1) * 0.5)
                err_up = sys_up_full[ibin2][ibinshape]
                err_down = sys_down_full[ibin2][ibinshape]
                shapebins_error_up.append(err_up)
                shapebins_error_down.append(err_down)
                if val > 0:
                    unc_rel_up = err_up / val
                    unc_rel_down = err_down / val
                    unc_rel_min = min(unc_rel_min, unc_rel_up, unc_rel_down)
                    unc_rel_max = max(unc_rel_max, unc_rel_up, unc_rel_down)
                    rel_unc_up.append(unc_rel_up)
                    rel_unc_down.append(unc_rel_down)
                    print(
                        "total rel. syst. unc.: ",
                        self.lvar2_binmin_gen[ibin2],
                        " ",
                        self.lvar2_binmax_gen[ibin2],
                        " ",
                        self.lvarshape_binmin_gen[ibinshape],
                        " ",
                        self.lvarshape_binmax_gen[ibinshape],
                        " ",
                        unc_rel_up,
                        " ",
                        unc_rel_down,
                    )
                else:
                    rel_unc_up.append(0.0)
                    rel_unc_down.append(0.0)
            print(f"total rel. syst. unc. (%): min. {(100. * unc_rel_min):.2g}, max. {(100. * unc_rel_max):.2g}")
            shapebins_centres_array = array("d", shapebins_centres)
            shapebins_contents_array = array("d", shapebins_contents)
            shapebins_widths_up_array = array("d", shapebins_widths_up)
            shapebins_widths_down_array = array("d", shapebins_widths_down)
            shapebins_error_up_array = array("d", shapebins_error_up)
            shapebins_error_down_array = array("d", shapebins_error_down)
            tgsys.append(
                TGraphAsymmErrors(
                    self.p_nbinshape_gen,
                    shapebins_centres_array,
                    shapebins_contents_array,
                    shapebins_widths_down_array,
                    shapebins_widths_up_array,
                    shapebins_error_down_array,
                    shapebins_error_up_array,
                )
            )
            full_unc_up.append(rel_unc_up)
            full_unc_down.append(rel_unc_down)
            # relative uncertainties per category

            tgsys_cat_z = []  # list of graphs with relative uncertainties for all categories in a given pt_jet bin
            for sys_cat in range(self.n_sys_cat):
                shapebins_contents_cat = []
                shapebins_error_up_cat = []
                shapebins_error_down_cat = []
                for ibinshape in range(self.p_nbinshape_gen):
                    shapebins_contents_cat.append(0)
                    if input_histograms_default[ibin2].GetBinContent(ibinshape + 1) == 0:
                        print("WARNING!!! Input histogram at bin", ibin2, " equal 0, skip", suffix)
                        continue
                    shapebins_error_up_cat.append(
                        sys_up[ibin2][ibinshape][sys_cat] / input_histograms_default[ibin2].GetBinContent(ibinshape + 1)
                    )
                    shapebins_error_down_cat.append(
                        sys_down[ibin2][ibinshape][sys_cat]
                        / input_histograms_default[ibin2].GetBinContent(ibinshape + 1)
                    )
                shapebins_contents_cat_array = array("d", shapebins_contents_cat)
                shapebins_error_up_cat_array = array("d", shapebins_error_up_cat)
                shapebins_error_down_cat_array = array("d", shapebins_error_down_cat)
                tgsys_cat_z.append(
                    TGraphAsymmErrors(
                        self.p_nbinshape_gen,
                        shapebins_centres_array,
                        shapebins_contents_cat_array,
                        shapebins_widths_down_array,
                        shapebins_widths_up_array,
                        shapebins_error_down_cat_array,
                        shapebins_error_up_cat_array,
                    )
                )
            tgsys_cat.append(tgsys_cat_z)

        # combine uncertainties from categories into groups

        tgsys_gr = []  # list of graphs with relative uncertainties for all groups, pt_jet bins
        for ibin2 in range(self.p_nbin2_gen):
            tgsys_gr_z = []  # list of graphs with relative uncertainties for all groups in a given pt_jet bin
            for gr in self.systematic_catgroups_list:
                tgsys_gr_z_cat = []  # lists of graphs with relative uncertainties for categories in a given group in a given pt_jet bin
                for sys_cat, cat in enumerate(self.systematic_catlabels):
                    if self.systematic_catgroups[sys_cat] == gr:
                        print(f"Group {gr}: Adding category {cat}")
                        tgsys_gr_z_cat.append(tgsys_cat[ibin2][sys_cat])
                tgsys_gr_z.append(combine_graphs(tgsys_gr_z_cat))
            tgsys_gr.append(tgsys_gr_z)

        # write the combined systematic uncertainties in a file
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            file_sys_out.cd()
            tgsys[ibin2].Write("tgsys_%s" % suffix)
            unc_hist_up = TH1F(
                "unc_hist_up_%s" % suffix,
                "",
                self.p_nbinshape_gen,
                self.lvarshape_binmin_gen[0],
                self.lvarshape_binmax_gen[-1],
            )
            unc_hist_down = TH1F(
                "unc_hist_down_%s" % suffix,
                "",
                self.p_nbinshape_gen,
                self.lvarshape_binmin_gen[0],
                self.lvarshape_binmax_gen[-1],
            )
            for ibinshape in range(self.p_nbinshape_gen):
                unc_hist_up.SetBinContent(ibinshape + 1, full_unc_up[ibin2][ibinshape])
                unc_hist_down.SetBinContent(ibinshape + 1, full_unc_down[ibin2][ibinshape])
            unc_hist_up.Write()
            unc_hist_down.Write()

        # relative statistical uncertainty of the central values
        h_default_stat_err = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            h_default_stat_err.append(input_histograms_default[ibin2].Clone("h_default_stat_err" + suffix))
            for i in range(h_default_stat_err[ibin2].GetNbinsX()):
                if input_histograms_default[ibin2].GetBinContent(ibinshape + 1) == 0:
                    print("WARNING!!! Input histogram at bin", ibin2, " equal 0, skip", suffix)
                    h_default_stat_err[ibin2].SetBinContent(i + 1, 0)
                    h_default_stat_err[ibin2].SetBinError(i + 1, 0)
                else:
                    h_default_stat_err[ibin2].SetBinContent(i + 1, 0)
                    h_default_stat_err[ibin2].SetBinError(
                        i + 1,
                        input_histograms_default[ibin2].GetBinError(i + 1)
                        / input_histograms_default[ibin2].GetBinContent(i + 1),
                    )

        for ibin2 in range(self.p_nbin2_gen):
            # plot the results with systematic uncertainties

            suffix = "%s_%g_%g" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cfinalwsys = TCanvas("cfinalwsys " + suffix, "final result with systematic uncertainties" + suffix)
            setup_canvas(cfinalwsys)
            leg_finalwsys = TLegend(0.7, 0.78, 0.85, 0.88)
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_histograms_default[ibin2], "data, pp, #sqrt{#it{s}} = 13 TeV", "P")
            setup_histogram(input_histograms_default[ibin2], get_colour(0, 0))
            y_min_g, y_max_g = get_y_window_gr([tgsys[ibin2]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.4
            y_margin_down = 0.05
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(
                *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            )
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(
                round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2)
            )
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw("AXIS")
            # input_histograms_default[ibin2].Draw("")
            setup_tgraph(tgsys[ibin2], get_colour(7, 0))
            tgsys[ibin2].Draw("5")
            input_histograms_default[ibin2].Draw("SAME")
            leg_finalwsys.AddEntry(tgsys[ibin2], "syst. unc.", "F")
            input_histograms_default[ibin2].Draw("AXISSAME")
            # PREL latex = TLatex(0.15, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(
                0.15,
                0.72,
                "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5"
                % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]),
            )
            draw_latex(latex2)
            latex3 = TLatex(
                0.15,
                0.67,
                "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8"
                % (
                    self.lpt_finbinmin[0],
                    self.p_latexnhadron,
                    min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2]),
                    self.p_latexnhadron,
                ),
            )
            draw_latex(latex3)
            leg_finalwsys.Draw("same")
            latex_SD = TLatex(0.15, 0.62, "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)")
            draw_latex(latex_SD)
            cfinalwsys.SaveAs("%s/%s_final_wsys_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix))

            # plot the relative systematic uncertainties for all categories together

            # preliminary figure
            i_shape = 0 if self.shape == "zg" else 1 if self.shape == "rg" else 2

            crelativesys = TCanvas("crelativesys " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys)
            crelativesys.SetCanvasSize(900, 800)
            crelativesys.SetBottomMargin(self.margins_can[0])
            crelativesys.SetLeftMargin(self.margins_can[1])
            crelativesys.SetTopMargin(self.margins_can[2])
            crelativesys.SetRightMargin(self.margins_can[3])
            crelativesys.SetRightMargin(0.25)
            leg_relativesys = TLegend(0.77, 0.2, 0.95, 0.85)
            setup_legend(leg_relativesys, textsize=self.fontsize)
            y_min_g, y_max_g = get_y_window_gr(tgsys_cat[ibin2])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            list_y_margin_up = [0.2, 0.35, 0.2]
            y_margin_up = list_y_margin_up[i_shape]
            y_margin_down = 0.05
            setup_histogram(h_default_stat_err[ibin2])
            h_default_stat_err[ibin2].SetMarkerStyle(0)
            h_default_stat_err[ibin2].SetMarkerSize(0)
            leg_relativesys.AddEntry(h_default_stat_err[ibin2], "stat. unc.", "E")
            for sys_cat in range(self.n_sys_cat):
                setup_tgraph(tgsys_cat[ibin2][sys_cat], get_colour(sys_cat + 1, 0))
                tgsys_cat[ibin2][sys_cat].SetTitle("")
                tgsys_cat[ibin2][sys_cat].SetLineWidth(3)
                tgsys_cat[ibin2][sys_cat].SetFillStyle(0)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetRangeUser(
                    *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                )
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetLimits(
                    round(self.lvarshape_binmin_gen[0 if self.shape == "nsd" else 1], 2),
                    round(self.lvarshape_binmax_gen[-1], 2),
                )
                if self.shape == "nsd":
                    tgsys_cat[ibin2][sys_cat].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tgsys_cat[ibin2][sys_cat], 0.2)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitle(self.v_varshape_latex)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitle("relative systematic uncertainty")
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitleOffset(self.offsets_axes[0])
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitleOffset(self.offsets_axes[1])
                leg_relativesys.AddEntry(tgsys_cat[ibin2][sys_cat], self.systematic_catlabels[sys_cat], "F")
                if sys_cat == 0:
                    tgsys_cat[ibin2][sys_cat].Draw("A2")
                else:
                    tgsys_cat[ibin2][sys_cat].Draw("2")
                unc_rel_min = 100.0
                unc_rel_max = 0.0
                for ibinshape in range(self.p_nbinshape_gen):
                    print(
                        "rel. syst. unc. ",
                        self.systematic_catlabels[sys_cat],
                        " ",
                        self.lvar2_binmin_gen[ibin2],
                        " ",
                        self.lvar2_binmax_gen[ibin2],
                        " ",
                        tgsys_cat[ibin2][sys_cat].GetErrorYhigh(ibinshape),
                        " ",
                        tgsys_cat[ibin2][sys_cat].GetErrorYlow(ibinshape),
                    )
                    unc_rel_min = min(
                        unc_rel_min,
                        tgsys_cat[ibin2][sys_cat].GetErrorYhigh(ibinshape),
                        tgsys_cat[ibin2][sys_cat].GetErrorYlow(ibinshape),
                    )
                    unc_rel_max = max(
                        unc_rel_max,
                        tgsys_cat[ibin2][sys_cat].GetErrorYhigh(ibinshape),
                        tgsys_cat[ibin2][sys_cat].GetErrorYlow(ibinshape),
                    )
                print(
                    f"rel. syst. unc. {self.systematic_catlabels[sys_cat]} (%): min. {(100. * unc_rel_min):.2g}, max. {(100. * unc_rel_max):.2g}"
                )
            h_default_stat_err[ibin2].Draw("same")
            h_default_stat_err[ibin2].Draw("axissame")
            # Draw LaTeX
            y_latex = self.y_latex_top
            list_latex = []
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys.Draw("same")
            crelativesys.SaveAs("%s/sys_unc_%s.eps" % (self.d_resultsallpdata, suffix))
            if ibin2 == 1:
                crelativesys.SaveAs("%s/%s_sys_unc_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix))
            gStyle.SetErrorX(0.5)

            # plot the relative systematic uncertainties for all categories together
            # same as above but categories combined into groups

            crelativesys_gr = TCanvas("crelativesys_gr " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys_gr)
            crelativesys_gr.SetCanvasSize(1000, 800)  # original width 900
            crelativesys_gr.SetBottomMargin(self.margins_can[0])
            crelativesys_gr.SetLeftMargin(9 / 10 * self.margins_can[1])  # scale for width 900 -> 1000
            crelativesys_gr.SetTopMargin(self.margins_can[2])
            # crelativesys_gr.SetRightMargin(0.25)
            crelativesys_gr.SetRightMargin(1 - 9 / 10 * (1 - 0.25))  # scale for width 900 -> 1000
            # leg_relativesys_gr = TLegend(.77, .2, 0.95, .85)
            leg_relativesys_gr = TLegend(0.77 * 9 / 10, 0.5, 0.95, 0.85)  # scale for width 900 -> 1000
            setup_legend(leg_relativesys_gr, textsize=self.fontsize)
            y_min_g, y_max_g = get_y_window_gr(tgsys_gr[ibin2])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            list_y_margin_up = [0.2, 0.35, 0.2]
            y_margin_up = list_y_margin_up[i_shape]
            y_margin_down = 0.05
            setup_histogram(h_default_stat_err[ibin2])
            h_default_stat_err[ibin2].SetMarkerStyle(0)
            h_default_stat_err[ibin2].SetMarkerSize(0)
            leg_relativesys_gr.AddEntry(h_default_stat_err[ibin2], "stat. unc.", "E")
            for sys_gr, gr in enumerate(self.systematic_catgroups_list):
                setup_tgraph(tgsys_gr[ibin2][sys_gr], get_colour(sys_gr + 1, 0))
                tgsys_gr[ibin2][sys_gr].SetTitle("")
                tgsys_gr[ibin2][sys_gr].SetLineWidth(3)
                tgsys_gr[ibin2][sys_gr].SetFillStyle(0)
                tgsys_gr[ibin2][sys_gr].GetYaxis().SetRangeUser(
                    *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                )
                tgsys_gr[ibin2][sys_gr].GetXaxis().SetLimits(
                    round(self.lvarshape_binmin_gen[0 if self.shape == "nsd" else 1], 2),
                    round(self.lvarshape_binmax_gen[-1], 2),
                )
                if self.shape == "nsd":
                    tgsys_gr[ibin2][sys_gr].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tgsys_gr[ibin2][sys_gr], 0.2)
                tgsys_gr[ibin2][sys_gr].GetXaxis().SetTitle(self.v_varshape_latex)
                tgsys_gr[ibin2][sys_gr].GetYaxis().SetTitle("relative systematic uncertainty")
                tgsys_gr[ibin2][sys_gr].GetXaxis().SetTitleOffset(self.offsets_axes[0])
                tgsys_gr[ibin2][sys_gr].GetYaxis().SetTitleOffset(self.offsets_axes[1])
                leg_relativesys_gr.AddEntry(tgsys_gr[ibin2][sys_gr], gr, "F")
                if sys_gr == 0:
                    tgsys_gr[ibin2][sys_gr].Draw("A2")
                else:
                    tgsys_gr[ibin2][sys_gr].Draw("2")
                unc_rel_min = 100.0
                unc_rel_max = 0.0
                for ibinshape in range(self.p_nbinshape_gen):
                    print(
                        "rel. syst. unc. ",
                        gr,
                        " ",
                        self.lvar2_binmin_gen[ibin2],
                        " ",
                        self.lvar2_binmax_gen[ibin2],
                        " ",
                        tgsys_gr[ibin2][sys_gr].GetErrorYhigh(ibinshape),
                        " ",
                        tgsys_gr[ibin2][sys_gr].GetErrorYlow(ibinshape),
                    )
                    unc_rel_min = min(
                        unc_rel_min,
                        tgsys_gr[ibin2][sys_gr].GetErrorYhigh(ibinshape),
                        tgsys_gr[ibin2][sys_gr].GetErrorYlow(ibinshape),
                    )
                    unc_rel_max = max(
                        unc_rel_max,
                        tgsys_gr[ibin2][sys_gr].GetErrorYhigh(ibinshape),
                        tgsys_gr[ibin2][sys_gr].GetErrorYlow(ibinshape),
                    )
                print(f"rel. syst. unc. {gr} (%): min. {(100. * unc_rel_min):.2g}, max. {(100. * unc_rel_max):.2g}")
            h_default_stat_err[ibin2].Draw("same")
            h_default_stat_err[ibin2].Draw("axissame")
            # Draw LaTeX
            y_latex = self.y_latex_top
            list_latex = []
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys_gr.Draw("same")
            crelativesys_gr.SaveAs("%s/sys_unc_gr_%s.eps" % (self.d_resultsallpdata, suffix))
            if ibin2 == 1:
                crelativesys_gr.SaveAs("%s/%s_sys_unc_gr_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix))
            gStyle.SetErrorX(0.5)


def main(args=None):
    """
    This is used as the entry point for ml-analysis.
    Read optional command line arguments and launch the analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-analysis", "-d", dest="database_analysis", help="analysis database to be used", required=True
    )
    parser.add_argument("--analysis", "-a", dest="type_ana", help="choose type of analysis", required=True)
    args = parser.parse_args(args)

    analyser = AnalyzerJetSystematics(args.database_analysis, args.type_ana)
    analyser.jetsystematics()


if __name__ == "__main__":
    main()
