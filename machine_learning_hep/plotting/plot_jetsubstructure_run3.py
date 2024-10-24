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
Script for plotting figures of the Run 3 HF-jet substructure analyses
Author: Vit Kucera <vit.kucera@cern.ch>
"""

# pylint: disable=too-many-lines, too-many-instance-attributes, too-many-statements, too-many-locals
# pylint: disable=too-many-nested-blocks, too-many-branches, consider-using-f-string
# pylint: disable=unused-variable

import argparse
import logging
import os
from array import array
from functools import reduce
from math import ceil
from pathlib import Path

import numpy as np
import yaml
from ROOT import TH1, TCanvas, TFile, TLegend, TLine, TVirtualPad, gROOT, gStyle

from machine_learning_hep.analysis.analyzer_jets import (
    string_range_pthf,
    string_range_ptjet,
)
from machine_learning_hep.logger import configure_logger, get_logger

# HF specific imports
from machine_learning_hep.utilities import (
    count_histograms,
    divide_graphs,
    divide_histograms,
    draw_latex_lines,
    get_colour,
    get_marker,
    get_mean_graph,
    get_mean_hist,
    get_mean_uncertainty,
    make_message_notfound,
    make_plot,
    make_ratios,
    reset_graph_outside_range,
    reset_hist_outside_range,
    setup_legend,
)
from machine_learning_hep.utils.hist import (
    bin_array,
    get_axis,
    get_bin_limits,
    get_dim,
    project_hist,
)


def shrink_err_x(graph, width=0.1):
    for i in range(graph.GetN()):
        graph.SetPointEXlow(i, width)
        graph.SetPointEXhigh(i, width)


# final ranges
x_range = {
    "zg": [0.1, 0.5],
    "rg": [0.0, 0.4],
    "nsd": [-0.5, 5.5],
    "zpar": [0.4, 1.0],
}

x_bins_run2 = {
    "zg": [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5],
    "rg": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    "nsd": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    "zpar": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}


class Plotter:
    def __init__(self, path_input_file: str, path_database_analysis: str, typean: str, var: str, mcordata: str):
        configure_logger(False)
        self.logger = get_logger()
        self.typean = typean
        self.var = var
        self.mcordata = mcordata
        self.method = "sidesub"
        self.logger.setLevel(logging.INFO)

        self.species = None
        if "D0Jet" in path_database_analysis:
            self.species = "D0"
        if "LcJet" in path_database_analysis:
            self.species = "Lc"

        with open(path_database_analysis, "r", encoding="utf-8") as file_db:
            db_analysis = yaml.safe_load(file_db)
        case = list(db_analysis.keys())[0]
        self.datap = db_analysis[case]
        self.db_typean = self.datap["analysis"][self.typean]

        # directories with analysis results
        self.dir_result_data = self.db_typean["data"]["resultsallp"]
        file_result_name = self.datap["files_names"]["resultfilename"]
        self.path_file_results = os.path.join(self.dir_result_data, file_result_name)

        # If input file path is not provided, take the analysis result.
        self.path_input_file = path_input_file if path_input_file else self.path_file_results
        self.dir_input = os.path.dirname(self.path_input_file)
        self.file_results = None

        self.logger.info("Input file: %s", self.path_input_file)

        # output directory for figures
        self.fig_formats = ["pdf", "png"]
        self.dir_out_figs = Path(f"{os.path.expandvars(self.dir_input)}/fig/plots")
        for fmt in self.fig_formats:
            (self.dir_out_figs / fmt).mkdir(parents=True, exist_ok=True)

        self.logger.info("Plots will be saved in %s", self.dir_out_figs)

        # plotting
        self.list_obj = []
        self.plot_order_default = None
        self.plot_order = None
        self.labels_obj = []
        self.list_latex = []
        self.list_colours = []
        self.list_markers = []
        self.list_new = []  # list to avoid loosing objects created in loops

        # LaTeX string
        self.latex_hadron = self.db_typean["latexnamehadron"]
        self.latex_ptjet = "#it{p}_{T}^{jet ch}"
        self.latex_pthf = "#it{p}_{T}^{%s} (GeV/#it{c})" % self.latex_hadron
        self.latex_obs = self.db_typean["observables"][self.var]["label"]
        self.latex_y = self.db_typean["observables"][self.var]["label_y"]

        # binning of hadron pt
        self.edges_pthf = np.asarray(self.cfg("sel_an_binmin", []) + self.cfg("sel_an_binmax", [])[-1:], "d")
        self.n_bins_pthf = len(self.edges_pthf) - 1

        # binning of jet pt
        # reconstruction level
        self.edges_ptjet_rec = self.db_typean["bins_ptjet"]
        self.n_bins_ptjet_rec = len(self.edges_ptjet_rec) - 1
        self.ptjet_rec_min = self.edges_ptjet_rec[0]
        self.ptjet_rec_max = self.edges_ptjet_rec[-1]
        self.edges_ptjet_rec_min = self.edges_ptjet_rec[:-1]
        self.edges_ptjet_rec_max = self.edges_ptjet_rec[1:]
        # generator level
        self.edges_ptjet_gen = self.db_typean["bins_ptjet"]
        self.edges_ptjet_gen_eff = self.db_typean["bins_ptjet"]
        self.n_bins_ptjet_gen = len(self.edges_ptjet_gen) - 1
        self.ptjet_gen_min = self.edges_ptjet_gen[0]
        self.ptjet_gen_max = self.edges_ptjet_gen[-1]
        self.edges_ptjet_gen_min = self.edges_ptjet_gen[:-1]
        self.edges_ptjet_gen_max = self.edges_ptjet_gen[1:]

        # binning of observable (z, shape,...)
        # reconstruction level
        if binning := self.cfg(f"observables.{var}.bins_det_var"):
            bins_tmp = np.asarray(binning, "d")
        elif binning := self.cfg(f"observables.{var}.bins_det_fix"):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f"observables.{var}.bins_var"):
            bins_tmp = np.asarray(binning, "d")
        elif binning := self.cfg(f"observables.{var}.bins_fix"):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error("No binning specified for %s, using defaults", var)
            bins_tmp = bin_array(10, 0.0, 1.0)
        self.edges_obs_rec = bins_tmp
        # self.n_bins_obs_rec = len(self.edges_obs_rec) - 1
        # self.obs_rec_min = float(self.edges_obs_rec[0])
        # self.obs_rec_max = float(self.edges_obs_rec[-1])

        # generator level
        if binning := self.cfg(f"observables.{var}.bins_gen_var"):
            bins_tmp = np.asarray(binning, "d")
        elif binning := self.cfg(f"observables.{var}.bins_gen_fix"):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f"observables.{var}.bins_var"):
            bins_tmp = np.asarray(binning, "d")
        elif binning := self.cfg(f"observables.{var}.bins_fix"):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error("No binning specified for %s, using defaults", var)
            bins_tmp = bin_array(10, 0.0, 1.0)
        self.edges_obs_gen = bins_tmp
        # self.n_bins_obs_gen = len(self.edges_obs_gen) - 1
        # self.obs_gen_min = float(self.edges_obs_gen[0])
        # self.obs_gen_max = float(self.edges_obs_gen[-1])

        # unfolding
        self.niter_unfolding = self.db_typean["unfolding_iterations"]
        self.choice_iter_unfolding = self.db_typean["unfolding_iterations_sel"]

        self.logger.info("Rec obs edges: %s, Gen obs edges: %s", self.edges_obs_rec, self.edges_obs_gen)
        self.logger.info("Rec ptjet edges: %s, Gen ptjet edges: %s", self.edges_ptjet_rec, self.edges_ptjet_gen)

        # official figures
        self.size_can = [800, 800]
        self.size_can_double = [800, 800]
        # self.margins_can = [0.1, 0.13, 0.1, 0.03]  # [bottom, left, top, right]
        self.margins_can = [0.08, 0.12, 0.08, 0.05]  # [bottom, left, top, right]
        # self.margins_can_double = [0.1, 0.1, 0.1, 0.1]
        # self.margins_can_double = [0., 0., 0., 0.]
        self.offsets_axes = [0.8, 1.3]
        # self.offsets_axes_double = [0.8, 0.8]
        # self.size_thg = 0.05
        # self.offset_thg = 0.85
        # self.fontsize = 0.06
        self.fontsize_glob = 0.032  # font size relative to the canvas height
        # self.scale_title = 1.3  # scaling factor to increase the size of axis titles
        self.tick_length = 0.02
        self.opt_plot_h = ""
        self.opt_leg_h = "P"  # marker
        self.opt_plot_g = "2P"  # marker and error rectangles
        self.opt_leg_g = "PF"  # L line, P maker, F box, E vertical error bar
        self.x_latex = 0.16
        # self.y_latex_top = 1. - self.margins_can[2] - self.fontsize_glob - self.tick_length - 0.01
        self.y_latex_top = None
        self.y_step_glob = 0.052
        self.leg_pos_default = [0.72, 0.7, 0.85, 0.8]
        self.leg_pos = self.leg_pos_default
        self.scale_text_leg_default = 0.8
        self.scale_text_leg = 0.8
        self.leg_horizontal_default = True
        self.leg_horizontal = True
        # self.y_margin_up = 0.46
        self.y_margin_up = 0.05
        self.y_margin_up_default = 0.05
        self.y_margin_down = 0.05
        self.y_margin_down_default = 0.05
        self.plot_errors_x = True  # plot horizontal error bars

        # axes titles
        self.title_x = self.latex_obs
        self.title_y = self.latex_y
        self.title_full = f";{self.title_x};{self.title_y}"
        self.title_full_default = self.title_full
        # self.title_full_ratio = ";%s;data/MC: ratio of %s" % (self.title_x, self.title_y)
        # text
        self.text_alice = "ALICE Preliminary, pp"  # preliminaries
        # self.text_alice = "#bf{ALICE}, pp, #sqrt{#it{s}} = 13.6 TeV"  # paper
        self.text_tagged = "%s-tagged" % self.latex_hadron
        # self.text_jets = "charged-particle jets, anti-#it{k}_{T}, #it{R} = 0.4"
        self.text_jets = self.text_tagged + " " + "charged-particle jets, anti-#it{k}_{T}, #it{R} = 0.4"
        self.text_ptjet = "%g #leq %s (GeV/#it{c}) < %g"
        self.text_ptcut = "#it{p}_{T, incl. ch. jet}^{leading track} #geq 5.33 GeV/#it{c}"
        self.text_etajet = "|#it{#eta}_{jet ch}| < 0.5"
        self.text_pth_yh = "%g #leq #it{p}_{T}^{%s} (GeV/#it{c}) < %g, |#it{y}_{%s}| < 0.8"
        self.text_sd = "Soft drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        self.text_run2 = "#sqrt{#it{s}} = 13 TeV"
        self.text_run3 = "#sqrt{#it{s}} = 13.6 TeV"
        # self.text_acc_h = "|#it{y}| < 0.8"
        # self.text_powheg = "POWHEG + PYTHIA 6 + EvtGen"
        self.text_monash = "PYTHIA 8 Monash"
        self.text_mode2 = "PYTHIA 8 CR Mode 2"
        self.range_x = None
        self.range_y = None

        # colour and marker indices
        self.c_lc_data = 0
        self.c_d0_data = 1
        self.c_lc_monash = 2
        self.c_lc_mode2 = 3
        self.c_lcd0_data = 6
        self.c_d0_monash = 4
        self.c_d0_mode2 = 5

        # markers
        self.m_lc_data = get_marker(0)
        self.m_d0_data = get_marker(1)
        self.m_lc_monash = 1  # get_marker(2)
        self.m_lc_mode2 = 1  # get_marker(3)
        self.m_lcd0_data = get_marker(4)
        self.m_d0_monash = get_marker(4)
        self.m_d0_mode2 = get_marker(5)

        # line styles
        self.l_monash = 2
        self.l_mode2 = 7

        self.path_results_other = "$HOME/mlhep"

    def cfg(self, param, default=None):
        return reduce(
            lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
            param.split("."),
            self.datap["analysis"][self.typean],
        )

    def save_canvas(self, can, name=""):
        """Save canvas"""
        if not name:
            name = can.GetName()
        for fmt in self.fig_formats:
            can.SaveAs(f"{self.dir_out_figs}/{fmt}/{name}.{fmt}")

    def crop_histogram(self, hist, var: str):
        """Constrain x range of histogram and reset outside bins."""
        if var not in x_range:
            return
        hist.GetXaxis().SetRangeUser(round(x_range[var][0], 2), round(x_range[var][1], 2))
        hist.GetXaxis().SetLimits(round(x_range[var][0], 2), round(x_range[var][1], 2))
        reset_hist_outside_range(hist, *x_range[var])

    def crop_graph(self, graph, var: str):
        """Constrain x range of graph and reset outside points."""
        if var not in x_range:
            return
        graph.GetXaxis().SetRangeUser(round(x_range[var][0], 2), round(x_range[var][1], 2))
        graph.GetXaxis().SetLimits(round(x_range[var][0], 2), round(x_range[var][1], 2))
        reset_graph_outside_range(graph, *x_range[var])

    def get_object(self, name: str, file=None):
        if file is None:
            file = self.file_results
        if not (obj := file.Get(name)):
            self.logger.fatal(make_message_notfound(name))
        if isinstance(obj, TH1):
            obj.SetDirectory(0)  # Decouple the object from the file.
        return obj

    def get_objects(self, *names: str, file=None):
        return [self.get_object(name, file) for name in names]

    def get_run2_lc_ff_data(self) -> dict:
        path_file_stat = f"{self.path_results_other}/run2/results/lc/unfolding_results.root"
        path_file_syst = f"{self.path_results_other}/run2/results/lc/systematics_results.root"
        self.logger.info("Getting Run 2 Lc data from %s %s.", path_file_stat, path_file_syst)
        dict_obj = {}
        pattern_stat = "unfolded_z_sel_pt_jet_%.2f_%.2f"
        pattern_sys = "tgsys_pt_jet_%.2f_%.2f"
        for iptjet in (0, 1):
            with TFile.Open(path_file_stat) as file_stat, TFile.Open(path_file_syst) as file_syst:
                name_stat = pattern_stat % (self.edges_ptjet_gen[iptjet], self.edges_ptjet_gen[iptjet + 1])
                name_syst = pattern_sys % (self.edges_ptjet_gen[iptjet], self.edges_ptjet_gen[iptjet + 1])
                dict_obj[iptjet] = {
                    "stat": self.get_object(name_stat, file_stat),
                    "syst": self.get_object(name_syst, file_syst),
                }
        return dict_obj

    def get_run2_lc_ff_sim(self) -> dict:
        path_file = f"{self.path_results_other}/run2/results/lc/simulations.root"
        self.logger.info("Getting Run 2 Lc sim from %s.", path_file)
        names = {"monash": "input_pythia8defaultpt_jet_7.00_15.00", "cr2": "input_pythia8colour2softpt_jet_7.00_15.00"}
        with TFile.Open(path_file) as file:
            return {title: self.get_object(name, file) for title, name in names.items()}

    def get_run2_d0_ff_sim(self) -> dict:
        path_file = f"{self.path_results_other}/run2/results/d0/simulations_3_D0.root"
        self.logger.info("Getting Run 2 D0 sim from %s.", path_file)
        names = {"monash": "input_pythia8defaultpt_jet_7.00_15.00", "cr2": "input_pythia8colour2softpt_jet_7.00_15.00"}
        with TFile.Open(path_file) as file:
            return {title: self.get_object(name, file) for title, name in names.items()}

    def get_run2_d0_sd(self) -> dict:
        path_file = f"{self.path_results_other}/run2/results/d0/results_all.root"
        self.logger.info("Getting Run 2 D0 SD from %s.", path_file)
        dict_obj = {}
        with TFile.Open(path_file) as file:
            for obs in ("zg", "rg", "nsd"):
                dict_obj[obs] = {}
                for flavour in ("hf", "incl"):
                    dict_obj[obs][flavour] = {}
                    for source in ("data", "pythia"):
                        dict_obj[obs][flavour][source] = {}
                        for type_data in ("stat", "syst"):
                            if source == "pythia" and type_data == "syst":
                                continue
                            name = f"{obs}_{flavour}_{source}_1_{type_data}"
                            dict_obj[obs][flavour][source][type_data] = self.get_object(name, file)
        return dict_obj

    def get_run2_d0_ff_data(self) -> dict:
        path_file = f"{self.path_results_other}/run2/results/d0/FFD0_Jakub_20220130.root"
        self.logger.info("Getting Run 2 D0 FF from %s.", path_file)
        names = {"stat": "hData_binned", "syst": "haeData_binned_syst"}
        with TFile.Open(path_file) as file:
            return {title: self.get_object(name, file) for title, name in names.items()}

    def get_run3_sim(self) -> dict:
        # path_file = "aliceml:/home/nzardosh/PYTHIA_Sim/PYTHIA8_Simulations/Plots/Run3/fOut.root"
        path_file = f"{self.path_results_other}/run3/simulations/fOut_v10.root"
        self.logger.info("Getting Run 3 sim from %s.", path_file)
        pattern = "fh_%s%s_%s_%.2f_JetpT_%.2f"
        obs = {"zg": "Zg", "rg": "Rg", "nsd": "Nsd", "zpar": "FF"}
        species = {"D0": "D0", "Lc": "Lc", "incl": "Inclusive"}
        source = {"monash": "M", "mode2": "SM2"}
        dict_obj = {}
        with TFile.Open(path_file) as file:
            for s_obs, obs in obs.items():
                dict_obj[s_obs] = {}
                for s_spec, spec in species.items():
                    dict_obj[s_obs][s_spec] = {}
                    for s_src, src in source.items():
                        if s_spec == "incl" and s_src == "mode2":
                            continue
                        dict_obj[s_obs][s_spec][s_src] = {}
                        for iptjet in (0, 1, 2, 3):
                            name = pattern % (
                                spec,
                                src,
                                obs,
                                self.edges_ptjet_gen[iptjet],
                                self.edges_ptjet_gen[iptjet + 1],
                            )
                            obj = self.get_object(name, file)
                            # obj.Scale(1. / obj.Integral(), "width")
                            dict_obj[s_obs][s_spec][s_src][iptjet] = obj
        return dict_obj

    def report_means(self, h_stat, h_syst, iptjet):
        mean_z_stat = get_mean_hist(h_stat)
        mean_z_syst = get_mean_graph(h_syst)
        hist_means_stat, hist_means_syst, hist_means_comb = get_mean_uncertainty(h_stat, h_syst, 100000)
        mean_z_var_comb = hist_means_comb.GetMean()
        sigma_z_var_comb = hist_means_comb.GetStdDev()
        mean_z_var_stat = hist_means_stat.GetMean()
        sigma_z_var_stat = hist_means_stat.GetStdDev()
        mean_z_var_syst = hist_means_syst.GetMean()
        sigma_z_var_syst = hist_means_syst.GetStdDev()
        make_plot(
            f"{self.var}_means_hf_comb_{iptjet}",
            list_obj=[hist_means_comb],
            suffix="pdf",
            title=f"HF mean variations comb {iptjet};{self.latex_obs}",
        )
        make_plot(
            f"{self.var}_means_hf_stat_{iptjet}",
            list_obj=[hist_means_stat],
            suffix="pdf",
            title=f"HF mean variations stat {iptjet};{self.latex_obs}",
        )
        make_plot(
            f"{self.var}_means_hf_syst_{iptjet}",
            list_obj=[hist_means_syst],
            suffix="pdf",
            title=f"HF mean variations syst {iptjet};{self.latex_obs}",
        )
        print(f"Mean HF {self.var} = stat {mean_z_stat} syst {mean_z_syst} ROOT stat {h_stat.GetMean()}")
        print(f"Mean HF {self.var} = var comb {mean_z_var_comb} +- {sigma_z_var_comb}")
        print(f"Mean HF {self.var} = var stat {mean_z_var_stat} +- {sigma_z_var_stat}")
        print(f"Mean HF {self.var} = var syst {mean_z_var_syst} +- {sigma_z_var_syst}")

    def get_text_range_ptjet(self, iptjet=-1):
        pt_min = self.edges_ptjet_gen[0] if iptjet < 0 else self.edges_ptjet_gen[iptjet]
        pt_max = self.edges_ptjet_gen[-1] if iptjet < 0 else self.edges_ptjet_gen[iptjet + 1]
        return self.text_ptjet % (pt_min, self.latex_ptjet, pt_max)

    def get_text_range_pthf(self, ipthf=-1, iptjet=-1, hadron=None):
        if hadron is None:
            hadron = self.latex_hadron
        pt_min = self.edges_pthf[0] if ipthf < 0 else self.edges_pthf[ipthf]
        pt_max = self.edges_pthf[-1] if ipthf < 0 else self.edges_pthf[ipthf + 1]
        pt_max = min(pt_max, self.edges_ptjet_gen[iptjet + 1]) if iptjet > -1 else pt_max
        return self.text_pth_yh % (pt_min, hadron, pt_max, hadron)

    def make_plot(self, name: str, can=None, pad=0, scale=1.0, colours=None, markers=None):
        """Wrapper method for calling make_plot and saving the canvas."""
        assert all(self.list_obj)
        n_obj = len(self.list_obj)
        # assert len(self.labels_obj) == n_obj
        self.list_colours = [get_colour(i) for i in range(n_obj)]
        if colours is not None:
            self.list_colours = colours
        self.list_markers = [get_marker(i) for i in range(n_obj)]
        if markers is not None:
            self.list_markers = markers

        margin_bottom = self.margins_can[0]  # size of the bottom margin relative to the canvas height
        margin_left = self.margins_can[1]  # size of the right margin relative to the canvas height
        margin_top = self.margins_can[2]  # size of the top margin relative to the canvas height
        margin_right = self.margins_can[3]  # size of the right margin relative to the canvas height
        padding_top_glob = self.tick_length + 0.01
        padding_left_glob = self.tick_length + 0.01
        padding_right_glob = self.tick_length + 0.01
        # padding_right_glob = self.tick_length + 0.5  # for a single-column horizontal legend
        if self.y_latex_top is None:
            y_latex_top_glob = 1.0 - (self.fontsize_glob + padding_top_glob)
            y_latex_top_loc = 1.0 - (self.fontsize_glob + padding_top_glob) / scale
        else:
            y_latex_top_glob = self.y_latex_top
            y_latex_top_loc = 1.0 - (1.0 - y_latex_top_glob) / scale
        if pad in (0, 1):
            y_latex_top_glob -= margin_top
            y_latex_top_loc -= margin_top / scale

        # Adjust global legend parameters.
        scale_text_leg = self.scale_text_leg
        leg_pos_glob = self.leg_pos.copy()
        n_entries_leg = len([s for s in self.labels_obj if s])
        n_rows = 1
        n_columns = 2
        if self.leg_horizontal:
            n_rows = ceil(n_entries_leg / n_columns)
            if self.list_latex:
                y_leg_max = y_latex_top_glob - self.y_step_glob * (len(self.list_latex) - 1 + 0.2)
            else:
                y_leg_max = 1.0 - margin_top - padding_top_glob
            y_leg_min = y_leg_max - n_rows * self.y_step_glob
            # leg_pos_glob = [self.x_latex, y_leg_min, 1. - margin_right - padding_right_glob, y_leg_max]
            leg_pos_glob = [
                margin_left + padding_left_glob,
                y_leg_min,
                1.0 - margin_right - padding_right_glob,
                y_leg_max,
            ]
        else:
            leg_pos_glob[1] = leg_pos_glob[3] - n_entries_leg * self.y_step_glob * scale_text_leg
        leg_height_glob = leg_pos_glob[3] - leg_pos_glob[1]

        # Recalculate local coordinates to preserve absolute size of text
        # and its absolute offset from the top of the panel.
        leg_pos_loc = leg_pos_glob.copy()
        y_panel_top_loc = 1.0
        panel_height_loc = 1.0
        if pad == 0:
            y_panel_top_loc -= margin_top
            panel_height_loc -= margin_bottom + margin_top
        else:
            if pad == 1:
                y_panel_top_loc -= margin_top / scale
                panel_height_loc -= margin_top / scale
                leg_pos_loc[3] = 1.0 - (1.0 - leg_pos_glob[3]) / scale
                leg_pos_loc[1] = leg_pos_loc[3] - leg_height_glob / scale
            else:
                leg_pos_loc[3] = 1.0 - (1.0 - leg_pos_glob[3] - margin_top) / scale
                leg_pos_loc[1] = leg_pos_loc[3] - leg_height_glob / scale
            if pad == self.get_n_pads(can):
                panel_height_loc -= margin_bottom / scale

        # Adjust panel margin for the height of text.
        y_margin_up_loc = self.y_margin_up
        y_margin_down_loc = self.y_margin_down
        if self.list_latex:
            y_latex_bottom_loc = y_latex_top_loc - self.y_step_glob / scale * (
                len(self.list_latex) - 1 + n_rows * int(self.leg_horizontal and n_entries_leg > 0)
            )
            y_margin_up_loc += (y_panel_top_loc - y_latex_bottom_loc) / panel_height_loc
        elif self.leg_horizontal and n_entries_leg > 0:
            y_margin_up_loc += (y_panel_top_loc - leg_pos_loc[1]) / panel_height_loc
        assert y_margin_down_loc + y_margin_up_loc < 1.0

        # Plot
        can, new = make_plot(
            name,
            can=can,
            pad=pad,
            scale=scale,
            list_obj=self.list_obj,
            labels_obj=self.labels_obj,
            plot_order=self.plot_order,
            opt_leg_h=self.opt_leg_h,
            opt_plot_h=self.opt_plot_h,
            opt_leg_g=self.opt_leg_g,
            opt_plot_g=self.opt_plot_g,
            offsets_xy=self.offsets_axes,
            size=self.size_can,
            font_size=self.fontsize_glob,
            colours=self.list_colours,
            markers=self.list_markers,
            leg_pos=leg_pos_loc,
            margins_y=[y_margin_down_loc, y_margin_up_loc],
            margins_c=self.margins_can,
            range_x=self.range_x,
            range_y=self.range_y,
            title=self.title_full,
        )
        leg = new[0]
        if self.leg_horizontal:
            n_entries_leg = leg.GetListOfPrimitives().GetSize()
            leg.SetNColumns(n_columns)
        leg.SetTextSize(self.fontsize_glob / scale * scale_text_leg)
        self.list_new += new
        if self.list_latex:
            self.list_new += draw_latex_lines(
                self.list_latex,
                x_start=self.x_latex,
                y_start=y_latex_top_loc,
                y_step=self.y_step_glob / scale,
                font_size=self.fontsize_glob / scale,
            )
        gStyle.SetErrorX(0.5)  # reset default width
        if not self.plot_errors_x:
            gStyle.SetErrorX(0)  # do not plot horizontal error bars of histograms
        self.save_canvas(can)
        gStyle.SetErrorX(0.5)  # reset default width
        return can, new

    def get_n_pads(self, can) -> int:
        """Count pads in a canvas."""
        if not can:
            return 0
        npads = 0
        for obj in can.GetListOfPrimitives():
            if obj.InheritsFrom(TVirtualPad.Class()):
                npads += 1
        return npads

    def set_pad_heights(self, can, panel_heights_ratios: list[float]) -> list[float]:
        """Divide canvas vertically into adjacent panels and set pad margins.
        Resulting canvas will have pads containing panels of heights in proportions in panel_heights_ratios.
        Returns heights of resulting pads."""
        epsilon = 1.0e-6
        # Calculate panel heights relative to the canvas height based on panel height ratios and canvas margins.
        margin_bottom = self.margins_can[0]  # height of the bottom margin relative to the canvas height
        margin_left = self.margins_can[1]  # height of the left margin relative to the canvas height
        margin_top = self.margins_can[2]  # height of the top margin relative to the canvas height
        margin_right = self.margins_can[3]  # height of the right margin relative to the canvas height
        h_usable = 1.0 - margin_top - margin_bottom  # usable fraction of the canvas height
        panel_heights = [h / sum(panel_heights_ratios) * h_usable for h in panel_heights_ratios]
        assert abs(sum(panel_heights) + margin_bottom + margin_top - 1.0) < epsilon
        # Create pads.
        n_pads = len(panel_heights)
        can.Divide(1, n_pads)
        pads = [can.cd(i + 1) for i in range(n_pads)]
        # Calculate heights of the pads relative to the canvas height.
        pad_heights = panel_heights
        pad_heights[0] += margin_top
        pad_heights[-1] += margin_bottom
        # Calculate pad edges. (from 1 to 0)
        y_edges = [1.0 - sum(pad_heights[0:i]) for i in range(n_pads + 1)]
        assert abs(y_edges[0] - 1.0) < epsilon
        assert abs(y_edges[-1]) < epsilon
        # Set pad edges and margins.
        for i in range(n_pads):
            pads[i].SetPad(0.0, y_edges[i + 1], 1.0, y_edges[i])
            pads[i].SetBottomMargin(0.0)
            pads[i].SetTopMargin(0.0)
            pads[i].SetLeftMargin(margin_left)
            pads[i].SetRightMargin(margin_right)
            pads[i].SetTicks(1, 1)
        pads[0].SetTopMargin(margin_top / pad_heights[0])
        pads[-1].SetBottomMargin(margin_bottom / pad_heights[-1])
        return pad_heights

    def plot(self):
        self.logger.info("Observable: %s %s", self.species, self.var)

        with TFile.Open(self.path_input_file) as self.file_results:
            name_hist_unfold_2d = f"h_ptjet-{self.var}_{self.method}_unfolded_{self.mcordata}_0"
            hist_unfold = self.get_object(name_hist_unfold_2d)
            axis_ptjet = get_axis(hist_unfold, 0)
            line_1 = TLine(x_range[self.var][0], 1.0, x_range[self.var][1], 1.0)
            line_1.SetLineStyle(9)
            line_1.SetLineColor(1)
            line_1.SetLineWidth(3)
            self.leg_pos = self.leg_pos_default

            plot_efficiency = False
            if plot_efficiency and self.mcordata == "data":
                # Efficiency
                self.logger.info("Plotting efficiency")
                self.list_obj = self.get_objects("h_pthf_effnew_pr", "h_pthf_effnew_np")
                self.labels_obj = ["prompt", "nonprompt"]
                self.title_full = f";{self.latex_pthf};{self.latex_hadron} efficiency"
                self.list_latex = [
                    self.text_alice,
                    self.text_jets,
                    f"{self.get_text_range_ptjet()}, {self.text_etajet}",
                    self.get_text_range_pthf(),
                ]
                self.make_plot(f"{self.species}_efficiency_{self.var}")

                bins_ptjet = (0, 1, 2, 3)
                for cat, label in zip(("pr", "np"), ("prompt", "non-prompt")):
                    self.list_obj = self.get_objects(
                        *(
                            f"h_ptjet-pthf_effnew_{cat}_"
                            f"{string_range_ptjet(get_bin_limits(axis_ptjet, iptjet + 1))}"
                            for iptjet in bins_ptjet
                        )
                    )
                    self.labels_obj = [self.get_text_range_ptjet(iptjet) for iptjet in bins_ptjet]
                    self.title_full = f";{self.latex_pthf};{label}-{self.latex_hadron} efficiency"
                    self.make_plot(f"{self.species}_efficiency_{self.var}_{cat}_ptjet")

                # TODO: efficiency (old vs new)

            # Results

            if self.species == "D0":
                # list_iptjet = [0, 1, 2, 3]  # indices of jet pt bins to process
                list_iptjet = [2, 3]  # indices of jet pt bins to process
            if self.species == "Lc":
                list_iptjet = [1]  # indices of jet pt bins to process
            plot_lc_vs_d0 = True
            list_stat_all = []
            list_syst_all = []
            list_labels_all = []
            list_markers_all = []
            list_colours_stat_all = []
            list_colours_syst_all = []

            # loop over jet pt
            for i_iptjet, iptjet in enumerate(list_iptjet):
                range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                string_ptjet = string_range_ptjet(range_ptjet)
                self.range_x = None

                # Sideband subtraction
                plot_sidebands = False
                if plot_sidebands and self.mcordata == "data":
                    self.logger.info("Plotting sideband subtraction")
                    # loop over hadron pt
                    for ipt in range(self.n_bins_pthf):
                        range_pthf = (self.edges_pthf[ipt], self.edges_pthf[ipt + 1])
                        string_pthf = string_range_pthf(range_pthf)

                        self.list_obj = self.get_objects(
                            f"h_ptjet-{self.var}_signal_{string_pthf}_{self.mcordata}",
                            f"h_ptjet-{self.var}_sideband_{string_pthf}_{self.mcordata}",
                            f"h_ptjet-{self.var}_subtracted_notscaled_{string_pthf}" f"_{self.mcordata}",
                        )
                        self.list_obj = [project_hist(h, [1], {0: (iptjet + 1, iptjet + 1)}) for h in self.list_obj]
                        self.labels_obj = ["signal region", "scaled sidebands", "after subtraction"]
                        self.title_full = f";{self.latex_obs};counts"
                        self.list_latex = [
                            self.text_alice,
                            self.text_jets,
                            f"{self.get_text_range_ptjet(iptjet)}, {self.text_etajet}",
                            self.get_text_range_pthf(ipt, iptjet),
                        ]
                        if self.var in ("zg", "rg", "nsd"):
                            self.list_latex.append(self.text_sd)
                            # self.list_latex.append(self.text_ptcut)
                        self.make_plot(
                            f"{self.species}_sidebands_{self.var}_{self.mcordata}_{string_ptjet}_{string_pthf}"
                        )

                # Feed-down subtraction
                plot_feeddown = False
                if plot_feeddown:
                    self.logger.info("Plotting feed-down subtraction")
                    self.list_obj = self.get_objects(
                        f"h_ptjet-{self.var}_{self.method}_effscaled_{self.mcordata}",
                        f"h_ptjet-{self.var}_feeddown_det_final_{self.mcordata}",
                        f"h_ptjet-{self.var}_{self.method}_{self.mcordata}",
                    )
                    self.list_obj[0] = self.list_obj[0].Clone(f"{self.list_obj[0].GetName()}_fd_before_{iptjet}")
                    self.list_obj[1] = self.list_obj[1].Clone(f"{self.list_obj[1].GetName()}_fd_{iptjet}")
                    self.list_obj[2] = self.list_obj[2].Clone(f"{self.list_obj[2].GetName()}_fd_after_{iptjet}")
                    axes = list(range(get_dim(self.list_obj[0])))
                    self.list_obj = [project_hist(h, axes[1:], {0: (iptjet + 1,) * 2}) for h in self.list_obj]
                    self.labels_obj = ["before subtraction", "feed-down", "after subtraction"]
                    self.title_full = f";{self.latex_obs};counts"
                    self.list_latex = [
                        self.text_alice,
                        self.text_jets,
                        f"{self.get_text_range_ptjet(iptjet)}, {self.text_etajet}",
                        self.get_text_range_pthf(-1, iptjet),
                    ]
                    if self.var in ("zg", "rg", "nsd"):
                        self.list_latex.append(self.text_sd)
                        # self.list_latex.append(self.text_ptcut)
                    self.make_plot(f"{self.species}_feeddown_{self.var}_{self.mcordata}_{string_ptjet}")

                    # TODO: feed-down (after 2D, fraction)

                # Unfolding
                plot_unfolding = False
                if plot_unfolding:
                    self.logger.info("Plotting unfolding")
                    self.list_obj = [
                        self.get_object(f"h_{self.var}_{self.method}_unfolded_{self.mcordata}_" f"{string_ptjet}_{i}")
                        for i in range(self.niter_unfolding)
                    ]
                    self.labels_obj = [f"iteration {i + 1}" for i in range(self.niter_unfolding)]
                    self.title_full = f";{self.latex_obs};counts"
                    self.make_plot(f"{self.species}_unfolding_convergence_{self.var}_{self.mcordata}_{string_ptjet}")

                    h_ref = self.get_object(f"h_{self.var}_{self.method}_unfolded_{self.mcordata}_{string_ptjet}_sel")
                    for h in self.list_obj:
                        h.Divide(h_ref)
                    self.title_full = f";{self.latex_obs};counts (variation/default)"
                    self.make_plot(
                        f"{self.species}_unfolding_convergence_ratio_{self.var}_{self.mcordata}_{string_ptjet}"
                    )

                    # TODO: unfolding (before/after)

                # Results
                self.logger.info("Plotting results")
                plot_run2_data = True
                self.list_latex = [
                    self.text_alice,
                    # self.list_latex = [f"{self.text_alice}, {self.text_run3}",
                    self.text_jets,
                    f"{self.get_text_range_ptjet(iptjet)}, {self.text_etajet}",
                    self.get_text_range_pthf(-1, iptjet),
                ]
                if self.var in ("zg", "rg", "nsd"):
                    self.list_latex.append(self.text_sd)
                    # self.list_latex.append(self.text_ptcut)
                self.plot_errors_x = False
                self.range_x = x_range[self.var]
                h_stat = self.get_object(
                    f"h_{self.var}_{self.method}_unfolded_{self.mcordata}_" f"{string_ptjet}_sel_selfnorm"
                )
                self.list_obj = [h_stat]
                self.plot_order = list(range(len(self.list_obj)))
                # self.labels_obj = [self.text_tagged]
                self.labels_obj = [self.text_run3]
                self.list_colours = [get_colour(i_iptjet)]
                self.list_markers = [get_marker(i_iptjet)]
                self.opt_plot_h = [self.opt_plot_h]
                self.opt_leg_h = [self.opt_leg_h]
                list_stat_all += self.list_obj
                label = self.get_text_range_ptjet(iptjet)
                if plot_run2_data:
                    label = f"{self.text_run3}, {label}"
                list_labels_all += [label]
                list_colours_stat_all += self.list_colours
                list_markers_all += self.list_markers
                path_syst = f"{os.path.expandvars(self.dir_input)}/systematics.root"
                if os.path.exists(path_syst):
                    self.logger.info("Getting systematics from %s", path_syst)
                    if not (file_syst := TFile.Open(path_syst)):
                        self.logger.fatal(make_message_notfound(path_syst))
                    gr_syst = self.get_object(f"sys_{self.var}_{string_ptjet}", file_syst)
                    if self.var == "nsd":
                        shrink_err_x(gr_syst)
                    list_syst_all.append(gr_syst)
                    # We need to plot the data on top of the systematics.
                    self.list_obj.insert(0, gr_syst)
                    self.plot_order.insert(0, -1)
                    # self.labels_obj.insert(0, self.text_tagged)
                    self.labels_obj.insert(0, self.text_run3)
                    self.labels_obj[1] = ""  # do not show the histogram in the legend
                    self.list_colours.insert(0, get_colour(i_iptjet))
                    self.list_markers.insert(0, get_marker(i_iptjet))
                    list_colours_syst_all.append(self.list_colours[0])
                self.title_full = self.title_full_default

                # Plot additional stuff.
                plot_run2_lc_ff_data = True
                plot_run2_lc_ff_sim = False

                plot_run2_d0_ff_data = True

                plot_run2_d0_sd = True
                plot_run2_d0_sd_hf_data = True
                plot_run2_d0_sd_hf_sim = False
                plot_run2_d0_sd_incl_data = False
                plot_run2_d0_sd_incl_sim = False

                plot_run3_sim = False
                plot_run3_d0_sd_hf_sim = True
                plot_run3_d0_sd_incl_sim = True

                plot_data = True
                plot_sim = True
                plot_incl = True

                # Plot Run 2, Lc, FF, data, 5-7, 7-15, 15-35 GeV/c
                if (
                    plot_run2_lc_ff_data
                    and plot_data
                    and self.species == "Lc"
                    and self.var == "zpar"
                    and iptjet in (0, 1)
                ):
                    run2_lc_ff_data = self.get_run2_lc_ff_data()
                    self.list_obj += [run2_lc_ff_data[iptjet]["syst"], run2_lc_ff_data[iptjet]["stat"]]
                    self.plot_order += [-1.5, max(self.plot_order) - 0.5]
                    self.labels_obj = [self.text_run3, "", self.text_run2, ""]
                    self.list_colours += [get_colour(-1)] * 2
                    self.list_markers += [get_marker(-1)] * 2
                    self.opt_plot_h += [""]
                    self.opt_leg_h += ["P"]
                # Plot Run 2, Lc, FF, sim, 7-15 GeV/c
                if (
                    plot_run2_lc_ff_sim
                    and plot_sim
                    and self.species == "Lc"
                    and self.var == "zpar"
                    and string_ptjet == string_range_ptjet((7, 15))
                ):
                    run2_lc_ff_sim = self.get_run2_lc_ff_sim()
                    run2_lc_ff_sim["monash"].SetLineStyle(self.l_monash)
                    run2_lc_ff_sim["cr2"].SetLineStyle(self.l_mode2)
                    self.list_obj += [run2_lc_ff_sim["monash"], run2_lc_ff_sim["cr2"]]
                    self.plot_order += [max(self.plot_order) + 1, max(self.plot_order) + 2]
                    self.labels_obj += [f"{self.text_monash}, {self.text_run2}", f"{self.text_mode2}, {self.text_run2}"]
                    self.list_colours += [get_colour(c) for c in (self.c_lc_monash, self.c_lc_mode2)]
                    self.list_markers += [1] * 2
                    self.opt_plot_h += ["hist e"] * 2
                    self.opt_leg_h += ["L"] * 2
                # Plot Run 2, D0, Soft drop, data + sim, 15-30 GeV/c
                if (
                    plot_run2_d0_sd
                    and self.species == "D0"
                    and self.var in ("zg", "rg", "nsd")
                    and string_ptjet == string_range_ptjet((15, 30))
                ):
                    run2_d0_sd = self.get_run2_d0_sd()
                    c = count_histograms(self.list_obj) + 1
                    m = count_histograms(self.list_obj)
                    for source in ("data", "pythia"):
                        for flavour in ("hf", "incl"):
                            c += 1
                            m += 1
                            for type_data in ("syst", "stat"):
                                if source == "pythia" and type_data == "syst":
                                    continue
                                if source == "pythia" and not plot_sim:
                                    continue
                                if source == "data" and not plot_data:
                                    continue
                                if flavour == "incl":
                                    if not plot_incl:
                                        continue
                                    if source == "data" and not plot_run2_d0_sd_incl_data:
                                        continue
                                    if source == "pythia" and not plot_run2_d0_sd_incl_sim:
                                        continue
                                elif flavour == "hf":
                                    if source == "data" and not plot_run2_d0_sd_hf_data:
                                        continue
                                    if source == "pythia" and not plot_run2_d0_sd_hf_sim:
                                        continue

                                obj = run2_d0_sd[self.var][flavour][source][type_data]
                                if self.var == "nsd" and type_data == "syst":
                                    shrink_err_x(obj)
                                self.list_obj += [obj]
                                if type_data == "syst":
                                    self.plot_order += [-1 - 1.0 / len(self.list_obj)]  # increasing between -2 and -1
                                else:
                                    self.plot_order += [max(self.plot_order) + 1]
                                label = f"R2 {flavour} {source}"
                                colour = get_colour(c + 3)
                                marker = get_marker(m)
                                if source == "data":
                                    label = self.text_run2
                                    if flavour == "incl":
                                        label += ", inclusive"
                                        colour = get_colour(3)
                                        marker = get_marker(2)
                                    else:
                                        colour = get_colour(-1)
                                        marker = get_marker(-1)
                                    if type_data == "stat":
                                        label = ""
                                if source == "pythia":
                                    marker = 1
                                self.labels_obj += [label]
                                self.list_colours += [colour]
                                self.list_markers += [marker]
                                if type_data == "stat":
                                    if source == "pythia":
                                        self.opt_plot_h += ["hist e"]
                                        self.opt_leg_h += ["L"]
                                    else:
                                        self.opt_plot_h += [""]
                                        self.opt_leg_h += ["P"]
                # Plot Run 2, D0, FF, data, 7-15 GeV/c (Jakub)
                if (
                    plot_run2_d0_ff_data
                    and plot_data
                    and self.species == "D0"
                    and self.var == "zpar"
                    and string_ptjet == string_range_ptjet((7, 15))
                ):
                    run2_d0_ff_data = self.get_run2_d0_ff_data()
                    self.list_obj += [run2_d0_ff_data["syst"], run2_d0_ff_data["stat"]]
                    self.plot_order += [-0.5, max(self.plot_order) + 1]
                    self.labels_obj += [self.text_run2, ""]
                    self.list_colours += [get_colour(-1)] * 2
                    self.list_markers += [get_marker(-1)] * 2
                    self.opt_plot_h += [""]
                    self.opt_leg_h += ["P"]
                # Plot Run 3, Lc or D0, SD and FF, sim (Nima)
                if plot_run3_sim and plot_sim and iptjet in (0, 1, 2, 3) and self.var != "zpar":
                    run3_sim = self.get_run3_sim()
                    l_spec = []
                    if plot_run3_d0_sd_hf_sim:
                        l_spec.append(self.species)
                    # if plot_incl and plot_run3_d0_sd_incl_sim and iptjet == 2:
                    if plot_incl and plot_run3_d0_sd_incl_sim:
                        l_spec.append("incl")
                    l_src = ["monash", "mode2"]
                    names_run3_sim = {
                        "incl": "inclusive",
                        "D0": "D^{0}-tagged",
                        "Lc": "#Lambda_{c}^{#plus}",
                        "mode2": self.text_mode2,
                        "monash": self.text_monash,
                    }
                    species = {"D0": "D0", "Lc": "Lc", "incl": "Inclusive"}
                    source = {"monash": "M", "mode2": "SM2"}
                    colours_run3_sim = {}
                    lines_run3_sim = {}
                    for s_spec in species:
                        colours_run3_sim[s_spec] = {}
                        lines_run3_sim[s_spec] = {}
                    colours_run3_sim["D0"]["monash"] = get_colour(self.c_lc_monash)
                    colours_run3_sim["D0"]["mode2"] = get_colour(self.c_lc_mode2)
                    colours_run3_sim["Lc"]["monash"] = get_colour(self.c_lc_monash)
                    colours_run3_sim["Lc"]["mode2"] = get_colour(self.c_lc_mode2)
                    colours_run3_sim["incl"]["monash"] = get_colour(self.c_lc_monash)
                    colours_run3_sim["incl"]["mode2"] = get_colour(self.c_lc_mode2)
                    lines_run3_sim["D0"]["monash"] = self.l_monash
                    lines_run3_sim["D0"]["mode2"] = self.l_mode2
                    lines_run3_sim["Lc"]["monash"] = self.l_monash
                    lines_run3_sim["Lc"]["mode2"] = self.l_mode2
                    lines_run3_sim["incl"]["monash"] = self.l_monash
                    lines_run3_sim["incl"]["mode2"] = self.l_mode2
                    for s_spec in l_spec:
                        for s_src in l_src:
                            if s_spec == "incl" and s_src == "mode2":
                                continue
                            if s_spec == "D0" and s_src == "monash":
                                continue
                            obj = run3_sim[self.var][s_spec][s_src][iptjet]
                            rebin = False
                            if rebin and self.var in ("zg", "rg") and s_spec == "incl":
                                n_bins = obj.GetNbinsX()
                                array_x = obj.GetXaxis().GetXbins().GetArray()
                                print(f"Array orig: {[array_x[i] for i in range(n_bins + 1)]}")
                                obj = obj.Rebin(
                                    len(x_bins_run2[self.var]) - 1,
                                    f"{obj.GetName()}_rebin",
                                    array("d", x_bins_run2[self.var]),
                                )
                                obj.Scale(0.5)
                                n_bins = obj.GetNbinsX()
                                array_x = obj.GetXaxis().GetXbins().GetArray()
                                print(f"Array rebinned: {[array_x[i] for i in range(n_bins + 1)]}")
                            obj.SetLineStyle(lines_run3_sim[s_spec][s_src])
                            self.list_obj += [obj]
                            self.plot_order += [max(self.plot_order) + 1]
                            self.labels_obj += [f"{names_run3_sim[s_src]}, {names_run3_sim[s_spec]}"]
                            self.list_colours += [colours_run3_sim[s_spec][s_src]]
                            self.list_markers += [1]
                            self.opt_plot_h += ["hist e"]
                            self.opt_leg_h += ["L"]
                self.leg_pos = [0.53, 0.8, 0.85, 0.68]
                if self.var == "rg":
                    if iptjet == 2:
                        self.leg_pos = [self.x_latex, 0.8, self.x_latex + 0.32, 0.63]
                        # self.leg_pos[3] -= self.y_step_glob  # if plotting leading track cut for inclusive jets
                    elif iptjet == 3:
                        self.leg_pos = [0.3, 0.8, 0.3 + 0.32, 0.25]
                elif self.species == "Lc" and self.var == "zpar":
                    self.leg_pos = [0.55, 0.8, 0.8, 0.68]
                self.leg_horizontal = True
                can, new = self.make_plot(
                    f"{self.species}_results_{self.var}_{self.mcordata}_{string_ptjet}",
                    colours=self.list_colours,
                    markers=self.list_markers,
                )
                # redo_leg = True
                # if redo_leg:
                #     leg = new[0]
                #     leg_new = leg.Clone()
                #     leg_new.Clear()
                #     for i in (0, 2, 1, 3):
                #         leg_new.AddEntry(self.list_obj[i], self.opt_leg[])

                # Reset defaults.
                self.plot_order = self.plot_order_default
                self.opt_plot_h = ""
                self.opt_leg_h = "P"
                self.leg_pos = self.leg_pos_default
                self.scale_text_leg = self.scale_text_leg_default
                self.leg_horizontal = self.leg_horizontal_default

                # Plot Lc vs D0.
                if (
                    plot_lc_vs_d0
                    and iptjet == 1
                    and self.var == "zpar"
                    and self.species == "Lc"
                    and plot_run2_d0_ff_data
                ):
                    self.list_latex = [
                        self.text_alice,
                        self.text_jets.replace(self.latex_hadron, "HF"),
                        f"{self.get_text_range_ptjet(iptjet)}, {self.text_etajet}",
                        self.get_text_range_pthf(-1, iptjet).replace(self.latex_hadron, "HF"),
                    ]
                    run2_d0_ff_data = self.get_run2_d0_ff_data()
                    run2_lc_ff_sim = self.get_run2_lc_ff_sim()
                    run2_lc_ff_sim["monash"].SetLineStyle(self.l_monash)
                    run2_lc_ff_sim["cr2"].SetLineStyle(self.l_mode2)
                    run2_d0_ff_sim = self.get_run2_d0_ff_sim()
                    run2_d0_ff_sim["monash"].SetLineStyle(self.l_monash + 2)
                    run2_d0_ff_sim["cr2"].SetLineStyle(self.l_mode2 + 2)
                    self.list_obj = [gr_syst, run2_d0_ff_data["syst"], h_stat, run2_d0_ff_data["stat"]]
                    # self.plot_order = list(range(len(self.list_obj)))
                    self.plot_order = [1, 0, 3, 2]
                    self.labels_obj = [
                        f"#Lambda_{{c}}^{{#plus}}, {self.text_run3}",
                        f"D^{{0}}, {self.text_run2}",
                        "",
                        "",
                    ]
                    self.list_colours = [get_colour(i) for i in (0, -1)] * 2
                    self.list_markers = [get_marker(i) for i in (0, -1)] * 2
                    self.leg_pos = [0.65, 0.0, 0.97, 0.75]
                    self.leg_horizontal = False
                    name_can = f"{self.species}_results_Lc-D0_{self.var}_{self.mcordata}_{string_ptjet}"
                    can = TCanvas(name_can, name_can)
                    pad_heights = self.set_pad_heights(can, [2, 1])
                    can, new = self.make_plot(
                        name_can,
                        can=can,
                        pad=1,
                        scale=pad_heights[0],
                        colours=self.list_colours,
                        markers=self.list_markers,
                    )
                    # ratio Lc/D0 bottom panel
                    rat_stat = divide_histograms(h_stat, run2_d0_ff_data["stat"])
                    rat_syst = divide_graphs(gr_syst, run2_d0_ff_data["syst"])
                    rat_monash = divide_histograms(run2_lc_ff_sim["monash"], run2_d0_ff_sim["monash"])
                    rat_cr2 = divide_histograms(run2_lc_ff_sim["cr2"], run2_d0_ff_sim["cr2"])
                    rat_monash.SetLineStyle(self.l_monash)
                    rat_cr2.SetLineStyle(self.l_mode2)
                    self.list_obj = [rat_syst, rat_stat, rat_monash, rat_cr2, line_1]
                    self.plot_order = list(range(len(self.list_obj)))
                    self.labels_obj = [
                        "data",
                        "",
                        f"{self.text_monash}, {self.text_run2}",
                        f"{self.text_mode2}, {self.text_run2}",
                    ]
                    self.list_colours = [get_colour(i) for i in (0, 0, self.c_lc_monash, self.c_lc_mode2)]
                    self.list_markers = [get_marker(0)] * 2 + [1, 1]
                    self.opt_plot_h = [self.opt_plot_h] + 2 * ["hist e"]
                    self.opt_leg_h = [self.opt_leg_h] + 2 * ["L"]
                    self.leg_horizontal = True
                    # self.scale_text_leg = 0.7
                    self.list_latex = []
                    self.title_full = f";{self.latex_obs};#Lambda_{{c}}^{{#plus}}/D^{{0}}"
                    can, new = self.make_plot(
                        name_can,
                        can=can,
                        pad=2,
                        scale=pad_heights[1],
                        colours=self.list_colours,
                        markers=self.list_markers,
                    )

                # Reset defaults.
                self.plot_order = self.plot_order_default
                self.opt_plot_h = ""
                self.opt_leg_h = "P"
                self.leg_pos = self.leg_pos_default
                self.scale_text_leg = self.scale_text_leg_default
                self.leg_horizontal = self.leg_horizontal_default

            self.logger.info("Plotting results for all pt jet together")
            self.plot_errors_x = False
            self.list_latex = [
                self.text_alice,
                f"{self.text_tagged} {self.text_jets}",
                f"{self.get_text_range_pthf(-1, iptjet)}, {self.text_etajet}",
            ]
            if not plot_run2_data:
                self.list_latex[0] = f"{self.text_alice}, {self.text_run3}"
            if self.var in ("zg", "rg", "nsd"):
                self.list_latex.append(self.text_sd)
                # self.list_latex.append(self.text_ptcut)
            self.leg_horizontal = True
            self.range_x = x_range[self.var]
            self.list_obj = list_syst_all + list_stat_all
            self.labels_obj = list_labels_all + [""] * len(list_stat_all)  # do not show the histograms in the legend
            self.list_colours = list_colours_syst_all + list_colours_stat_all
            self.list_markers = list_markers_all * (1 + int(bool(list_syst_all)))
            if plot_run2_data:
                h_run2, g_run2 = None, None
                if plot_run2_d0_sd and self.species == "D0" and self.var in ("zg", "rg", "nsd"):
                    h_run2 = run2_d0_sd[self.var]["hf"]["data"]["stat"]
                    g_run2 = run2_d0_sd[self.var]["hf"]["data"]["syst"]
                    if self.var == "nsd":
                        shrink_err_x(g_run2)
                # TODO: if plot_run2_d0_ff_data
                # TODO: if plot_run2_lc_ff_data
                if h_run2 is not None:
                    n_obj = len(self.list_obj)
                    self.plot_order = list(range(n_obj)) + [-1, -0.5]
                    self.list_obj += [g_run2, h_run2]
                    self.labels_obj += [f"{self.text_run2}, {self.get_text_range_ptjet(2)}", ""]
                    self.list_colours += [get_colour(-1)] * 2
                    self.list_markers += [get_marker(-1)] * 2
                    self.leg_horizontal = True
                    self.leg_pos = [0.52, 0.65, 0.85, 0.73]
                    self.y_margin_up = 0.04
            self.title_full = self.title_full_default
            name_can = f"{self.species}_results_{self.var}_{self.mcordata}_ptjet-all"
            can = TCanvas(name_can, name_can)
            pad_heights = self.set_pad_heights(can, [3, 1])
            can, new = self.make_plot(
                name_can, can=can, pad=1, scale=pad_heights[0], colours=self.list_colours, markers=self.list_markers
            )
            self.plot_order = self.plot_order_default
            # ratio low-pt/high-pt bottom panel
            iptjet_ref = list_iptjet[-1]  # reference pt jet bin
            assert iptjet_ref in list_iptjet
            i_iptjet_ref = list_iptjet.index(iptjet_ref)
            list_ratio_stat, list_ratio_syst = make_ratios(list_stat_all, list_syst_all, i_iptjet_ref, False)
            self.list_obj = list_ratio_syst + list_ratio_stat + [line_1]
            self.labels_obj = []
            self.list_latex = []
            self.y_margin_up = 0.06  # to fix cropped number on the axis
            self.title_full = f";{self.latex_obs};ratio to    "
            can, new = self.make_plot(
                name_can, can=can, pad=2, scale=pad_heights[1], colours=self.list_colours, markers=self.list_markers
            )
            width = self.y_step_glob * 2.5
            leg = TLegend(
                self.margins_can[1] / 2 - width / 3,
                1.0 - self.y_step_glob / pad_heights[1],
                self.margins_can[1] / 2 + width / 1,
                1.0,
            )
            setup_legend(leg)
            leg.AddEntry(list_syst_all[-1], " ", "FP")
            can.cd(2)
            leg.Draw()
            gStyle.SetErrorX(0.5)  # reset default width
            if not self.plot_errors_x:
                gStyle.SetErrorX(0)  # do not plot horizontal error bars of histograms
            self.save_canvas(can)
            gStyle.SetErrorX(0.5)  # reset default width

            # Reset defaults.
            self.plot_order = self.plot_order_default
            self.opt_plot_h = ""
            self.opt_leg_h = "P"
            self.leg_pos = self.leg_pos_default
            self.leg_horizontal = self.leg_horizontal_default
            self.y_margin_up = self.y_margin_up_default

            # Lc vs D0
            if plot_lc_vs_d0 and self.species == "Lc" and self.var == "zpar":
                self.logger.info("Plotting Lc vs D0")
                self.plot_errors_x = False
                self.list_latex = [
                    self.text_alice,
                    self.text_jets,
                    f"{self.get_text_range_pthf(-1, iptjet)}, {self.text_etajet}",
                ]
                if self.var in ("zg", "rg", "nsd"):
                    self.list_latex.append(self.text_sd)
                    # self.list_latex.append(self.text_ptcut)
                self.leg_horizontal = True
                self.range_x = x_range[self.var]
                # Get D0 results
                path_input_file_d0 = self.path_input_file.replace("lc", "d0")
                path_syst_d0 = path_syst.replace("lc", "d0")
                with TFile.Open(path_input_file_d0) as file_results_d0, TFile.Open(path_syst_d0) as file_syst_d0:
                    names_his = []
                    names_syst = []
                    for i_iptjet, iptjet in enumerate(list_iptjet):
                        range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                        string_ptjet = string_range_ptjet(range_ptjet)
                        name_his = f"h_{self.var}_{self.method}_unfolded_{self.mcordata}_{string_ptjet}_sel_selfnorm"
                        names_his.append(name_his)
                        names_syst.append(f"sys_{self.var}_{string_ptjet}")
                    list_stat_all_d0 = [self.get_object(name_his, file_results_d0) for name_his in names_his]
                    list_syst_all_d0 = [self.get_object(name_syst, file_syst_d0) for name_syst in names_syst]
                if self.var == "nsd":
                    for gr_syst in list_syst_all_d0:
                        shrink_err_x(gr_syst)

                # To fix?
                self.list_obj = list_syst_all + list_syst_all_d0 + list_stat_all + list_stat_all_d0
                self.labels_obj = list_labels_all + list_labels_all
                self.list_colours = (
                    list_colours_syst_all + list_colours_stat_all + list_colours_syst_all + list_colours_stat_all
                )
                self.list_markers = list_markers_all * 2 * (1 + int(bool(list_syst_all)))
                self.title_full = self.title_full_default

                name_can = f"{self.species}_results_Lc-D0_{self.var}_{self.mcordata}_ptjet-all"
                self.make_plot(name_can, colours=self.list_colours, markers=self.list_markers)

            self.plot_errors_x = True


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-analysis", "-d", dest="database_analysis", help="analysis database to be used", required=True
    )
    parser.add_argument("--analysis", "-a", dest="type_ana", help="choose type of analysis", required=True)
    parser.add_argument("--input", "-i", dest="input_file", help="results input file")

    args = parser.parse_args()

    gROOT.SetBatch(True)

    list_vars = ["zg", "nsd", "rg", "zpar"]
    # list_vars = ["zpar"]
    for var in list_vars:
        print(f"Processing observable {var}")
        # for mcordata in ("data", "mc"):
        for mcordata in ["data"]:
            plotter = Plotter(args.input_file, args.database_analysis, args.type_ana, var, mcordata)
            plotter.plot()


if __name__ == "__main__":
    main()
