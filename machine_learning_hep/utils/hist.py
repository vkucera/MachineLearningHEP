from collections import deque
import itertools

import numpy as np
import pandas as pd
import ROOT


def bin_array(nbins, low, high):
    return np.linspace(float(low), float(high), nbins + 1, 'd')


def get_axis(hist, axis: int):
    if isinstance(hist, ROOT.THn):
        return hist.GetAxis(axis)

    if isinstance(hist, ROOT.TH1):
        if axis == 0:
            return hist.GetXaxis()
        if axis == 1:
            return hist.GetYaxis()
        if axis == 2:
            return hist.GetZaxis()
        raise ValueError

    raise NotImplementedError


def get_dim(hist):
    return hist.GetDimension() if isinstance(hist, ROOT.TH1) else hist.GetNdimensions()


def get_range(hist, axis: int):
    axis = get_axis(hist, axis)
    return (axis.GetFirst(), axis.GetLast())


def get_nbins(hist, axis:int):
    return get_axis(hist, axis).GetNbins()


# pylint: disable=too-many-branches
def project_hist(hist, axes: list, limits: dict[int, tuple[int]]):
    # TODO: add consistent suffix for projections
    # TODO: add option for uncertainties ???
    if len(axes) == 2:
        axes = axes[:] # slice to avoid modifying the list passed as parameter
        axes.reverse() # compensation for ROOT signature using ydim, xdim for 2d projection
    reset = False
    if isinstance(hist, ROOT.THn):
        assert len(axes) < hist.GetNdimensions()
        ranges = []
        for iaxis in range(hist.GetNdimensions()):
            ranges.append(get_range(hist, iaxis))
        for iaxis, bins in limits.items():
            get_axis(hist, iaxis).SetRange(bins[0], bins[1])
            if bins[0] == 1 and bins[1] == get_nbins(hist, iaxis):
                get_axis(hist, iaxis).SetBit(ROOT.TAxis.kAxisRange)
            reset |= bins[1] < bins[0]
        hproj = hist.Projection(*axes, 'e') if len(axes) < 4 else hist.Projection(len(axes), np.asarray(axes, 'i'), 'e')
        for iaxis in limits:
            get_axis(hist, iaxis).SetRange(*ranges[iaxis])
        if reset:
            hproj.Reset()
        return hproj

    if isinstance(hist, ROOT.TH3):
        assert len(axes) < hist.GetDimension()
        ranges = []
        for iaxis in range(hist.GetDimension()):
            ranges.append(get_range(hist, iaxis))
            if iaxis in limits:
                bins = limits[iaxis]
                get_axis(hist, iaxis).SetRange(bins[0], bins[1])
                if bins[0] == 1 and bins[1] == get_nbins(hist, iaxis):
                    get_axis(hist, iaxis).SetBit(ROOT.TAxis.kAxisRange)
                reset |= bins[1] < bins[0]
        proj_spec = ""
        for axis in axes:
            proj_spec += ('x' if axis == 0 else 'y' if axis == 1 else 'z')
        hproj = hist.Project3D(proj_spec)
        for iaxis in limits:
            get_axis(hist, iaxis).SetRange(*ranges[iaxis])
        if reset:
            hproj.Reset()
        return hproj

    if isinstance(hist, ROOT.TH2):
        assert len(axes) == 1
        if axes[0] == 0:
            bins = limits.get(1, [])
            reset = len(bins) == 2 and bins[1] < bins[0]
            hproj = hist.ProjectionX("_px", *bins)
            if reset:
                hproj.Reset()
            return hproj
        if axes[0] == 1:
            bins = limits.get(0, [])
            reset = len(bins) == 2 and bins[1] < bins[0]
            hproj = hist.ProjectionY("_py", *bins)
            if reset:
                hproj.Reset()
            return hproj
        raise ValueError

    raise NotImplementedError


def create_hist(name, title, *bin_specs):
    """Create ROOT histogram from standard bin specifications or arrays"""
    rhist = {1: ROOT.TH1F, 2: ROOT.TH2F, 3: ROOT.TH3F, 4: ROOT.THnF}
    var_bins = [hasattr(spec, '__len__') for spec in bin_specs]
    assert all(var_bins) or not any(var_bins), f'either all bins must be variable or fixed width: {bin_specs=}'
    dim = len(bin_specs) if all(var_bins) else len(bin_specs) / 3
    assert dim in range(1, 10), 'only dimensions from 1 to 10 are supported'

    if all(var_bins):
        nbins = list(map(lambda a: len(a) - 1, bin_specs))

    if dim <= 3:
        if all(var_bins):
            bin_defs = zip(nbins, bin_specs)
            bin_specs = [arg for axis in bin_defs for arg in axis]
        return rhist[min(dim, 4)](name, title, *bin_specs)

    if all(var_bins):
        nbins = np.asarray(nbins, 'i')
        return rhist[min(dim, 4)](name, title, dim, nbins, bin_specs)

    raise NotImplementedError

# TODO: generalize which columns can contain arrays
# pylint: disable=too-many-branches
def fill_hist(hist, dfi: pd.DataFrame, weights = None, arraycols = None, write = False):
    """
    Fill histogram from dataframe

    :param hist: ROOT.TH1,2,3
    :param dfi: pandas series/dataframe (1 to 3 columns)
    :param weights: weights per row
    :param array: dataframe contains arrays
    :param write: call Write() after filling
    """
    dim_hist = hist.GetDimension() if isinstance(hist, ROOT.TH1) else hist.GetNdimensions()
    dim_df = dfi.shape[1] if dfi.ndim > 1 else dfi.ndim
    assert dim_df in range(1, 10), f'{dim_df} not supported'
    assert dim_df == dim_hist, 'dimensions of df and histogram do not match'
    if len(dfi) == 0:
        return
    if dim_hist == 1:
        if not arraycols:
            hist.FillN(len(dfi), np.float64(dfi), weights or ROOT.nullptr)
        else:
            dfi.apply(lambda row: [hist.Fill(v) for v in row])
    elif dim_hist == 2:
        if not arraycols:
            hist.FillN(len(dfi), np.float64(dfi.iloc[:, 0]), np.float64(dfi.iloc[:, 1]), weights or ROOT.nullptr)
        else:
            assert weights is None, 'weights not supported'
            dfi.apply(lambda row: [hist.Fill(row.iloc[0], v) for v in row.iloc[1]], axis=1)
    elif dim_hist == 3:
        # TODO: why does TH3 not support FillN?
        # hist.FillN(len(dfi), np.float64(dfi.iloc[:, 0]), np.float64(dfi.iloc[:, 1]), np.float64(dfi.iloc[:, 2]),
        #            weights or np.float64(len(dfi)*[1.]))
        assert weights is None, 'weights not supported'
        if not arraycols:
            dfi.apply(lambda row: hist.Fill(row.iloc[0], row.iloc[1], row.iloc[2]), axis=1)
        else:
            assert arraycols == [1, 2], 'other cases not yet implemented'
            dfi.apply(lambda row: [hist.Fill(row.iloc[0], v[0], v[1])
                                   for v in zip(row.iloc[i] for i in arraycols)], axis=1)
    elif dim_hist > 3:
        assert weights is None, 'weights not supported'
        if not arraycols:
            dfi.apply(lambda row: hist.Fill(*row), axis=1)
        else:
            m = [-1] * dim_hist
            idx = 0
            for i in arraycols:
                m[i] = idx
                idx += 1
            def fill_row(row):
                # for v in zip(*[row.iloc[i] for i in arraycols]):
                #     hist.Fill(np.asarray([row.iloc[i] if i not in arraycols else v[m[i]]
                #                           for i in range(dim_hist)], 'd'))
                gen = (hist.Fill(np.asarray([row.iloc[i] if i not in arraycols else v[m[i]]
                                             for i in range(dim_hist)], 'd'))
                       for v in zip(*[row.iloc[i] for i in arraycols]))
                deque(gen, maxlen=0)
            dfi.apply(fill_row, axis=1)
    if write:
        hist.Write()


def fill_hist_fast(hist, dfi, write=False):
    """
    Fill histogram by setting bin instead of going through Fill(...)
    aor every row (which is slow).
    """
    dim_hist = hist.GetDimension() if isinstance(hist, ROOT.TH1) else hist.GetNdimensions()
    dim_df = dfi.shape[1] if dfi.ndim > 1 else dfi.ndim
    assert dim_df in range(3, 4), f'{dim_df} not supported'
    assert dim_df == dim_hist, 'dimensions of df and histogram do not match'
    bin_it = [range(get_nbins(hist, i)+2) for i in range(get_dim(hist))]
    for binids in itertools.product(*bin_it):
        df = dfi
        for i in range(get_dim(hist)):
            if binids[i] == 0:
                # underflow
                df = df.loc[df.iloc[:, i] <  get_axis(hist, i).GetXmin()]
            elif binids[i] == (get_nbins(hist, i) + 1):
                # overflow
                df = df.loc[df.iloc[:, i] >= get_axis(hist, i).GetXmax()]
            else:
                df = df.loc[(df.iloc[:, i] >= get_axis(hist, i).GetBinLowEdge(binids[i])) &
                            (df.iloc[:, i] <  get_axis(hist, i).GetBinUpEdge(binids[i]))]
        hist.SetBinContent(*binids, len(df))
    if write:
        hist.Write()


def scale_bin(hist, factor, *bin_indices):
    """Scale histogram bin-wise by given factor"""
    hist.SetBinContent(*bin_indices, hist.GetBinContent(*bin_indices) * factor)
    hist.SetBinError(*bin_indices, hist.GetBinError(*bin_indices) * factor)


def sum_hists(hists, name = None):
    """
    Return histogram with sum of all histograms from iterable
    """
    hist = None
    for h in hists:
        if h is None:
            continue
        if hist is None:
            hist = h.Clone(name or (h.GetName() + '_cloned'))
        else:
            hist.Add(h)
    return hist


def ensure_sumw2(hist):
    if hist.GetSumw2N() < 1:
        hist.Sumw2()
