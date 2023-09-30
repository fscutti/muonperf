import os
import math

import numpy as np
import ROOT as R

from copy import copy
from tqdm import tqdm
from astropy.io import fits


def check_path(path):
    """Creates a path if it does not exists."""
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)


def save_as_root(output_path, output_file, hist_list):
    """Saves list of histograms inside a root file."""
    check_path(output_path)
    output_ptr = R.TFile.Open(os.path.join(output_path, output_file), "RECREATE")
    for h in hist_list:
        output_ptr.WriteObject(h, h.GetName())


def save_as_fits(output_path, hist_list):
    """Saves list of histograms as fits files."""
    check_path(output_path)
    for h in hist_list:
        save_to_fits(h, output_path)


def get_files(path):
    """Get file list inside a path."""
    file_list = []
    for dir_path, dir_names, file_names in os.walk(path):
        file_list.extend(file_names)
        break
    return file_list


def get_histograms(path, file_list, hist_name, new_hist_name=None):
    """Retrieve an histogram name from a list of files."""
    hists = {}
    for file_name in file_list:
        f = os.path.join(path, file_name)
        f_ptr = R.TFile.Open(f)
        hists[file_name] = copy(f_ptr.Get(hist_name))
        if new_hist_name is not None:
            hists[file_name].SetNameTitle(new_hist_name, new_hist_name)
    return hists


def bin_loop(hist):
    """Generator for looping over 2D histogram bins."""
    n_bins_x, n_bins_y = hist.GetNbinsX(), hist.GetNbinsY()

    for x_idx in range(1, n_bins_x + 1):
        for y_idx in range(1, n_bins_y + 1):
            yield hist.GetBin(x_idx, y_idx), x_idx, y_idx


def clone(hist, new_tag=None, new_name=None):
    """Cloning histogram with new name and new title."""
    h_new = hist.Clone()

    if new_tag is not None:
        h_new_name = "_".join([hist.GetName(), new_tag])

    elif new_name is not None:
        h_new_name = "_".join(["h", new_name])

    h_new.SetNameTitle(h_new_name, h_new_name)
    return h_new


def get_uncertainty(hist, time=1):
    """Returns an assiciated histogram containing uncertainties.
    Time units are in seconds."""

    h_unc = clone(hist, new_tag="poisson_err")

    for glob_idx, _, _ in bin_loop(hist):
        rate = hist.GetBinContent(glob_idx)
        uncertainty = math.sqrt(rate / time)
        h_unc.SetBinContent(glob_idx, uncertainty)

    return h_unc


def get_randomisation(hist_nom, hist_unc):
    """Get randomised histogram."""

    h_rand = clone(hist_nom, new_tag="rand")

    for glob_idx, _, _ in bin_loop(h_rand):
        mu = hist_nom.GetBinContent(glob_idx)
        sigma = hist_unc.GetBinContent(glob_idx)

        rand = np.random.normal(mu, sigma)
        # If the rate is negative, this means that the acquisition
        # time is not large enough to make a sensible measurement.
        # Then the rate is assumed to be zero. This scenario has
        # to be catched by the opacity interpolation algorithm which
        # will assign zero as opacity and its uncertainty.
        if rand < 0.0:
            rand = 0

        h_rand.SetBinContent(glob_idx, rand)

    return h_rand


def get_opacity_interpolation(hist_flux, hist_flux_unc, lookup_file):
    """This function returns an interpolated opacity histogram."""

    h_opacity = clone(hist_flux, new_name="interp_opacity")
    h_opacity_err = clone(hist_flux, new_name="interp_opacity_err")

    hist_lookup = {}
    _hist_lookup = {}

    lookup_ptr = R.TFile.Open(lookup_file)

    for h in lookup_ptr.GetListOfKeys():
        histogram = copy(lookup_ptr.Get(h.GetName()))

        if not "hp_opacity" in histogram.GetName():
            continue

        hist_elevation = int(histogram.GetName().split("_")[-1])

        _hist_lookup[hist_elevation] = histogram

    for el in sorted(_hist_lookup.keys()):
        hist_lookup[el] = _hist_lookup[el]

    lookup_ptr.Close()

    n_bins_tot = hist_flux.GetNbinsX() * hist_flux.GetNbinsY()

    for glob_idx, x_idx, y_idx in tqdm(
        bin_loop(hist_flux), total=n_bins_tot, colour="red"
    ):
        flux = hist_flux.GetBinContent(glob_idx)
        flux_err = hist_flux_unc.GetBinContent(glob_idx)

        elevation = hist_flux.GetYaxis().GetBinLowEdge(y_idx)
        elevation += hist_flux.GetYaxis().GetBinUpEdge(y_idx)
        elevation /= 2.0

        op, op_err = interpolate_opacity(flux, flux_err, elevation, hist_lookup)

        h_opacity.SetBinContent(glob_idx, op)
        h_opacity_err.SetBinContent(glob_idx, op_err)

    return h_opacity, h_opacity_err


def opacity_uncertainty(
    rate, sigma_rate, r1, sigma_r1, r2, sigma_r2, o1, sigma_o1, o2, sigma_o2
):
    """Returns the uncertainty on the interpolated opacity."""
    sigma = (((o2 - o1) / (r2 - r1)) ** 2) * (sigma_rate**2)
    sigma += (((o2 - o1) / (r2 - r1)) ** 2) * (sigma_r1**2)
    sigma += ((1 - (rate - r1) / (r2 - r1)) ** 2) * (sigma_o1**2)
    sigma += (((rate - r1) / (r2 - r1)) ** 2) * (sigma_o2**2)
    sigma += (((rate - r1) * (o2 - o1) / ((r2 - r1) ** 2)) ** 2) * (sigma_r2**2)

    return math.sqrt(sigma)


def choose_interpolation_hist(elevation, lookup_hists):
    """Chooses a lookup function among those available in the file."""
    chosen_elevation = min(lookup_hists, key=lambda e: abs(e - elevation))
    return lookup_hists[chosen_elevation]


def interpolate_opacity(rate, sigma_rate, elevation, lookup_hists):
    """Perform opacity interpolation based on lookup functions."""
    opacity = 0.0
    opacity_error = 0.0

    if rate > 0.0 and elevation < 87:
        hist = choose_interpolation_hist(elevation, lookup_hists)

        # If the interpolation fails the function will return the follwing values.
        opacity = max(
            1,
            np.random.uniform(
                hist.GetXaxis().GetBinCenter(1),
                hist.GetXaxis().GetBinCenter(1) + hist.GetXaxis().GetBinWidth(1) / 2.0,
            ),
        )
        opacity_error = hist.GetXaxis().GetBinWidth(1) / 2.0

        for bin_idx in range(1, hist.GetNbinsX() + 1):
            r1 = hist.GetBinContent(bin_idx)

            # find valid value of next bin
            for next_idx in range(bin_idx + 1, hist.GetNbinsX() + 1):
                r2 = hist.GetBinContent(next_idx)

                if r2 > 0.0:
                    break

            if rate < r1 and rate >= r2:
                o1 = hist.GetXaxis().GetBinCenter(bin_idx)
                o2 = hist.GetXaxis().GetBinCenter(next_idx)

                sigma_o1 = hist.GetXaxis().GetBinWidth(bin_idx) / 2.0
                sigma_o2 = hist.GetXaxis().GetBinWidth(next_idx) / 2.0

                sigma_r1 = hist.GetBinError(bin_idx)
                sigma_r2 = hist.GetBinError(next_idx)

                # use linear interpolation
                opacity = o1 + (rate - r1) * (o2 - o1) / (r2 - r1)

                opacity_error = opacity_uncertainty(
                    rate,
                    sigma_rate,
                    r1,
                    sigma_r1,
                    r2,
                    sigma_r2,
                    o1,
                    sigma_o1,
                    o2,
                    sigma_o2,
                )
                break

    return opacity, opacity_error


def save_to_fits(hist, out_path):
    """Writes 2d histogram to fits file."""

    out_file = hist.GetName() + ".fits"
    out_file = os.path.join(out_path, out_file)

    # When grabbing the Y axis we still have to call the Xmax/Xmin.
    # ROOT is strange...
    x_min = hist.GetXaxis().GetXmin()
    y_min = hist.GetYaxis().GetXmin()

    x_max = hist.GetXaxis().GetXmax()
    y_max = hist.GetYaxis().GetXmax()

    x_n_bins = hist.GetNbinsX()
    y_n_bins = hist.GetNbinsY()

    x_axis_name = hist.GetXaxis().GetTitle()
    y_axis_name = hist.GetYaxis().GetTitle()

    image = np.empty(shape=(x_n_bins, y_n_bins))

    for x_idx in range(image.shape[0]):
        for y_idx in range(image.shape[1]):
            image[x_idx, y_idx] = hist.GetBinContent(x_idx + 1, y_idx + 1)

    hdu = fits.PrimaryHDU(image)

    hdu.header[
        "X"
    ] = f"variable: {x_axis_name}, n_bins: {x_n_bins}, min: {x_min}, max: {x_max}"
    hdu.header[
        "Y"
    ] = f"variable: {y_axis_name}, n_bins: {y_n_bins}, min: {y_min}, max: {y_max}"

    hdu.writeto(out_file, overwrite=True)
