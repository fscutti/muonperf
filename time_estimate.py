import sys
import os
import math

from copy import copy

import numpy as np
from scipy import stats
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import ROOT
import cmasher

from pyrate.utils import ROOT_utils as RU


def diabolical_flip(hist):
    flipped_hist = copy(hist)
    flipped_hist.Reset()
    flipped_hist.SetNameTitle(
        hist.GetName() + "_" + "flipped", hist.GetName() + "_" + "flipped"
    )

    for g_idx, az_idx, el_idx in RU.bin_loop(hist):
        az = hist.GetXaxis().GetBinCenter(az_idx)
        flipped_az = (90 - az) % 360

        new_az_idx = flipped_hist.GetXaxis().FindBin(flipped_az)

        flipped_hist.SetBinContent(new_az_idx, el_idx, hist.GetBinContent(g_idx))

    return flipped_hist


def make_mpl_plot(
    hist,
    cmap="cmr.torch",
    savefig=None,
    log_scale=False,
    title="",
    units="",
    z_min=None,
    z_max=None,
    draw_grid=False,
    flip=False,
    colorbar=True,
):
    directory = os.path.dirname(savefig)

    if flip:
        hist = copy(diabolical_flip(hist))

    if not os.path.exists(directory):
        os.makedirs(directory)

    n_az_bins = hist.GetXaxis().GetNbins()
    n_el_bins = hist.GetYaxis().GetNbins()

    data = np.empty([n_az_bins, n_el_bins])

    for g_idx, az_idx, el_idx in RU.bin_loop(hist):
        data[az_idx - 1, el_idx - 1] = hist.GetBinContent(g_idx)

    az_min = hist.GetXaxis().GetXmin()
    az_max = hist.GetXaxis().GetXmax()
    az_bin_width = round((az_max - az_min) / n_az_bins)

    el_min = hist.GetYaxis().GetXmin()
    el_max = hist.GetYaxis().GetXmax()
    el_bin_width = round((el_max - el_min) / n_el_bins)

    # Convert angles to suitable format for a polar plot.
    # az = np.arange(az_min, az_max + az_bin_width) * (2 * np.pi / 360)
    # el = np.arange(el_min, el_max + el_bin_width)

    az = np.arange(az_min, az_max + az_bin_width, az_bin_width) * (2 * np.pi / 360)
    el = np.arange(el_min, el_max + el_bin_width, el_bin_width)

    # Determine the tick labels for the elevation angles.
    el_axis = np.abs(el - 90)
    el_axis_spc = np.ceil(abs(el_axis[-1] - el_axis[0]) / 5).astype(int)
    el_label = el_axis[::el_axis_spc]

    # Create an angle meshgrid
    Az, El = np.meshgrid(az, el_axis)

    # Create new figure
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(projection="polar")

    # Set properties of this polar plot to mimic azimuth and elevation
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_thetamin(az_min)
    ax.set_thetamax(az_max)
    ax.set_rgrids(el_label, [r"$%s\degree$" % (abs(90 - x)) for x in el_label])

    cmap = plt.get_cmap(cmap)

    pc = None

    # Create colormesh of flux values
    if log_scale:
        pc = ax.pcolormesh(
            Az,
            El,
            data.T,
            cmap=cmap,
            shading="flat",
            norm=colors.LogNorm(vmin=z_min, vmax=z_max),
        )

    else:
        pc = ax.pcolormesh(
            Az,
            El,
            data.T,
            cmap=cmap,
            shading="flat",
            vmin=z_min,
            vmax=z_max,
        )
    
    if colorbar:
        cbar = fig.colorbar(pc)
        cbar.set_label(units)

    plt.title(title)

    plt.tight_layout()

    if not draw_grid:
        plt.grid()

    # If savefig is not None, save the figure
    if savefig is not None:
        plt.savefig(savefig, dpi=500)
        plt.close(fig)

    # Else, simply show it
    else:
        plt.show()


def get_counts_hist(file_path, time=1, surface=1):
    flux_file = ROOT.TFile.Open(file_path, "READ")
    flux_hist = flux_file.Get("h_flux")

    counts_hist = copy(flux_hist)
    counts_hist.Reset()
    counts_hist.SetNameTitle("h_counts", "h_counts")

    # Normalisation constants.
    deg = np.pi / 180.0
    energy_factor = pow(10, 4) - pow(10, -3)

    surface_factor = surface
    time_factor = time

    for g_idx, az_idx, el_idx in RU.bin_loop(flux_hist):
        # In the flux histogram el=y, az=x.
        az = flux_hist.GetXaxis().GetBinCenter(az_idx)
        el = flux_hist.GetYaxis().GetBinCenter(el_idx)

        az_up = flux_hist.GetXaxis().GetBinUpEdge(az_idx)
        az_low = flux_hist.GetXaxis().GetBinLowEdge(az_idx)

        el_up = flux_hist.GetYaxis().GetBinUpEdge(el_idx)
        el_low = flux_hist.GetYaxis().GetBinLowEdge(el_idx)

        flux = flux_hist.GetBinContent(g_idx)

        n_muons = flux
        n_muons *= energy_factor
        n_muons *= surface_factor
        n_muons *= time_factor
        n_muons *= (
            deg * np.fabs(np.sin(el_up * deg) - np.sin(el_low * deg)) * (az_up - az_low)
        )

        counts_hist.SetBinContent(g_idx, n_muons)
        counts_hist.SetBinError(g_idx, math.sqrt(n_muons))

    return counts_hist


def get_acceptance_counts_hist(counts_hist, file_path):
    """The total number of entries in the returned histogram is the number
    of muons one expects to see."""

    acc_counts_hist = copy(counts_hist)
    acc_counts_hist.Reset()
    acc_counts_hist.SetNameTitle("h_acc_counts", "h_acc_counts")

    acceptance_file = ROOT.TFile.Open(file_path, "READ")
    acceptance_hist = acceptance_file.Get("p2_ang_acc_vs_true")

    for g_idx, az_idx, el_idx in RU.bin_loop(counts_hist):
        # In the flux histogram el=y, az=x.
        az = counts_hist.GetXaxis().GetBinCenter(az_idx)
        el = counts_hist.GetYaxis().GetBinCenter(el_idx)

        # In the acceptance histogram el=x, az=y.
        acc_g_idx = acceptance_hist.FindBin(90.0 - el, az)
        acceptance = acceptance_hist.GetBinContent(acc_g_idx)

        n_muons = counts_hist.GetBinContent(g_idx)
        n_acc_muons = n_muons * acceptance
        acc_counts_hist.SetBinContent(g_idx, n_acc_muons)
        acc_counts_hist.SetBinError(g_idx, math.sqrt(n_acc_muons))

    return acc_counts_hist


def get_stats_test_hist(
    nom_hist, alt_hist, time_unit_in_seconds=1, surface=1, n_detectors=1, z_p=1
):
    stat_hist = copy(nom_hist)
    stat_hist.Reset()
    stat_hist.SetNameTitle("h_stat_test", "h_stat_test")

    for g_idx, az_idx, el_idx in RU.bin_loop(nom_hist):
        s = nom_hist.GetBinContent(g_idx)

        estimated_time = sys.maxsize

        if s > 0:
            b = alt_hist.GetBinContent(g_idx)
            e = math.sqrt(s)

            C = (s - b) / e

            C *= math.sqrt(surface)
            C *= math.sqrt(n_detectors)

            estimated_time = (z_p / C) ** 2
            estimated_time /= time_unit_in_seconds

        stat_hist.SetBinContent(g_idx, estimated_time)

    return stat_hist


time_units = {}
time_units["days"] = 86400
# time_units["hours"] = 3600
# time_units["minutes"] = 60
# time_units["seconds"] = 1

confidence_levels = {}
# confidence_levels["TwoSidedTest 95% CL"] = stats.norm.ppf(1 - 0.025)
confidence_levels["OneSidedTest 95% CL"] = stats.norm.ppf(1 - 0.05)
confidence_levels["OneSigmaTest"] = 1

detector_surfaces = {}
detector_surfaces["Side 15cm"] = 0.15 * 0.15
detector_surfaces["Side 20cm"] = 0.2 * 0.2

number_of_detectors = {}
number_of_detectors["Ndet_1"] = 1
number_of_detectors["Ndet_10"] = 10
number_of_detectors["Ndet_20"] = 20
number_of_detectors["Ndet_60"] = 60
number_of_detectors["Ndet_100"] = 100

rebin = {}
rebin["Az 1deg El 1deg"] = (1, 1)
rebin["Az 2deg El 5deg"] = (2, 5)
rebin["Az 5deg El 5deg"] = (5, 5)
#rebin["Az 5deg El 9deg"] = (5, 9)
#rebin["Az 10deg El 9deg"] = (10, 9)
rebin["Az 10deg El 10deg"] = (10, 10)

acceptance_filtered = {}
#acceptance_filtered["AccFilter"] = True
acceptance_filtered["NoFilter"] = False

#nom_flux_file = "misc/sgm_nominal.root"

#nom_flux_file = "misc/costerfield_nominal.root"
#alt_flux_file = "misc/costerfield_novoids.root"

nom_flux_file = "misc/submarine_nominal.root"
alt_flux_file = "misc/submarine_nosub.root"

acceptance_file = "NewCosterfieldEffMap_5000000_5x5_Strip.root"

histograms = {}
for a_name, a_value in acceptance_filtered.items():
    for r_name, r_value in rebin.items():
        # Converting flux histograms to number of events.
        # Notice that we do not renormalise for time units or detector surface,
        # i.e. time units are still in seconds and detector area 1 sqm.
        h_nom = get_counts_hist(nom_flux_file, time=1, surface=1)
        h_alt = get_counts_hist(alt_flux_file, time=1, surface=1)

        if a_value:
            # Eventually filtering for acceptance.
            h_nom = get_acceptance_counts_hist(copy(h_nom), acceptance_file)
            h_alt = get_acceptance_counts_hist(copy(h_alt), acceptance_file)

        if r_value:
            # Eventually rebin the histogram.
            h_nom = copy(h_nom).Rebin2D(*r_value, h_nom.GetName() + " " + r_name)
            h_alt = copy(h_alt).Rebin2D(*r_value, h_alt.GetName() + " " + r_name)

        for n_name, n_value in number_of_detectors.items():
            for d_name, d_value in detector_surfaces.items():
                for c_name, c_value in confidence_levels.items():
                    for t_name, t_value in time_units.items():
                        # Getting the statistical test histogram.
                        h_test = get_stats_test_hist(
                            h_nom,
                            h_alt,
                            time_unit_in_seconds=t_value,
                            surface=d_value,
                            z_p=c_value,
                        )

                        h_name = []
                        h_name.append(a_name)
                        h_name.append(r_name)
                        h_name.append(n_name)
                        h_name.append(d_name)
                        h_name.append(c_name)
                        h_name.append(t_name)

                        h_name = "_".join(h_name).replace(" ", "_")

                        histograms[h_name] = h_test


#"""
#path = "CosterFieldStatTestLogNoFlip"
path = "SubmarineLogNoFlip"
for h_name, h_value in tqdm(histograms.items()):
    h_path = os.path.join(path, h_name)
    h_units = h_name.split("_")[-1]
    h_title = h_name.replace(f"_{h_units}", "")

    make_mpl_plot(
        h_value,
        savefig=h_path + "_log",
        log_scale=True,
        #title=h_title,
        units=h_units,
        z_min=1e-1,
        z_max=1e5,
        flip=False,
        colorbar=False,
    )

#path = "CosterFieldStatTestNoFlip"
path = "SubmarineNoFlip"
for h_name, h_value in tqdm(histograms.items()):
    h_path = os.path.join(path, h_name)
    h_units = h_name.split("_")[-1]
    h_title = h_name.replace(f"_{h_units}", "")

    make_mpl_plot(
        h_value,
        savefig=h_path,
        log_scale=False,
        #title=h_title,
        units=h_units,
        z_min=0,
        z_max=3000,
        flip=False,
        colorbar=False,
    )
#"""


#path = "SGMFluxes"
path = "CosterFieldFluxesNoFlip"
z_title = "$\mathrm{N_{\mu}/[GeV\;x\;s\;x\;sr\;x\;m^{2}]}$"

nom_flux_file_ptr = ROOT.TFile.Open(nom_flux_file, "READ")
nom_flux_hist = nom_flux_file_ptr.Get("h_flux")

# Nominal flux.
make_mpl_plot(
    nom_flux_hist,
    cmap="cmr.ocean",
    savefig=os.path.join(path, "nom_flux"),
    log_scale=False,
    title="Nominal Flux",
    units=z_title,
    draw_grid=True,
    flip=False,
)
#"""
alt_flux_file_ptr = ROOT.TFile.Open(alt_flux_file, "READ")
alt_flux_hist = alt_flux_file_ptr.Get("h_flux")

# Alternative hypothesis flux.
make_mpl_plot(
    alt_flux_hist,
    cmap="cmr.ocean",
    savefig=os.path.join(path, "alt_flux"),
    log_scale=False,
    title="No Tunnels Flux",
    units=z_title,
    draw_grid=True,
    flip=False,
)
#"""
