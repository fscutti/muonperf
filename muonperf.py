import os
import sys
import utils
import ROOT as R

from tqdm import tqdm

# Acquisition time is seconds.
acquisition_time = {
    "eight_hours": 28800,
    "one_week": 604800,
    "three_weeks": 3 * 604800,
    "four_weeks": 4 * 604800,
    "two_months": 2 * 4 * 604800,
}

input_path = "/Users/fscutti/github/performance/data/input"
lookup_path = "/Users/fscutti/github/performance/data/lookup"
output_path = "/Users/fscutti/github/performance/data/output"

input_files = utils.get_files(input_path)
lookup_files = utils.get_files(lookup_path)

input_fluxes = utils.get_histograms(
    input_path, input_files, "h_flux", new_hist_name="h_pumas_flux"
)
input_opacities = utils.get_histograms(
    input_path, input_files, "h_opacity", new_hist_name="h_pumas_opacity"
)
input_fluxes_err = utils.get_histograms(
    input_path, input_files, "h_sigma", new_hist_name="h_pumas_flux_sim_err"
)

lookup_file = os.path.join(lookup_path, lookup_files[0])

# flux_uncertainites = {}
# flux_randomisation = {}
# opacity_interpolation = {}

for time_tag, seconds in tqdm(
    acquisition_time.items(), total=len(acquisition_time), colour="yellow"
):
    for h_file_name, h_flux in tqdm(
        input_fluxes.items(), total=len(input_fluxes), colour="green"
    ):  
        file_name = h_file_name.split(".")[0]
        output_file = f"Planning_{file_name}_{time_tag}.root"
        fits_path = os.path.join(output_path, output_file.split(".")[0])

        # Time unit is in seconds.
        h_flux_unc = utils.get_uncertainty(h_flux, time=seconds)

        h_flux_rand = utils.get_randomisation(h_flux, h_flux_unc)

        h_opacity_inter, h_opacity_inter_err = utils.get_opacity_interpolation(
            h_flux_rand, h_flux_unc, lookup_file
        )

        h_opacity = input_opacities[h_file_name]
        h_simulated_flux_err = input_fluxes_err[h_file_name]

        # Save histograms inside the same root file.
        utils.save_as_root(
            output_path,
            output_file,
            [
                h_flux,
                h_flux_unc,
                h_flux_rand,
                h_opacity_inter,
                h_opacity_inter_err,
                h_opacity,
                h_simulated_flux_err,
            ],
        )

        # Save histograms under the same path and as separate fits files.
        utils.save_as_fits(
            fits_path,
            [
                h_flux,
                h_flux_unc,
                h_flux_rand,
                h_opacity_inter,
                h_opacity_inter_err,
                h_opacity,
                h_simulated_flux_err,
            ],
        )
