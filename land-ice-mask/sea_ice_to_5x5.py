#!/usr/bin/env python

"""Adjust 1x1 sea-ice concentration to 5x5."""

import argparse
from datetime import datetime
from pathlib import Path
from types import NoneType
from warnings import warn
import yaml

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from dcent_infilling.utils import (
    convolution_wgts,
    get_coordnames,
    process_da,
    get_time_coordname,
)


config_help = """Configuration YAML file to use. With 'reduce_ice_mask' section
with:
- ice_concentration_file : path to the (extended) HadISST.2.2.2.0 sea ice
  concentration file.
- terrain_file : path to the land mask file - at the target resolution for the
  ice concentration.
- output_file : path for the output file.
- var_name : Name of the sea-ice concentration variable in the input dataset
  (defaults to "sic").
- start_year : start year of the output sea ice concentration file (defaults to
  1850). Note, if the input file starts after this then the start date will
  reflect the input file.
- end_year : end year of the output sea ice concentration file (defaults to
  current year). Note, if the input file ends before this then the end date will
  reflect the input file.
"""


parser = argparse.ArgumentParser(description="Generate a combined land-lake-sea mask")
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=True,
    type=str,
    help=config_help,
)
parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    required=False,
    action="store_true",
    help="Print more debugging output",
)


def sea_ice_frac_computer(
    sic_da: xr.DataArray,
    dx: int = 5,
    dy: int = 5,
    verbose: bool = False,
):
    """
    Front end/view to compute weighted sic at lower resolution
    See sea_ice_lat_wgted_kernel.

    Parameters
    ----------
    sic_da : xarray.DataArray or iris.cube
        Data cube for sic, must have proper latitude coords!
    dx, dy : int
        Dimension of the kernel to be convolved into lower resolution
    verbose : bool
        More stdout

    Returns
    -------
    ans_arr : np.ndarray
        Lower resolution than sic_da (len(lats)//dy, len(lons)//dx)

    """
    timename = get_time_coordname(sic_da, raise_if_missing=True)
    latname, lonname = get_coordnames(sic_da, raise_if_missing=True)
    lats = sic_da[latname].data
    lons = sic_da[lonname].data
    times = sic_da[timename].data

    wgts_1 = convolution_wgts(lats, lons)

    if (times.size == 1) and (len(sic_da.shape) == 2):
        sic_da = sic_da.expand_dims(dim=timename)
        if verbose:
            print(f"New axis added to sic_da: {sic_da.shape}")

    result = np.zeros((times.size, len(lats) // dy, len(lons) // dx))

    for date_i in range(times.size):
        if verbose:
            print(sic_da[date_i])

        nan_mask = np.isnan(sic_da[date_i].values)
        wgts_2 = np.where(nan_mask, 0.0, wgts_1)

        result[date_i, :, :] = process_da(
            sic_da[date_i],
            wgts=wgts_2,
            lats=lats,
            lons=lons,
            dy=dy,
            dx=dx,
            verbose=verbose,
        )

    # Check the result and adjust to max 1 if required
    if np.any(res_gt_1 := (result > 1)):
        if verbose:
            warn(f"Have {np.sum(res_gt_1)} values above 1")
        if (max_result := np.max(result)) - 1 > 1e-5:
            raise ValueError(f"Have values larger than 1 + 1e-5 ({max_result}).")
        result = np.where(res_gt_1, 1, result)

    return result


def main() -> NoneType:  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)
    ice_config = config.get("reduce_ice_mask", {})

    verbose = args.verbose

    in_file = Path(ice_config["ice_concentration_file"])
    terrain_file = Path(ice_config["terrain_file"])
    out_file = Path(ice_config["output_file"])
    var_name = ice_config.get("var_name", "sic")

    start_year = ice_config.get("start_year", 1850)
    end_year = ice_config.get("end_year", datetime.now().year)

    sic = xr.load_dataset(in_file)[var_name]
    time_name = get_time_coordname(sic)
    lat_name, lon_name = get_coordnames(sic, raise_if_missing=True)

    if any(sic[lon_name] > 180.0):
        # Convert 0 - 360 to -180 - 180
        sic.coords[lon_name] = ((sic.coords[lon_name] + 180) % 360) - 180

    sic = sic.sortby(lat_name).sortby(lon_name)

    if verbose:
        print(f"{sic.coords = }")

    years = sic.coords[time_name].dt.year
    sic = sic.sel({time_name: ((start_year <= years) & (years <= end_year))})

    terrain = xr.open_dataset(terrain_file)
    out_coords = xr.Coordinates(
        coords={
            time_name: sic.cf["time"].values,
            lat_name: terrain.cf["latitude"].values,
            lon_name: terrain.cf["longitude"].values,
        }
    )
    out_da = xr.DataArray(name="land sea-ice mask", coords=out_coords)
    threshold = ice_config.get("threshold", 0.15)

    sic_ge_threshold = xr.where(
        np.isnan(sic),
        np.nan,
        (sic >= threshold),
    )
    sic_15_5x5_ar = sea_ice_frac_computer(sic_ge_threshold, verbose=verbose)
    out_da.values = sic_15_5x5_ar

    if verbose:
        print(f"Writing to {out_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_da.to_netcdf(out_file)

    return None


if __name__ == "__main__":
    main()
