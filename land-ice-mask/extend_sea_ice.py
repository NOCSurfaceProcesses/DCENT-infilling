#!/usr/bin/env python

"""
Artificially extend HadISST2 by repeating monthly images from the last year of data.

Add the final year of data until a target year is added, and filled. Missing
full years are added by copying the previous year. The final year is filled (if
required) by adding the required months from the previous year. This will
result in duplicated data for the final years.

Source of data is available: https://www.metoffice.gov.uk/hadobs/hadisst2/

Written by S. C. Chan
Modified by J. T. Siddons
"""

import argparse
from pathlib import Path
from datetime import datetime
from types import NoneType
import yaml

import xarray as xr
import polars as pl

from dcent_infilling.utils import get_time_coordname  # For datetime manipulation


config_help = """Configuration YAML file to use. With 'extend_ice_mask' section
with:
- ice_concentration_file : path to the HadISST2 sea ice concentration dataset.
- output_file : path to write the extended sea ice concentration data.
- target_year : year to extend the data to - defaults to the current year.
- var_name : name of the sea-ice concentration variable in the input file
  (defaults to "sic").
- time_name: name of the time coordinate in the sea ice concentration dataset
  (defaults to "time").
"""


parser = argparse.ArgumentParser(
    description="Extend the ice-mask by appending the values from the previous year."
)
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=True,
    type=str,
    help=config_help,
)


def get_last_datetime(da: xr.DataArray, time_name: str) -> datetime:
    """
    Get the last datetime from the file, we want to fill from this to the end
    of that year.
    """
    last_date = pl.Series(da[time_name].values).last()
    if not isinstance(last_date, datetime):
        raise TypeError("Time dimension is not datetime")
    return last_date


def fill_year(da: xr.DataArray, time_name: str) -> xr.DataArray:
    """Fill the final year of the array."""
    last_date = get_last_datetime(da, time_name)
    last_month = last_date.month
    if last_month == 12:
        # Don't need to add anything
        return da.copy()

    print(f"Filling year = {last_date.year}.")

    inds = slice(-12, -last_month)
    extension = da.isel({time_name: inds})

    times = pl.Series(extension.coords[time_name].values)
    new_times = times.dt.replace(year=times.dt.year() + 1)
    extension.coords[time_name] = new_times.to_numpy()

    return xr.concat([da, extension], time_name)


def add_year(da: xr.DataArray, target_year: int, time_name: str) -> xr.DataArray:
    """Add extra years, until the target year has been reached."""
    sic_extended = da.copy()
    last_date = get_last_datetime(sic_extended, time_name)
    last_year = last_date.year

    while last_year < target_year:
        print(f"Adding year = {last_year + 1}")
        inds = slice(-12, None)
        extension = sic_extended.isel({time_name: inds})

        times = pl.Series(extension.coords[time_name].values)
        new_times = times.dt.replace(year=times.dt.year() + 1)
        extension.coords[time_name] = new_times.to_numpy()

        sic_extended = xr.concat([sic_extended.copy(), extension], time_name)
        last_year = get_last_datetime(sic_extended, time_name).year

    return sic_extended


def main() -> NoneType:  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)
    job_config = config.get("extend_ice_mask", {})

    in_file = Path(job_config["ice_concentration_file"])
    out_file = Path(job_config["output_file"])

    var_name = job_config.get("var_name", "sic")
    target_year = job_config.get("target_year", datetime.now().year)

    if not in_file.is_file():
        raise FileNotFoundError(f"HadISST2 file {in_file} not found.")

    # Load Sea-Ice concentration file
    sic = xr.open_dataset(in_file)[var_name]

    time_name = job_config.get(
        "time_name",
        get_time_coordname(sic, raise_if_missing=True),
    )

    # NOTE: add years first, then fill the final year
    sic_extended = add_year(sic, target_year=target_year, time_name=time_name)
    sic_extended = fill_year(sic_extended, time_name=time_name)

    print(f"Final Extended Result: {sic_extended = }")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sic_extended.to_netcdf(out_file)

    return None


if __name__ == "__main__":
    main()
