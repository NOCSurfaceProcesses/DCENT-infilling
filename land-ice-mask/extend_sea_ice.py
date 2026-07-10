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

from pathlib import Path
from datetime import datetime
from types import NoneType

import xarray as xr
import polars as pl  # For datetime manipulation

BASE_PATH: Path = Path("/path/to/HadISST2")
IN_FILE: Path = BASE_PATH / "HadISST.2.2.2.0_sea_ice_concentration.nc"
OUT_FILE: Path = BASE_PATH / "HadISST.2.2.2.0_sea_ice_concentration_extended.nc"

TARGET_YEAR: int = 2025
VAR_NAME: str = "sic"
TIME_NAME: str = "time"


def get_last_datetime(da: xr.DataArray) -> datetime:
    """
    Get the last datetime from the file, we want to fill from this to the end
    of that year.
    """
    last_date = pl.Series(da[TIME_NAME].values).last()
    if not isinstance(last_date, datetime):
        raise TypeError("Time dimension is not datetime")
    return last_date


def fill_year(da: xr.DataArray) -> xr.DataArray:
    """Fill the final year of the array."""
    last_date = get_last_datetime(da)
    last_month = last_date.month
    if last_month == 12:
        # Don't need to add anything
        return da.copy()

    print(f"Filling year = {last_date.year}.")

    inds = slice(-12, -last_month)
    extension = da.isel({TIME_NAME: inds})

    times = pl.Series(extension.coords[TIME_NAME].values)
    new_times = times.dt.replace(year=times.dt.year() + 1)
    extension.coords[TIME_NAME] = new_times.to_numpy()

    return xr.concat([da, extension], TIME_NAME)


def add_year(da: xr.DataArray) -> xr.DataArray:
    """Add extra years, until the target year has been reached."""
    sic_extended = da.copy()
    last_date = get_last_datetime(sic_extended)
    last_year = last_date.year

    while last_year < TARGET_YEAR:
        print(f"Adding year = {last_year + 1}")
        inds = slice(-12, None)
        extension = sic_extended.isel({TIME_NAME: inds})

        times = pl.Series(extension.coords[TIME_NAME].values)
        new_times = times.dt.replace(year=times.dt.year() + 1)
        extension.coords[TIME_NAME] = new_times.to_numpy()

        sic_extended = xr.concat([sic_extended.copy(), extension], TIME_NAME)
        last_year = get_last_datetime(sic_extended).year

    return sic_extended


def main() -> NoneType:  # noqa: D103
    if not IN_FILE.is_file():
        raise FileNotFoundError(f"HadISST2 file {IN_FILE} not found.")

    # Load Sea-Ice concentration file
    sic = xr.open_dataset(IN_FILE)[VAR_NAME]

    # NOTE: add years first, then fill the final year
    sic_extended = add_year(sic)
    sic_extended = fill_year(sic_extended)

    print(f"Final Extended Result: {sic_extended = }")
    sic_extended.to_netcdf(OUT_FILE)

    return None


if __name__ == "__main__":
    main()
