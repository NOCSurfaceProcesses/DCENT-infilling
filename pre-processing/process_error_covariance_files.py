#!/usr/bin/env python

"""
Script to convert DCENT error covariance table file to covariance matrices and
output to netCDF. This applies for the SST error covariances.

The input netCDF files contain a "covariance" field, which is a 7 x ... array
that can be converted to a DataFrame. There is an associated attr
("explanation") which details the fields:
    - row index
    - column index
    - longitude index of the row index
    - latitude index of the row index
    - longitude index of the column index
    - latitude index of the column index
    - covariance

"""

import argparse
from functools import partial
import os

from itertools import product

import polars as pl
import numpy as np
import xarray as xr
import yaml

from dcent_infilling.error_covariance import (
    _process_table,
    _create_grid,
    _output_coords,
    load_table,
)


config_help = """Configuration YAML file to use, with:
- in_path : path to the input error covariance tables (with format fill spaces
  for 'year' and 'month', e.g
  /path/to/error_covariance/Uncertainty_reso_5_{year}_{month:02d}.nc).
- out_path : output path for the error covariance matrix files, with format
  fill spaces for 'year' and 'month'.
- start_year : start year to process (default is 1850).
- end_year : last year to process (default is 2025).
- overwrite : true / false indicating whether to overwrite an output file if it
  exists.
"""


parser = argparse.ArgumentParser(
    description="Convert error covariance table/sparse format into matrix format."
)
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=True,
    type=str,
    help=config_help,
)


def _load_file(
    year: int,
    month: int,
    file_path: str,
) -> tuple[pl.DataFrame | None, xr.DataArray | None]:
    file = file_path.format(year=year, month=month)
    return load_table(file)


def main() -> None:  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)

    start_year = config.get("start_year", 1850)
    end_year = config.get("end_year", 2025)

    in_path = config["in_path"]
    out_path = config["out_path"]

    grid, lat_df, lon_df = _create_grid()
    n = int(np.prod(grid.shape))
    out_coords = _output_coords(n, grid)

    load_file = partial(_load_file, file_path=in_path)

    years: list[int] = list(range(start_year, end_year + 1))
    months = range(1, 13)  # 1 -> 12
    n_files = len(years) * len(months)

    for i, (year, month) in enumerate(product(years, months)):
        out_file = out_path.format(year=year, month=month)
        print(f"Doing {year}-{month:02d} | {i / n_files:.2%}")
        if os.path.isfile(out_file) and not config.get("overwrite", False):
            print(f"Output file: {out_file} already exists. Skipping.")
            # Done already
            continue
        cov_frame, sigma2 = load_file(year=year, month=month)
        if cov_frame is None or sigma2 is None:
            continue
        da = _process_table(
            cov_frame=cov_frame,
            sigma2=sigma2,
            grid=grid,
            lat_df=lat_df,
            lon_df=lon_df,
            out_coords=out_coords,
            n=n,
        )
        da.to_netcdf(out_file)

        del da, cov_frame, sigma2

    return None


if __name__ == "__main__":
    main()
