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

import os

from itertools import product
from typing import Any

import polars as pl
import numpy as np
import xarray as xr
from polars.datatypes.classes import DataTypeClass


from glomar_gridding.grid import grid_from_resolution, map_to_grid
from glomar_gridding.io import load_array


BASE_PATH: str = "/path/to/sst"
PATH: str = os.path.join(BASE_PATH, "Uncertainty_reso_5_{year}_{month:02d}.nc")

OUT_DIR: str = "/path/to/sst_err_corr"
OUT_PATH: str = os.path.join(OUT_DIR, "Error_Cov_Mat_5_{year}_{month:02d}.nc")

YEARS: tuple[int, int] = (1850, 2024)
MONTHS: list[int] = list(range(1, 13))


def _create_grid() -> tuple[xr.DataArray, pl.DataFrame, pl.DataFrame]:
    grid = grid_from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (2.5, 360)],
        coord_names=["latitude", "longitude"],
    )
    lats: np.ndarray = grid["latitude"].values
    lons: np.ndarray = grid["longitude"].values
    lat_df: pl.DataFrame = pl.DataFrame({"lat": lats}).with_row_index("index", offset=1)
    lon_df: pl.DataFrame = pl.DataFrame({"lon": lons}).with_row_index("index", offset=1)
    return grid, lat_df, lon_df


def _load_file(
    year: int, month: int
) -> tuple[pl.DataFrame | None, xr.DataArray | None]:
    file = PATH.format(year=year, month=month)
    if not os.path.isfile(file):
        print(f"    File: {file} not found. Skipping.")
        return None, None
    cov_table: np.ndarray = load_array(file, "cov").values
    schema: list[tuple[str, DataTypeClass]] = [
        ("row_idx", pl.UInt16),
        ("col_idx", pl.UInt16),
        ("row_lon_idx", pl.UInt16),
        ("row_lat_idx", pl.UInt16),
        ("col_lon_idx", pl.UInt16),
        ("col_lat_idx", pl.UInt16),
        ("covariance", pl.Float32),
    ]
    cov_frame: pl.DataFrame = pl.from_numpy(
        cov_table,
        orient="col",
        schema=schema,
    )
    sigma2 = load_array(file, "sigma2")
    return cov_frame, sigma2


def _add_grid_pos(
    cov_frame: pl.DataFrame,
    lat_df: pl.DataFrame,
    lon_df: pl.DataFrame,
) -> pl.DataFrame:
    # Get latitude and longitudes
    cov_frame = cov_frame.join(
        lat_df,
        left_on="row_lat_idx",
        right_on="index",
        how="left",
        coalesce=True,
    ).rename({"lat": "row_lat"})
    cov_frame = cov_frame.join(
        lon_df,
        left_on="row_lon_idx",
        right_on="index",
        how="left",
        coalesce=True,
    ).rename({"lon": "row_lon"})
    cov_frame = cov_frame.join(
        lat_df,
        left_on="col_lat_idx",
        right_on="index",
        how="left",
        coalesce=True,
    ).rename({"lat": "col_lat"})
    cov_frame = cov_frame.join(
        lon_df,
        left_on="col_lon_idx",
        right_on="index",
        how="left",
        coalesce=True,
    ).rename({"lon": "col_lon"})
    return cov_frame


def _add_grid_idx(
    cov_frame: pl.DataFrame,
    grid: xr.DataArray,
) -> pl.DataFrame:
    # Align to the grid
    cov_frame = cov_frame.pipe(
        map_to_grid,
        grid=grid,
        obs_coords=["row_lat", "row_lon"],
        grid_coords=["latitude", "longitude"],
    ).rename({"grid_idx": "row_grid_idx"})
    cov_frame = cov_frame.pipe(
        map_to_grid,
        grid=grid,
        obs_coords=["col_lat", "col_lon"],
        grid_coords=["latitude", "longitude"],
    ).rename({"grid_idx": "col_grid_idx"})
    return cov_frame


def _get_cov_mat(cov_frame: pl.DataFrame, n: int) -> np.ndarray:
    cov = np.full((n, n), fill_value=np.nan, dtype=np.float32)
    cov[cov_frame["row_grid_idx"], cov_frame["col_grid_idx"]] = cov_frame["covariance"]
    return cov


def _output_coords(n, grid: xr.DataArray) -> xr.Coordinates:
    coord_names: list[str] = [str(c) for c in grid.coords.keys()]
    coord_df = pl.from_records(
        list(grid.coords.to_index()),
        schema=coord_names,
        orient="row",
    )

    out_coords: dict[str, Any] = {"index_1": range(n), "index_2": range(n)}
    for i in range(1, 3):
        out_coords.update(
            {f"{c}_{i}": (f"index_{i}", coord_df[c]) for c in coord_df.columns}
        )
    return xr.Coordinates(out_coords)


def main() -> None:  # noqa: D103
    grid, lat_df, lon_df = _create_grid()
    n = int(np.prod(grid.shape))
    out_coords = _output_coords(n, grid)
    years: list[int] = list(range(YEARS[0], YEARS[1] + 1))
    n_files = len(years) * len(MONTHS)

    for i, (year, month) in enumerate(product(years, MONTHS)):
        out_file = OUT_PATH.format(year=year, month=month)
        print(f"Doing {year}-{month:02d} | {i / n_files:.2%}")
        # if os.path.isfile(out_file):
        #     print(f"    Output file: {out_file} already exists. Skipping.")
        #     # Done already
        #     continue
        cov_frame, sigma2 = _load_file(year, month)
        if cov_frame is None or sigma2 is None:
            continue
        cov_frame = cov_frame.pipe(
            _add_grid_pos,
            lat_df=lat_df,
            lon_df=lon_df,
        )
        cov_frame = cov_frame.pipe(
            _add_grid_idx,
            grid=grid,
        )
        cov_mat: np.ndarray = _get_cov_mat(cov_frame, n=n)
        if sigma2.shape != grid.shape:
            sigma2 = sigma2.transpose()
        np.fill_diagonal(cov_mat, sigma2.values.flatten())
        da = xr.DataArray(
            name="error_covariance",
            data=cov_mat.astype(np.float32),
            coords=out_coords,
        )
        da.to_netcdf(out_file)

        del da, cov_mat, cov_frame, sigma2

    return None


if __name__ == "__main__":
    main()
