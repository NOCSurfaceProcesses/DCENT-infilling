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
from pathlib import Path
from typing import Any

import polars as pl
import numpy as np
import xarray as xr
from polars.datatypes.classes import DataTypeClass


from glomar_gridding.grid import grid_from_resolution, map_to_grid


BASE_PATH: str = "/path/to/sst"
PATH: str = os.path.join(BASE_PATH, "Uncertainty_reso_5_{year}_{month:02d}.nc")

OUT_DIR: str = "/path/to/sst_err_corr"
OUT_PATH: str = os.path.join(OUT_DIR, "Error_Cov_Mat_5_{year}_{month:02d}.nc")

YEARS: tuple[int, int] = (1850, 2024)
MONTHS: list[int] = list(range(1, 13))

OVERWRITE: bool = False


def _create_grid(
    resolution: int = 5,
) -> tuple[xr.DataArray, pl.DataFrame, pl.DataFrame]:
    grid = grid_from_resolution(
        resolution=resolution,
        bounds=[(-90, 90), (0, 360)],
        coord_names=["latitude", "longitude"],
        definition="left",
    )
    lats: np.ndarray = grid["latitude"].values
    lons: np.ndarray = grid["longitude"].values
    lat_df: pl.DataFrame = pl.DataFrame({"lat": lats}).with_row_index("index", offset=1)
    lon_df: pl.DataFrame = pl.DataFrame({"lon": lons}).with_row_index("index", offset=1)
    return grid, lat_df, lon_df


def load_table(
    file: str | Path,
) -> tuple[pl.DataFrame | None, xr.DataArray | None]:
    """Load the error covariance table and variance from a file."""
    file = Path(file)
    if not file.is_file():
        print(f"File {file} not found.")
        return None, None

    ds = xr.open_dataset(file)
    cov_table: np.ndarray = ds["cov"].values
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
    sigma2 = ds["sigma2"].compute()
    return cov_frame, sigma2


def _load_file(
    year: int,
    month: int,
    file_path: str | None = None,
) -> tuple[pl.DataFrame | None, xr.DataArray | None]:
    file_path = file_path or PATH
    file = file_path.format(year=year, month=month)
    return load_table(file)


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
    cov_frame = map_to_grid(
        cov_frame,
        grid=grid,
        obs_coords=["row_lat", "row_lon"],
        grid_coords=["latitude", "longitude"],
    ).rename({"grid_idx": "row_grid_idx"})
    cov_frame = map_to_grid(
        cov_frame,
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


def process_error_table(
    in_file: str | Path,
    resolution: int = 5,
) -> xr.DataArray:
    """Process a single error covariance table from a file."""
    cov_frame, sigma2 = load_table(in_file)
    if cov_frame is None or sigma2 is None:
        raise FileNotFoundError(f"Could not find or open {in_file}")
    grid, lat_df, lon_df = _create_grid(resolution=resolution)
    n = int(np.prod(grid.shape))
    out_coords = _output_coords(n, grid)
    return _process_table(
        cov_frame=cov_frame,
        sigma2=sigma2,
        grid=grid,
        lat_df=lat_df,
        lon_df=lon_df,
        out_coords=out_coords,
        n=n,
    )


def _process_table(
    cov_frame: pl.DataFrame,
    sigma2: xr.DataArray,
    grid: xr.DataArray,
    lat_df: pl.DataFrame,
    lon_df: pl.DataFrame,
    out_coords: xr.Coordinates,
    n: int,
) -> xr.DataArray:
    if sigma2.shape != grid.shape:
        sigma2 = sigma2.transpose()
    if sigma2.shape != grid.shape:
        raise ValueError(
            f"Could not align sigma2 to grid. {grid.shape = }, {sigma2.shape = }."
        )

    cov_frame = _add_grid_pos(
        cov_frame,
        lat_df=lat_df,
        lon_df=lon_df,
    )

    cov_frame = _add_grid_idx(
        cov_frame,
        grid=grid,
    )

    cov_mat: np.ndarray = _get_cov_mat(cov_frame, n=n)

    # Acts in-place
    np.fill_diagonal(cov_mat, sigma2.values.flatten())
    da = xr.DataArray(
        name="error_covariance",
        data=cov_mat.astype(np.float32),
        coords=out_coords,
    )
    return da


def main() -> None:  # noqa: D103
    grid, lat_df, lon_df = _create_grid()
    n = int(np.prod(grid.shape))
    out_coords = _output_coords(n, grid)
    years: list[int] = list(range(YEARS[0], YEARS[1] + 1))
    n_files = len(years) * len(MONTHS)

    for i, (year, month) in enumerate(product(years, MONTHS)):
        out_file = OUT_PATH.format(year=year, month=month)
        print(f"Doing {year}-{month:02d} | {i / n_files:.2%}")
        if os.path.isfile(out_file) and not OVERWRITE:
            print(f"Output file: {out_file} already exists. Skipping.")
            # Done already
            continue
        cov_frame, sigma2 = _load_file(year, month)
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
