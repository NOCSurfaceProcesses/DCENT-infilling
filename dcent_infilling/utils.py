"""Utility functions."""

from warnings import warn

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr


def get_time_coordname(
    da: xr.DataArray,
    raise_if_missing: bool = False,
) -> str | None:
    """Get the time coordinate name."""
    time = da.cf["time"].name
    if not isinstance(time, str):
        msg = "Could not determine 'time' name"
        if raise_if_missing:
            raise ValueError(msg)
        warn(msg)
        time = None
    return time


def get_coordnames(
    da: xr.DataArray,
    raise_if_missing: bool = False,
) -> tuple[str | None, str | None]:
    """Attempt auto detect y-x coord names."""
    lat = da.cf["latitude"].name
    if not isinstance(lat, str):
        msg = "Could not determine 'lat' name"
        if raise_if_missing:
            raise ValueError(msg)
        warn(msg)
        lat = None
    lon = da.cf["longitude"].name
    if not isinstance(lon, str):
        msg = "Could not determine 'lon' name"
        if raise_if_missing:
            raise ValueError(msg)
        warn(msg)
        lon = None
    return (lat, lon)


def convolution_wgts(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Get the weights for the convolution to decrease resolution."""
    zoom_x = np.cos(np.deg2rad(lats))
    zoom_y = np.ones_like(lons)

    y_wgt_1, x_wgt_1 = np.meshgrid(zoom_y, zoom_x)
    return x_wgt_1 * y_wgt_1


def process_da(
    da: xr.DataArray,
    wgts: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    dy: int,
    dx: int,
    verbose: bool = False,
) -> np.ndarray:
    """Decrease resolution of a grid by a scale of dy, dx."""
    result = np.zeros((len(lats) // dy, len(lons) // dx))
    for y in range(len(lats) // dy):
        y0 = y * dy
        y_slice = slice(y0, y0 + dy)

        for x in range(len(lons) // dx):
            x0 = x * dx
            x_slice = slice(x0, x0 + dx)

            da_mini = da.values[y_slice, x_slice]

            kernel = wgts[y_slice, x_slice]
            norm = np.sum(kernel)
            kernel = np.zeros_like(kernel) if norm == 0 else kernel / norm

            if np.all(np.isnan(da_mini)):
                cell_result = 0.0
            else:
                cell_result = np.nansum(da_mini * kernel)
            if verbose:
                print(f"{da_mini = }")
                print(f"{kernel = }")
                print(f"{da_mini * kernel = }")
                print(f"{cell_result = }")
            if np.isnan(cell_result):
                raise ValueError(f"Answer for cell ({y = }, {x = }) is NaN")
            if verbose and (y % 4 == 0) and (x % 4 == 0) and (np.abs(lats[y0]) >= 55.0):
                print(
                    f"{lats[y0] = }, {lons[x0] = }, " + f"{kernel = }, {cell_result = }"
                )
            result[y, x] = cell_result
    return result
