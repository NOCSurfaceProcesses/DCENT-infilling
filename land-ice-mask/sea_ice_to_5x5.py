#!/usr/bin/env python

"""Adjust 1x1 sea-ice concentration to 5x5."""

from pathlib import Path
from types import NoneType
from warnings import warn

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

BASE_PATH: Path = Path("/path/to/HadISST2")
IN_FILE: Path = BASE_PATH / "HadISST.2.2.2.0_sea_ice_concentration.nc"
TERRAIN_FILE: Path = BASE_PATH / "land_lowres_kernel_esa_hybrid.nc"

OUT_FILE: Path = BASE_PATH / "HadISST2_5x5_1850_2025_replic.nc"

# WARNING: This will print a lot of data!
VERBOSE = False

START_YEAR: int = 1850
END_YEAR: int = 2025


def get_coordnames(da: xr.DataArray) -> tuple[str, str, str]:
    """Attempt auto detect t-y-x coord names."""
    time = da.cf["time"].name
    if not isinstance(time, str):
        raise ValueError("Could not determine 'time' name")
    lat = da.cf["latitude"].name
    if not isinstance(lat, str):
        raise ValueError("Could not determine 'latitude' name")
    lon = da.cf["longitude"].name
    if not isinstance(lon, str):
        raise ValueError("Could not determine 'longitude' name")
    return (time, lat, lon)


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
    timename, latname, lonname = get_coordnames(sic_da)
    lats = sic_da[latname].data
    lons = sic_da[lonname].data
    times = sic_da[timename].data

    zoom_x = np.cos(np.deg2rad(lats))
    zoom_y = np.ones_like(lons)

    y_wgt_1, x_wgt_1 = np.meshgrid(zoom_y, zoom_x)
    wgts_1 = x_wgt_1 * y_wgt_1

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

        for y in range(len(lats) // dy):
            y0 = y * dy
            y_slice = slice(y0, y0 + dy)
            for x in range(len(lons) // dx):
                x0 = x * dx
                x_slice = slice(x0, x0 + dx)
                sic_mini = sic_da.values[date_i, y_slice, x_slice]

                kernel = wgts_2[y_slice, x_slice]
                norm = np.sum(kernel)
                kernel = np.zeros_like(kernel) if norm == 0 else kernel / norm

                if np.all(np.isnan(sic_mini)):
                    cell_result = 0.0
                else:
                    cell_result = np.nansum(sic_mini * kernel)
                if verbose:
                    print(f"{date_i = }, {y = }, {x = }")
                    print(f"{sic_mini = }")
                    print(f"{kernel = }")
                    print(f"{sic_mini * kernel = }")
                    print(f"{cell_result = }")
                if np.isnan(cell_result):
                    raise ValueError("Answer is NaN")
                result[date_i, y, x] = cell_result
                if (
                    verbose
                    and (y % 4 == 0)
                    and (x % 4 == 0)
                    and (np.abs(lats[y0]) >= 55.0)
                ):
                    print(
                        f"{lats[y0] = }, {lons[x0] = }, "
                        + f"{kernel = }, {cell_result = }"
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
    sic = xr.load_dataset(IN_FILE)["sic"]
    time_name, lat_name, lon_name = get_coordnames(sic)

    if any(sic[lon_name] > 180.0):
        # Convert 0 - 360 to -180 - 180
        sic.coords[lon_name] = ((sic.coords[lon_name] + 180) % 360) - 180

    sic = sic.sortby(lat_name).sortby(lon_name)

    if VERBOSE:
        print(f"{sic.coords = }")

    years = sic.coords[time_name].dt.year
    sic = sic.sel({time_name: ((START_YEAR <= years) & (years <= END_YEAR))})

    terrain = xr.open_dataset(TERRAIN_FILE)
    out_coords = xr.Coordinates(
        coords={
            time_name: sic.cf["time"].values,
            lat_name: terrain.cf["latitude"].values,
            lon_name: terrain.cf["longitude"].values,
        }
    )
    out_da = xr.DataArray(name="land sea-ice mask", coords=out_coords)
    threshold = 0.15

    sic_ge_threshold = xr.where(
        np.isnan(sic),
        np.nan,
        (sic >= threshold),
    )
    sic_15_5x5_ar = sea_ice_frac_computer(sic_ge_threshold, verbose=VERBOSE)
    out_da.values = sic_15_5x5_ar

    if VERBOSE:
        print(f"Writing to {OUT_FILE}")

    out_da.to_netcdf(OUT_FILE)

    return None


if __name__ == "__main__":
    main()
