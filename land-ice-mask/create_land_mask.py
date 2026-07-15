#!/usr/bin/env python

"""
Regrid higher resolution land terrain to lower resolution.

Open (and resolved) questions:
- What are inputs from OSTIA land pixel bits? (Open)
    - Lake uses ESA ArcLake, otherwise unknown
- ESA CCI SST uses ESA CCI Land Cover without lakes

The land-sea component is formed from an ESA CCI SST instance, utilising the
mask field (which is a bitmask, the 2nd bit indicating if the pixel represents land).

Steps:

1. Load Terrain NetCDF file
2. Load Lakes NetCDF file
3. Select locations where we have large lakes, a selected list:
    - 'CASPIAN'
    - 'SUPERIOR'
    - 'MICHIGAN'
    - 'HURON'
    - 'ERIE'
    - 'ONTARIO'
4. Exclude all other lakes
5. Extract land from the SST dataset
6. Join lake array and terrain array
7. Convolve to lower-resolution

"""

import argparse
from datetime import datetime
from pathlib import Path
from types import NoneType

from glomar_gridding.grid import grid_from_resolution

import numpy as np
import polars as pl
import cf_xarray  # pylint: disable=unused-import  # noqa: F401
import xarray as xr
import yaml

from dcent_infilling.utils import get_coordnames, convolution_wgts, process_da


config_help = """Configuration YAML file to use. With 'land_mask' section with:
- sst_input_file : path to a high-resolution ESA-CCI SST frame as a netcdf file
  with a 'mask' array.
- lake_file : path to the ESA ArcLake NetCDF file - expected to be the same
  resolution as the sst_input_file.
- lake_id_file : path to a feather file containing a mapping between "Lake ID"
  and "Lake Name" referencing the "lakeid" array in the lake_file.
- output_file : path to the output file. Parent directories will be created as
  needed.
- target_resolution : integer or list of 2 integers indicating the lat and lon
  resolutions. If an integer then the resolution is the same in both directions.
- lakes_to_keep : list of lake names to keep.
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


def standarise_array_coords(
    da: xr.DataArray,
    units: str = "1",
) -> xr.DataArray:
    """Make xarray dataarray."""
    lat_name, lon_name = get_coordnames(da)
    lat_resolution_actual = da.coords[lat_name][1] - da.coords[lat_name][0]
    lat_resolution = abs(lat_resolution_actual)
    lon_resolution = abs(da.coords[lon_name][1] - da.coords[lon_name][0])

    lats = np.arange(
        -90.0 + lat_resolution / 2,
        90.0,
        lat_resolution,
        dtype=np.float32,
    )
    lons = np.arange(
        -180 + lon_resolution / 2,
        180.0,
        lon_resolution,
        dtype=np.float32,
    )

    if lat_resolution_actual < 0:
        lats = lats[::-1]

    new_da = xr.DataArray(
        name=da.name,
        data=da.values,
        coords=xr.Coordinates(
            coords={
                "latitude": (["latitude"], lats, {"units": "degrees_north"}),
                "longitude": (["longitude"], lons, {"units": "degrees_east"}),
            }
        ),
        attrs={"units": units},
    )
    return new_da.sortby(["latitude", "longitude"])


def process_lakes(
    lake_file: Path,
    lake_id_file: Path,
    lakes_to_keep: list[str] | None = None,
) -> xr.DataArray:
    """Process Lakes - select key lakes."""
    lake_da = standarise_array_coords(
        xr.load_dataset(lake_file)["lakeid"].astype(np.uint16)
    )

    if lakes_to_keep is None:
        # Keep all lakes
        lake_da.values = np.where(lake_da.values >= 1, 1, 0)
        return lake_da

    if not lakes_to_keep:
        # Keep no lakes - return 0s
        lake_da.values = np.zeros_like(lake_da.values)
        return lake_da

    lake_ids = pl.read_ipc(
        lake_id_file,
        memory_map=False,
        columns=["Lake Name", "Lake ID"],
    )
    lake_ids_subset = lake_ids.filter(
        pl.col("Lake Name").str.to_uppercase().is_in(lakes_to_keep)
    )
    if lake_ids_subset.height != len(lakes_to_keep):
        raise ValueError(
            "Mismatch between lakes_to_keep and ids_to_keep:\n"
            + f"Lakes to keep: {', '.join(lakes_to_keep)}\n"
            + f"Lakes remaining in frame: {', '.join(lake_ids_subset['Lake Name'])}\n"
        )

    ids_to_keep = lake_ids_subset.get_column("Lake ID").sort().to_numpy()
    lake_da.values = np.where(np.isin(lake_da.values, ids_to_keep), 1, 0)
    return lake_da.astype(np.uint8)


def process_mask(
    land_mask_file: Path,
    lake_file: Path,
    lake_id_file: Path,
    lakes_to_keep: list[str] | None = None,
) -> xr.DataArray:
    """Combine land and lake masks."""
    lake_da = process_lakes(
        lake_file=lake_file,
        lake_id_file=lake_id_file,
        lakes_to_keep=lakes_to_keep,
    )
    # Mask
    land_da = xr.load_dataset(land_mask_file)["mask"][0].astype(np.uint8)
    # Mask is a bitmask, the 2nd bit indicates 'land' (result is True is land)
    land_da.values = (np.bitwise_and(land_da.values, 2) == 2).astype(np.uint8)
    land_da.name = "mask"
    land_da = standarise_array_coords(land_da)
    land_da.values = np.where(lake_da.values == 1, 0, land_da.values)
    return land_da


def convolve_mask(
    mask: xr.DataArray,
    target_grid: xr.DataArray,
    verbose: bool = False,
    attrs: dict | None = None,
) -> xr.DataArray:
    """Reduce the resolution of a mask to a target grid."""
    latname, lonname = get_coordnames(mask, raise_if_missing=True)
    lats = mask.coords[latname].values
    lons = mask.coords[lonname].values

    target_latname, target_lonname = get_coordnames(target_grid, raise_if_missing=True)
    dy_ = len(lats) / len(target_grid.coords[target_latname])
    dx_ = len(lons) / len(target_grid.coords[target_lonname])

    if (dy := int(dy_)) != dy_:
        raise ValueError(
            "Cannot get scaling for latitude (must be an integer scaling)."
        )

    if (dx := int(dx_)) != dx_:
        raise ValueError(
            "Cannot get scaling for longitude (must be an integer scaling)."
        )

    wgts = convolution_wgts(lats, lons)

    result = process_da(
        mask,
        wgts=wgts,
        lats=lats,
        lons=lons,
        dy=dy,
        dx=dx,
        verbose=verbose,
    )

    attrs = attrs or mask.attrs

    return xr.DataArray(
        name=mask.name,
        data=result,
        coords=target_grid.coords,
        attrs=attrs,
    )


def get_target_grid(
    resolution: list[int] | None,
) -> xr.DataArray:
    """Get the target grid for a given resolution."""
    resolution = resolution or [5, 5]
    return grid_from_resolution(
        resolution=list(resolution),
        bounds=[(-90, 90), (-180, 180)],
        coord_names=["latitude", "longitude"],
        definition="left",
    )


def get_final_land_mask(
    config: dict,
    verbose: bool,
):
    """Get the final land mask."""
    land_mask = process_mask(
        land_mask_file=config["sst_input_file"],
        lake_file=config["lake_file"],
        lake_id_file=config["lake_id_file"],
        lakes_to_keep=config.get("lakes_to_keep", []),
    )

    target_resolution = config.get("target_resolution", 5)
    # Check valid target resolution
    if isinstance(target_resolution, (int, float)):
        target_resolution = [target_resolution, target_resolution]
    elif (
        isinstance(target_resolution, (tuple, list))
        and len(target_resolution) == 2
        and all(isinstance(x, (int, float)) for x in target_resolution)
    ):
        target_resolution = list(target_resolution)
    else:
        raise ValueError(
            "TARGET_RESOLUTION must be numeric, or a tuple of two numeric values. "
            + f"Got {target_resolution = }"
        )

    target_grid = get_target_grid(resolution=target_resolution)  # type: ignore (int,float,list)
    method_string = (
        "upscaling by counting pixels above threshold and do "
        + "a latitude weighted average"
    )
    doi = (
        "https://doi.org/10.1038/s41597-024-03147-w,"
        + " https://doi.org/10.7488/ds/106,"
        + " https://doi.org/10.3390/rs9010036"
    )
    extra_attrs = {
        "title": (
            f"Upscaled {target_resolution[0]} x {target_resolution[1]} "
            + "land area fraction"
        ),
        "institution": "National Oceanography Centre",
        "source": (
            "ESA-CCI SST L4 analysis mask (derived from ESA CCI LC), "
            + "ESA Arc-Lake 1.1 aux"
        ),
        "history": f"Date produced {datetime.now().date()}",
        "references": doi,
        "comment": ("contact: steven.chan@noc.ac.uk; methods: " + method_string),
    }

    result = convolve_mask(
        land_mask,
        target_grid=target_grid,
        verbose=verbose,
        attrs={"units": "1"},
    )
    result = result.astype(np.float32)

    result_ds = result.to_dataset()
    result_ds = result_ds.cf.add_bounds(["latitude", "longitude"])
    result_ds.attrs = extra_attrs
    return result_ds


def main() -> NoneType:
    """Generate the land mask."""
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)
    land_config = config.get("land_mask", {})

    land_mask = get_final_land_mask(config=land_config, verbose=args.verbose)
    out_file = Path(land_config["out_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    land_mask.to_netcdf(out_file)
    return None


if __name__ == "__main__":
    main()
