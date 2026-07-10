"""
Regrid higher resolution land terrain to lower resolution.

Open (and resolved) questions:
- What are inputs from OSTIA land pixel bits? (Open)
    - Lake uses ESA ArcLake, otherwise unknown
- ESA CCI SST uses ESA CCI Land Cover without lakes

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
4.

"""

from datetime import datetime
from pathlib import Path
from types import NoneType

from glomar_gridding.grid import grid_from_resolution

import numpy as np
import polars as pl
import cf_xarray  # pylint: disable=unused-import  # noqa: F401
import xarray as xr

from .utils import get_coordnames, convolution_wgts, process_da


TARGET_RESOLUTION: int | tuple[int, int] = 5
LAND_MASK_FILE: Path = Path()
LAKE_FILE: Path = Path()
LAKE_ID_FILE: Path = Path()
LAKES_TO_KEEP: list[str] = [
    "CASPIAN",
    "SUPERIOR",
    "MICHIGAN",
    "HURON",
    "ERIE",
    "ONTARIO",
]
OUT_FILE: Path = Path()


def standarise_array_coords(
    da: xr.DataArray,
    units: str = "1",
    lat_rev: bool = True,
) -> xr.DataArray:
    """Make xarray dataarray."""
    lats = np.arange(-89.975, 89.975 + 0.0001, 0.05).astype(np.float32)
    lons = np.arange(-179.975, 179.975 + 0.001, 0.05).astype(np.float32)

    if lat_rev:
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
    land_da = standarise_array_coords(land_da, lat_rev=False)
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


def get_final_land_mask():
    """Get the final land mask."""
    land_mask = process_mask(
        land_mask_file=LAND_MASK_FILE,
        lake_file=LAKE_FILE,
        lake_id_file=LAKE_ID_FILE,
        lakes_to_keep=LAKES_TO_KEEP,
    )

    # Check valid target resolution
    if isinstance(TARGET_RESOLUTION, (int, float)):
        target_resolution = [TARGET_RESOLUTION, TARGET_RESOLUTION]
    elif (
        isinstance(TARGET_RESOLUTION, (tuple, list))
        and len(TARGET_RESOLUTION) == 2
        and all(isinstance(x, (int, float)) for x in TARGET_RESOLUTION)
    ):
        target_resolution = list(TARGET_RESOLUTION)
    else:
        raise ValueError(
            "TARGET_RESOLUTION must be numeric, or a tuple of two numeric values. "
            + f"Got {TARGET_RESOLUTION = }"
        )

    target_grid = get_target_grid(resolution=target_resolution)
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
        verbose=False,
        attrs={"units": "1"},
    )
    result = result.astype(np.float32)

    result_ds = result.to_dataset()
    result_ds = result_ds.cf.add_bounds(["latitude", "longitude"])
    result_ds.attrs = extra_attrs
    return result_ds


def main() -> NoneType:
    """Generate the land mask."""
    land_mask = get_final_land_mask()
    land_mask.to_netcdf(OUT_FILE)
    return None


if __name__ == "__main__":
    main()
