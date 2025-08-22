#!/usr/bin/env python

"""
Script to stitch land and sea surface temperatures using weights.

By J. Siddons 2025-02.
For Python >= 3.11
"""

import os

import yaml
import polars as pl
import xarray as xr

from glomar_gridding.io import load_array, load_dataset

CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "config_blend.yaml")
YEAR_RANGE: tuple[int, int] = (1850, 2024)


def prep_variable(
    path: str,
    variable: str,
    member: int,
) -> pl.DataFrame:
    """Load ensemble member data as a polars frame for the input variable."""
    return (
        pl.from_pandas(load_dataset(path, member=member).to_dataframe().reset_index())
        .filter(pl.col("time").dt.year().is_between(*YEAR_RANGE, closed="both"))
        .with_columns(pl.col("time").dt.replace(day=15, hour=12).name.keep())
        .rename(
            {"epsilon": f"{variable}_epsilon", "n_obs": f"{variable}_n_obs"},
            strict=False,
        )
    )


def prep_weights(
    land_mask_path: str,
    ice_mask_path: str,
    land_mask_var_name: str = "mask",
    ice_mask_var_name: str = "mask",
    land_mask_coords: tuple[str, str] = ("latitude", "longitude"),
    ice_mask_coords: tuple[str, str] = ("latitude", "longitude"),
    to_0_360: bool = True,
) -> pl.DataFrame:
    """Compute the land+ice fraction mask."""
    if not (os.path.isfile(land_mask_path) and os.path.isfile(ice_mask_path)):
        raise FileNotFoundError("Cannot find one of the mask files")
    # NOTE: this is the fraction of sea that is ice covered, not the fraction
    #       of the grid box that is ice covered.
    ice_mask = pl.from_pandas(
        load_array(ice_mask_path, ice_mask_var_name).to_dataframe().reset_index()
    ).rename({ice_mask_var_name: "sea_ice_fraction"})
    # NOTE: this is the fraction of the grid box that is land
    land_mask = pl.from_pandas(
        load_array(land_mask_path, land_mask_var_name).to_dataframe().reset_index()
    ).rename({land_mask_var_name: "land_fraction"})
    ice_mask = ice_mask.join(
        land_mask,
        left_on=ice_mask_coords,
        right_on=land_mask_coords,
        how="left",
        coalesce=True,
    )
    ice_mask = ice_mask.with_columns(
        (
            pl.col("land_fraction")
            + (1 - pl.col("land_fraction")) * pl.col("sea_ice_fraction")
        ).alias("weight")
    ).drop(["land_fraction", "sea_ice_fraction"], strict=False)

    if to_0_360:
        ice_mask = ice_mask.with_columns(
            pl.when(pl.col("longitude").lt(0))
            .then(pl.col("longitude") + 360)
            .otherwise(pl.col("longitude"))
            .alias("longitude")
        )

    return ice_mask


def main() -> None:  # noqa: D103
    with open(CONFIG_PATH, "r") as io:
        config: dict = yaml.safe_load(io)

    e0 = config.get("e0", 1)
    ef = config.get("ef", 200)
    ensembles: list[int] = list(range(e0, ef + 1))

    sst_path: str = config.get("sst_path", "")
    sst_var_name: str = config.get("sst_var_name", "sst_anom")

    lsat_path: str = config.get("lsat_path", "")
    lsat_var_name: str = config.get("lsat_var_name", "lsat_anom")

    out_path: str = config.get("out_path", "")
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    out_var_name: str = config.get("out_var_name", "combined")

    land_mask_path: str = config.get("land_mask_path", "")
    land_mask_var_name: str = config.get("land_mask_var_name", "mask")
    ice_mask_path: str = config.get("ice_mask_path", "")
    ice_mask_var_name: str = config.get("ice_mask_var_name", "mask")

    coords = ["time", "lat", "lon"]
    weights_var_name = "weight"

    weights = (
        prep_weights(
            land_mask_path=land_mask_path,
            land_mask_var_name=land_mask_var_name,
            ice_mask_path=ice_mask_path,
            ice_mask_var_name=ice_mask_var_name,
            to_0_360=True,
        )
        .rename({"latitude": "lat", "longitude": "lon"})
        .filter(pl.col("time").dt.year().is_between(*YEAR_RANGE, closed="both"))
        .with_columns(pl.col("time").dt.replace(day=15, hour=12).name.keep())
        .select([*coords, weights_var_name])
        .unique()
    )
    print("Prepped the land-ice mask")

    n_members = len(ensembles)
    for i, member in enumerate(ensembles):
        sst_array = prep_variable(sst_path, variable="sst", member=member)
        lsat_array = prep_variable(lsat_path, variable="lsat", member=member)

        combined_df = sst_array.join(lsat_array, on=coords, how="left")
        combined_df = combined_df.join(weights, on=coords, how="left")

        out_array_member_path = out_path.format(member=member)

        combined_df = combined_df.with_columns(
            (
                pl.col(lsat_var_name) * pl.col(weights_var_name)
                + (1 - pl.col(weights_var_name)) * pl.col(sst_var_name)
            ).alias(out_var_name)
        )
        combined_df = combined_df.sort(coords)

        xr.Dataset.from_dataframe(combined_df.to_pandas().set_index(coords)).to_netcdf(
            out_array_member_path
        )
        print(f"Done ensemble member: {member} | {(i + 1) / n_members:.2%}")

    return None


if __name__ == "__main__":
    main()
