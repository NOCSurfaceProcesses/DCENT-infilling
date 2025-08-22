#!/usr/bin/env python
"""
Script to run Kriging for DCENT.

By A. Faulkner for python version 3.0 and up.
Modified by J. Siddons (2025-01). Requires python >= 3.11.

Encodes the uncertainty from the sampling into the gridded field for the
ensemble. This is done by generating a simulated field and observations and
computing a simulated gridded field.
See: https://doi.org/10.1029/2019JD032361
"""

# global
import atexit  # State management (run function on exit including error)
from datetime import datetime
import os
import shutil


if "POLARS_MAX_THREADS" not in os.environ:
    os.environ["POLARS_MAX_THREADS"] = "16"

# argument parser
import argparse
import yaml

# math tools
import numpy as np

# data handling tools
import polars as pl
import xarray as xr

# GloMarGridding
from glomar_gridding import __version__ as gg_version
from glomar_gridding.grid import (
    map_to_grid,
    assign_to_grid,
    grid_from_resolution,
)
from glomar_gridding.io import load_dataset, load_array, get_recurse
from glomar_gridding.stochastic import StochasticKriging, scipy_mv_normal_draw
from glomar_gridding.utils import (
    init_logging,
    get_date_index,
    get_month_midpoint,
)

# Debugging
import logging
import warnings


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config_DCENT.yaml"),
    help="Path to yaml file containing configuration settings",
    type=str,
)


def _set_seed(ensemble: int, month: int):
    np.random.seed(ensemble * (10**4) + month)
    return None


def _save_state(
    ds: xr.Dataset,
    ds_file: str,
    config: dict,
    config_file: str,
    member: int,
    year: int,
    month: int,
) -> None:
    logging.error(
        f"Job failed for {member = }, {year = }, {month = }. "
        + f"Writing out results to this point to {ds_file}. "
        + f"Writing state to config file {config_file}",
    )
    print("BACKING UP STATE!")
    ds.to_netcdf(ds_file + ".bak")
    config["state"] = {
        "member": member,
        "year": year,
        "month": month,
    }
    with open(config_file, "w") as io:
        yaml.safe_dump(config, io)


def _parse_args(
    parser,
) -> dict:
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config: dict = yaml.safe_load(io)

    return config


def _get_sst_error_cov(
    year: int,
    month: int,
    error_covariance_path: str,
) -> np.ndarray:
    file_path = error_covariance_path.format(year=year, month=month)
    ds = load_dataset(file_path)
    if not np.max(ds["longitude_1"]) > 350:
        raise ValueError("Coordinate system of error covariance is not 0-360")
    return ds["error_covariance"].values


def _get_lsat_error_cov(
    year: int,
    month: int,
    error_cov: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    date_int = (year - 1850) * 12 + (month - 1)
    sigma2 = error_cov[date_int, :, :]
    if sigma2.shape != shape:
        sigma2 = sigma2.transpose()
        if sigma2.shape != shape:
            raise ValueError(
                "Cannot align lsat error covariance matrix to grid "
                + f"for {year = }, {month = }"
            )
    return np.diag(sigma2.flatten())


def _get_obs_groups(
    data_path: str | None,
    var: str,
    year_range: tuple[int, int],
    member: int,
) -> dict[tuple[object, ...], pl.DataFrame]:
    if data_path is None:
        raise ValueError("'observations_path' key not set in config")

    def _read_file(member: int, data_path=data_path) -> pl.DataFrame:
        data_path = data_path.format(member=member)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"{data_path} cannot be found!")
        obs = pl.from_pandas(xr.open_dataset(data_path).to_dataframe().reset_index())
        obs = (
            obs.with_columns(
                [
                    pl.col("time").dt.year().alias("year"),
                    pl.col("time").dt.month().alias("month"),
                    pl.lit(member).alias("member"),
                ]
            )
            .filter(pl.col("year").is_between(*year_range, closed="both"))
            .filter(pl.col(var).is_not_nan() & pl.col(var).is_not_null())
            .unique(subset=["lon", "lat", "time", var])
            .select(
                [
                    "lon",
                    "lat",
                    "year",
                    "month",
                    "member",
                    pl.col(var).cast(pl.Float32),
                ]
            )
        )
        return obs

    df = _read_file(member=member, data_path=data_path)

    return df.partition_by(by=["year", "month"], as_dict=True)


def _initialise_xarray(
    grid: xr.DataArray,
    variable: str,
    year_range: tuple[int, int],
    member: int,
) -> xr.Dataset:
    # Reference date is start of the first year in the output data
    ref_date = datetime(year_range[0], 1, 1, 0, 0)
    # Time dimension is not unlimited
    # Mid-point of every month (Jan 1990 -> 1990-01-16 12:00)
    # Matches HadCRUT times
    _coords: dict = {
        "time": (
            (
                get_month_midpoint(
                    pl.datetime_range(
                        datetime(year_range[0], 1, 15, 12),
                        datetime(year_range[1], 12, 15, 12),
                        interval="1mo",
                        closed="both",
                        eager=True,
                    )
                )
                - ref_date
            ).dt.total_hours()
            / 24
        ).to_numpy()
    }
    # Add the spatial coordinates of the grid
    _coords.update({c: grid.coords[c].values for c in grid.coords})

    # Create a Coordinates object to use for DataArray creation (can't just use
    # dict)
    coords = xr.Coordinates(_coords)
    ds = xr.Dataset(
        coords=coords,
        attrs={
            "produced": str(datetime.today()),
            "produced_by": os.environ["USER"],
            "glomar_gridding_version": gg_version,
            "ensemble_member": str(member),
        },
    )

    # Define a 3D variable to hold the data
    ds[f"{variable}_anom"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom",
        attrs={
            "standard_name": f"infilled unperturbed {variable} anomaly",
            "long_name": f"infilled unperturbed {variable} anomaly",
            "units": "deg K",  # degrees Kelvin
        },
    )

    # Define a 3D variable to hold the data
    ds[f"{variable}_anom_uncert"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom_uncert",
        attrs={
            "standard_name": "kriging uncertainty",
            "long_name": f"{variable} anomaly uncertainty",
            "units": "deg K",  # degrees Kelvin
        },
    )

    # Define a 3D variable to hold the data
    ds[f"{variable}_alpha"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_alpha",
        attrs={
            "standard_name": f"kriging alpha for {variable}",
            "long_name": f"{variable} anomaly alpha",
            "units": "1",
        },
    )

    # Define a 3D variable to hold the data
    ds[f"{variable}_n_obs"] = xr.DataArray(
        coords=coords,
        name="n_obs",
        attrs={
            "standard_name": "Number of observations in each gridcell",
            "units": "",
        },
    )

    # Define a 3D variable to hold the epsilon perturbation value
    ds[f"{variable}_epsilon"] = xr.DataArray(
        coords=coords,
        name="epsilon",
        attrs={
            "standard_name": f"{variable} perturbation epsilon",
            "units": "deg K",
        },
    )

    ds[f"{variable}_anom_perturbed"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom_perturbed",
        attrs={
            "standard_name": f"infilled perturbed {variable} anomaly",
            "long_name": f"infilled perturbed {variable} anomaly",
            "units": "deg K",  # degrees Kelvin
        },
    )

    ds[f"{variable}_simulated_field"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_simulated_field",
        attrs={
            "standard_name": f"simulated {variable} anomaly field",
            "long_name": f"simulated {variable} anomaly field",
            "units": "deg K",  # degrees Kelvin
        },
    )

    ds[f"{variable}_simulated_obs"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_simulated_obs",
        attrs={
            "standard_name": f"simulated {variable} anomaly observation values",
            "long_name": f"simulated {variable} anomaly observation values",
            "units": "deg K",  # degrees Kelvin
        },
    )

    ds[f"{variable}_simulated_gridded"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_simulated_gridded",
        attrs={
            "standard_name": f"infilled simulated {variable} anomaly",
            "long_name": f"infilled simulated {variable} anomaly",
            "units": "deg K",  # degrees Kelvin
        },
    )

    # Update the attributes of the coordinates
    ds.lat.attrs["units"] = "degrees_north"
    ds.lat.attrs["long_name"] = "latitude"
    ds.lat.attrs["standard_name"] = "latitude"
    ds.lat.attrs["axis"] = "Y"

    ds.lon.attrs["units"] = "degrees_east"
    ds.lon.attrs["long_name"] = "longitude"
    ds.lon.attrs["standard_name"] = "longitude"
    ds.lon.attrs["axis"] = "X"

    ds.time.attrs["long_name"] = "time"
    ds.time.attrs["units"] = f"days since {ref_date.strftime('%Y-%m-%d')}"
    ds.time.attrs["calendar"] = "standard"

    return ds


def main():  # noqa: C901, D103
    config = _parse_args(parser)

    config["summary"] = {}
    config["summary"]["start"] = str(datetime.today())
    config["summary"]["user"] = os.environ["USER"]
    config["summary"]["glomar_gridding"] = gg_version
    config["summary"]["numpy"] = np.__version__
    config["summary"]["polars"] = pl.__version__
    config["summary"]["xarray"] = xr.__version__

    log_file: str | None = get_recurse(config, "setup", "log_file", default=None)
    init_logging(log_file)

    logging.info("Loaded configuration")

    # set boundaries for the domain
    lon_west: float = get_recurse(config, "domain", "west", default=0.0)
    lon_east: float = get_recurse(config, "domain", "east", default=360.0)
    lat_south: float = get_recurse(config, "domain", "south", default=-90.0)
    lat_north: float = get_recurse(config, "domain", "north", default=90.0)
    year_start: int = get_recurse(config, "domain", "startyear", default=1850)
    year_stop: int = get_recurse(config, "domain", "endyear", default=2024)
    member_start: int = get_recurse(config, "domain", "startmember", default=1)
    member_stop: int = get_recurse(config, "domain", "endmember", default=200)

    members = range(member_start, member_stop + 1)
    years = range(year_start, year_stop + 1)
    months = range(1, 13)

    output_grid: xr.DataArray = grid_from_resolution(
        resolution=5.0,
        bounds=[
            # Centre of each 5-degree box
            (lat_south + 2.5, lat_north + 2.5),
            (lon_west + 2.5, lon_east + 2.5),
        ],
        coord_names=["lat", "lon"],
        # dtype=np.float32,
    )
    shape = output_grid.shape
    output_lat: np.ndarray = output_grid.coords["lat"].values
    output_lon: np.ndarray = output_grid.coords["lon"].values

    logging.info("Initialised Output Grid")
    logging.info(f"{output_lat = }")
    logging.info(f"{output_lon = }")

    # what variable is being processed
    variable: str = get_recurse(config, "domain", "variable", default="sst")

    # path to output directory
    interpolation_covariance_type: str = "ellipse"
    interpolation_covariance_path: str | None = get_recurse(
        config, variable, "interpolation_covariance_path", default=None
    )
    if not interpolation_covariance_path:
        raise KeyError("Missing config value for 'interpolation_covariance_path'")
    output_directory: str = get_recurse(
        config, "output", "path", default=os.path.dirname(__file__)
    )
    output_directory = os.path.join(
        output_directory, interpolation_covariance_type, variable
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    logging.info(f"{output_directory = }")
    file_copy = os.path.join(output_directory, os.path.basename(__file__))
    config_copy = os.path.join(output_directory, "config.yaml")
    config["summary"]["file_copy"] = os.path.basename(file_copy)
    logging.info(f"Copying this file to {file_copy}")
    shutil.copyfile(os.path.abspath(__file__), file_copy)

    data_path: str = get_recurse(config, variable, "observations_path")

    error_covariance_path: str = get_recurse(config, variable, "error_covariance_path")

    interp_covariances = [
        load_array(
            interpolation_covariance_path,
            month=month,
            variable=variable,
        ).values
        for month in months
    ]
    logging.info("Loaded ellipse interpolation covariances")

    # Recover from a previous state
    state = "state" in config
    state_member = get_recurse(config, "state", "member", default=0)
    state_year = get_recurse(config, "state", "year", default=0)
    state_month = get_recurse(config, "state", "month", default=0)

    for member in members:
        if state and member < state_member:
            print(f"Skipping {member = }")
            continue

        match variable:
            case "sst":

                def _get_error_cov(year: int, month: int):
                    return _get_sst_error_cov(
                        year=year,
                        month=month,
                        error_covariance_path=error_covariance_path,
                    )

            case "lsat":
                member_err_cov_path = error_covariance_path.format(member=member)
                member_err_cov = load_array(member_err_cov_path, "sigma2").values

                def _get_error_cov(year: int, month: int):
                    return _get_lsat_error_cov(
                        year=year,
                        month=month,
                        error_cov=member_err_cov,
                        shape=shape,
                    )

            case _:
                raise NotImplementedError(
                    f"No method to get error covariance for {variable = }"
                )

        logging.info(f"Starting for ensemble member {member}")

        out_filename = f"kriged_member_{member:03d}.nc"
        out_filename = os.path.join(output_directory, out_filename)

        if state and member == state_member:
            logging.info("Recovering from State")
            ds = load_dataset(out_filename + ".bak")
        else:
            ds = _initialise_xarray(
                grid=output_grid,
                variable=variable,
                year_range=(year_start, year_stop),
                member=member,
            )

        yr_mo = _get_obs_groups(
            data_path,
            variable,
            year_range=(year_start, year_stop),
            member=member,
        )

        logging.info("Loaded Observations")

        for month in months:
            if state and member == state_member and month < state_month:
                print(f"Skipping {month = }")
                continue

            np.random.seed(_set_seed(member, month))
            interp_covariance = interp_covariances[month - 1]
            print(f"{interp_covariance = }")
            logging.info("Loaded ellipse interpolation covariance")

            simulated_states = scipy_mv_normal_draw(
                np.zeros(interp_covariance.shape[0]),
                interp_covariance,
                ndraws=len(years),
            )
            logging.info(
                "Done first random draws - full spatial from " + "interp_covariance"
            )

            for i_year, year in enumerate(years):
                if (
                    state
                    and member == state_member
                    and month == state_month
                    and year < state_year
                ):
                    print(f"Skipping {year = }")
                    continue

                # Save state if failure
                atexit.unregister(_save_state)
                atexit.register(
                    _save_state,
                    ds=ds,
                    ds_file=out_filename,
                    config=config,
                    config_file=config_copy,
                    member=member,
                    month=month,
                    year=year,
                )

                timestep = get_date_index(year, month, year_start)
                print("Current month and year: ", (month, year))

                # Get data subset for month / year
                mon_df: pl.DataFrame = yr_mo.get((year, month), pl.DataFrame())
                if mon_df.height == 0:
                    warnings.warn(
                        f"Current year, month ({year}, {month}) "
                        + "has no data. Skipping."
                    )
                    continue
                print(f"{mon_df = }")

                error_covariance = _get_error_cov(
                    year=year,
                    month=month,
                )
                logging.info("Got Error Covariance")
                print(f"{error_covariance = }")

                if len(years) > 1:
                    simulated_state = simulated_states[i_year]
                else:
                    simulated_state = simulated_states

                mon_df = map_to_grid(
                    mon_df,
                    output_grid,
                    grid_coords=["lat", "lon"],
                )
                logging.info("Aligned observations to output grid")

                # count obs per grid for output
                gridbox_counts = mon_df["grid_idx"].value_counts()
                grid_obs_2d = assign_to_grid(
                    gridbox_counts["count"].to_numpy(),
                    gridbox_counts["grid_idx"].to_numpy(),
                    output_grid,
                )

                grid_idx = mon_df.get_column("grid_idx").to_numpy()

                # Some nans in the error covariance
                error_cov_diag = pl.Series(
                    "err_cov", np.diag(error_covariance)[grid_idx]
                )
                mon_df = mon_df.with_columns(error_cov_diag)
                mon_df = mon_df.drop_nulls("err_cov").drop_nans("err_cov")

                grid_idx = mon_df.get_column("grid_idx").to_numpy()
                grid_obs = mon_df.get_column(variable).to_numpy()

                # Initialise Kriging Class
                stoch_krig = StochasticKriging(
                    interp_covariance,
                    idx=grid_idx,
                    obs=grid_obs,
                    error_cov=error_covariance,
                )

                # Perform Stochastic two-stage Kriging and collect outputs
                anom_perturbed = stoch_krig.solve(
                    simulated_state=simulated_state,
                )
                anom = stoch_krig.gridded_field
                epsilon = stoch_krig.epsilon
                uncert = stoch_krig.get_uncertainty()
                alpha = stoch_krig.constraint_mask()
                logging.info("Completed Stochastic Kriging")

                print(f"{anom = }")
                print(f"{np.all(np.isnan(anom)) = }")
                print(f"{np.any(np.isnan(anom)) = }")
                print("-" * 10)
                print(f"{uncert = }")
                print(f"{grid_obs_2d = }")

                # reshape output into 2D
                anom = np.reshape(anom, output_grid.shape).astype(np.float32)
                anom_perturbed = np.reshape(anom_perturbed, output_grid.shape).astype(
                    np.float32
                )
                uncert = np.reshape(uncert, output_grid.shape).astype(np.float32)
                epsilon = np.reshape(epsilon, output_grid.shape).astype(np.float32)
                alpha = np.reshape(alpha, output_grid.shape).astype(np.float32)
                sim_grid = stoch_krig.simulated_grid.reshape(output_grid.shape).astype(
                    np.float32
                )
                simulated_state = simulated_state.reshape(output_grid.shape).astype(
                    np.float32
                )

                logging.info("Reshaped kriging outputs")

                # Write the data.
                # This writes each time slice to the xarray
                ds[f"{variable}_anom"][timestep, :, :] = anom
                ds[f"{variable}_anom_perturbed"][timestep, :, :] = anom_perturbed
                ds[f"{variable}_anom_uncert"][timestep, :, :] = uncert
                ds[f"{variable}_epsilon"][timestep, :, :] = epsilon
                ds[f"{variable}_alpha"][timestep, :, :] = alpha
                ds[f"{variable}_n_obs"][timestep, :, :] = grid_obs_2d.astype(np.int16)
                ds[f"{variable}_simulated_gridded"][timestep, :, :] = sim_grid
                ds[f"{variable}_simulated_field"][timestep, :, :] = simulated_state
                ds[f"{variable}_simulated_obs"][timestep, :, :] = assign_to_grid(
                    stoch_krig.simulated_obs,
                    grid_idx=stoch_krig.idx,
                    grid=output_grid,
                ).astype(np.float32)

                logging.info(f"Wrote data for {year = }, {month = }")

            # first print the Dataset object to see what we've got
            # close the Dataset.
        ds.to_netcdf(out_filename, unlimited_dims=["time"])
        atexit.unregister(_save_state)
        logging.info("Dataset is closed!")

    config["summary"]["end"] = str(datetime.today())
    logging.info("DONE")
    with open(config_copy, "w") as io:
        yaml.safe_dump(config, io)


if __name__ == "__main__":
    main()
