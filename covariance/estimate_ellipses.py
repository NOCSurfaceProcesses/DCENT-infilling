"""Get SST ellipse parameters."""

import argparse
import yaml

from pathlib import Path
from math import tau

import numpy as np
import xarray as xr

from glomar_gridding.ellipse import EllipseModel, EllipseBuilder

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=True,
    type=str,
    help="configuration yaml file to use",
)


def default_fill_value(
    config: dict,
    variable: str = "sst",
) -> tuple[float, float, float]:
    """Get the default fill values."""
    use_hadcrut_defaults = config.get("HadCRUT_defaults", False)

    if not use_hadcrut_defaults:
        return (
            config.get("L_fill_value", -999.9),
            config.get("theta_fill_value", -999.9),
            config.get("stdev_fill_value", -999.9),
        )
    elif variable == "sst":
        return np.sqrt(2) * 1300.0, 0.0, 0.6
    elif variable == "lsat":
        return np.sqrt(2) * 1300.0, 0.0, 1.2
    else:
        raise ValueError("Could not set default values")


def main():  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)

    variable = config.get("variable", "sst")

    out_path = Path(config["io"]["base_path"])
    infile = Path(config["io"]["train_file"])

    varname = config["io"]["train_var"]
    da = xr.load_dataset(infile)[varname]

    if "lat" in da.coords:
        da = da.rename({"lat": "latitude", "lon": "longitude"})

    outfile_template = config["io"]["ellipse_file"]
    for month in range(1, 13):
        outfile = out_path / outfile_template.format(month=month)
        print(f"{outfile = }")

        da_mini = da.sel(time=(da.time.dt.month == month))

        coords = da_mini.coords
        print(repr(da_mini))
        training_arr = np.ma.masked_greater(da_mini.values, 1e5)
        training_arr = np.ma.masked_where(
            np.broadcast_to(np.any(training_arr.mask, axis=0), training_arr.shape),
            training_arr,
        )
        training_arr_mean = np.mean(training_arr, axis=0)
        training_arr = training_arr - training_arr_mean
        print(repr(da_mini.time))
        print(training_arr.shape)

        # Mask check
        mask_check = np.all(
            np.all(training_arr[0].mask == training_arr.mask, axis=0),
        )
        if not np.all(mask_check):
            raise ValueError("Mask check fail")

        ellipse = EllipseModel(**config.get("model_params", {}))
        ellipse_builder = EllipseBuilder(training_arr, coords)
        fit_config = config.get("fit_params", {})

        fill_val_L, fill_val_theta, fill_val_stdev = default_fill_value(
            fit_config, variable
        )

        default_values = [
            fill_val_L,  # lx
            fill_val_L,  # ly
            fill_val_theta,  # theta
            fill_val_stdev,  # stdev
            -1,  # success
            -1,  # niter
        ]
        init_values = [
            fit_config.get("init_value_Lx", 2_000.0),
            fit_config.get("init_value_Ly", 2_000.0),
            fit_config.get("init_value_theta", 0.0),
        ]
        # Uniformative prior of parameter range
        fit_bounds = [
            (
                fit_config.get("Lx_lower_bound", 30_000.0),
                fit_config.get("Lx_upper_bound", 30_000.0),
            ),
            (
                fit_config.get("Ly_lower_bound", 300.0),
                fit_config.get("Ly_upper_bound", 300.0),
            ),
            (
                fit_config.get("theta_lower_bound", -tau),
                fit_config.get("theta_upper_bound", tau),
            ),
        ]
        fit_max_distance = fit_config.get("fit_max_distance", 10_000.0)
        ellipse_params = ellipse_builder.compute_params(
            default_value=default_values,
            matern_ellipse=ellipse,
            max_distance=fit_max_distance,
            guesses=init_values,
            bounds=fit_bounds,
        )
        list_of_vars = list(ellipse_params.keys())
        if len(list_of_vars) != len(default_values):
            raise ValueError(
                "Mismatch between number of parameters and expected number  "
                + "of parameters: "
                + f"Expected: {len(default_values)}, got: {len(list_of_vars)}."
            )
        if not fit_config.get("HadCRUT_defaults", False):
            for varname, default_value in zip(list_of_vars, default_values):
                ellipse_params[varname] = ellipse_params[varname].where(
                    ellipse_params[varname] != default_value
                )

        ellipse_params.to_netcdf(outfile)
    print("Complete")


if __name__ == "__main__":
    main()
