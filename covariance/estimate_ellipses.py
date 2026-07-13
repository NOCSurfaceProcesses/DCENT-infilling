"""Get SST ellipse parameters."""

import argparse
import yaml

from pathlib import Path

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


def main():  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)

    variable = config.get("variable", "sst")

    inpath = Path(config["io"]["base_path"])
    infile = inpath / config["io"]["train_file"]

    varname = config["io"]["train_var"]
    da = xr.load_dataset(infile)[varname]

    if "lat" in da.coords:
        da = da.rename({"lat": "latitude", "lon": "longitude"})

    outfile_template = config["io"]["ellipse_file"]
    for month in range(1, 13):
        outfile = inpath / outfile_template.format(month=month)
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

        ellipse = EllipseModel(**config["model_params"])
        ellipse_builder = EllipseBuilder(training_arr, coords)
        fit_config = config.get("fit_params", {})
        use_hadcrut_defaults = fit_config.get("HadCRUT5_defaults", False)
        if not use_hadcrut_defaults:
            fill_val_L = -999.9
            fill_val_theta = -999.9
            fill_val_stdev = -999.9
        elif variable == "sst":
            fill_val_L = np.sqrt(2) * 1300.0
            fill_val_theta = 0.0
            fill_val_stdev = 0.6
        elif variable == "lsat":
            fill_val_L = np.sqrt(2) * 1300.0
            fill_val_theta = 0.0
            fill_val_stdev = 1.2
        else:
            raise ValueError("Could not set default values")

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
            (-2.0 * np.pi, 2.0 * np.pi),
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
        if not use_hadcrut_defaults:
            for varname, default_value in zip(list_of_vars, default_values):
                ellipse_params[varname] = ellipse_params[varname].where(
                    ellipse_params[varname] != default_value
                )

        ellipse_params.to_netcdf(outfile)
    print("Complete")


if __name__ == "__main__":
    main()
