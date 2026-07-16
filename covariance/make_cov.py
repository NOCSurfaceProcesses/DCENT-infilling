#!/usr/bin/env python

"""Make the temperature covariance from the ellipses."""

import argparse
from datetime import datetime
from typing import Literal
import yaml

from pathlib import Path

import numpy as np
import scipy as sp
import xarray as xr

from glomar_gridding.ellipse import EllipseModel, EllipseCovarianceBuilder
from glomar_gridding.covariance_tools import eigenvalue_clip
from glomar_gridding.grid import cross_coords
from glomar_gridding.io import get_recurse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    dest="config",
    required=True,
    type=str,
    help="configuration yaml file to use",
)


def resolve_svalbard(
    Lx: np.ndarray,
    Ly: np.ndarray,
    month: int,
    max_allowed: float = 29995.0,
) -> np.ndarray:
    """Resolve a large value near Svalbard by setting Lx -> Ly."""
    # Do big Lx check (aka Svalbard fudge)
    Lx_max = np.nanmax(Lx)
    if Lx_max <= max_allowed:
        return Lx
    above_allowed_idx = np.where(Lx > max_allowed)
    num_above_allowed = len(above_allowed_idx[0])
    if num_above_allowed > 1:
        raise ValueError("More than 1 value above maximum allowed value")

    # Expect the position to be around Svalbard
    expected_pos = (np.array([33]), np.array([37]))
    if above_allowed_idx != expected_pos:
        raise ValueError("Maximum value is not in expected position (Svalbard)")

    print(f"Svalbard fudge applied; {month = }")
    Lx[above_allowed_idx] = Ly[above_allowed_idx]
    print(f"{Lx[above_allowed_idx] = }, {Ly[above_allowed_idx]}")

    return Lx


def resolve_kergulen(
    Lx: np.ndarray,
    Ly: np.ndarray,
    month: int,
    max_allowed: float = 29995.0,
) -> np.ndarray:
    """Resolve a large value near Svalbard by setting Lx -> Ly."""
    # Do big Lx check (aka Svalbard fudge)
    Lx_max = np.nanmax(Lx)
    if Lx_max <= max_allowed:
        return Lx
    above_allowed_idx = np.where(Lx > max_allowed)
    num_above_allowed = len(above_allowed_idx[0])
    if num_above_allowed > 1:
        raise ValueError("More than 1 value above maximum allowed value")

    # Expect the position to be around Svalbard
    expected_pos = (np.array([8]), np.array([49]))
    if above_allowed_idx != expected_pos:
        raise ValueError("Maximum value is not in expected position (Svalbard)")

    print(f"Svalbard fudge applied; {month = }")
    Lx[above_allowed_idx] = Ly[above_allowed_idx]
    print(f"{Lx[above_allowed_idx] = }, {Ly[above_allowed_idx]}")

    return Lx


def generate_output_ds(coords: xr.Coordinates) -> xr.Dataset:
    """Generate the covariance shaped empty dataset with appropriate coordinates."""
    out_coords = cross_coords(coords)
    out_coords["index_1"].attrs = {
        "units": "1",
    }
    out_coords["latitude_1"].attrs = {
        "long_name": "latitude_1",
        "units": "degrees_north",
    }
    out_coords["longitude_1"].attrs = {
        "long_name": "longitude_1",
        "units": "degrees_north",
    }
    out_coords["index_2"].attrs = {
        "units": "1",
    }
    out_coords["latitude_2"].attrs = {
        "long_name": "latitude_2",
        "units": "degrees_north",
    }
    out_coords["longitude_2"].attrs = {
        "long_name": "longitude_2",
        "units": "degrees_north",
    }
    return xr.Dataset(coords=out_coords)


def summarise_cov(spatial_cov: EllipseCovarianceBuilder) -> dict:
    """Generate summary information for the covariance."""
    cov_mat = spatial_cov.cov_ns.copy()
    eigvals = sp.linalg.eigvalsh(cov_mat)
    summary = {
        "shape": cov_mat.shape,
        "determinant": np.linalg.det(cov_mat),
        "trace": np.trace(cov_mat),
        "smallest_eigv": np.min(eigvals),
        "largest_eigv": np.max(eigvals),
        "eigvals": eigvals,
    }
    return summary


def adjust_positive_def(
    spatial_cov: EllipseCovarianceBuilder,
    eigenvalue_clip_method: Literal["explained_variance", "laloux_clip"],
    **eigenvalue_clip_kwargs,
) -> EllipseCovarianceBuilder:
    """Adjust the covariance matrix to positive definite by eigenvalue clipping."""
    summary = summarise_cov(spatial_cov)

    if summary.get("smallest_eigv", 0.0) < 0:
        for k, v in summary.items():
            if k == "eigvals":
                continue
            print(f"Before: {k} = {v}")
        spatial_cov.cov_ns = eigenvalue_clip(
            spatial_cov.cov_ns,
            eigenvalue_clip_method,  # type: ignore
            **eigenvalue_clip_kwargs,
        )
        revised_summary = summarise_cov(spatial_cov)
        for k, v in revised_summary.items():
            if k == "eigvals":
                continue
            print(f"Revised: {k} = {v}")
    else:
        print("No need to adjust eigenvalues")

    return spatial_cov


def expand_covariance(
    spatial_cov: EllipseCovarianceBuilder,
    diag_fill_value: float,
    off_diag_fill_value: float,
) -> EllipseCovarianceBuilder:
    """Expand from compressed/masked covariance to cover full grid."""
    old_size = spatial_cov.cov_ns.shape
    print(f"{old_size = }")
    spatial_cov.uncompress_cov(
        diag_fill_value=diag_fill_value,
        fill_value=off_diag_fill_value,
    )
    new_size = spatial_cov.cov_ns.shape
    print(f"{new_size = }")
    return spatial_cov


def main():  # noqa: D103
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config = yaml.safe_load(io)

    variable = config.get("variable", "sst")
    basepath = Path(config["io"]["base_path"])

    parmfile_template = config["io"]["ellipse_file"]
    outfile_180_template = config["io"]["cov_180_file"]
    outfile_360_template = config["io"]["cov_360_file"]

    ellipse = EllipseModel(**config.get("model_params", {}))
    v = ellipse.v

    cov_params = config.get("cov", {})
    max_dist_cov = cov_params.get("max_dist", 6_000.0)
    max_allowed_lx = cov_params.get("max_allowed_lx", 29995.0)
    fill_before_clipping = cov_params.get("fill_before_clipping", False)
    diag_fill_value_ = cov_params.get("diag_fill_value")
    off_diag_fill_value = cov_params.get("off_diag_fill_value", 0.0)

    eigenvalue_clip_method = cov_params.get(
        "eigenvalue_clip_method", "explained_variance"
    )
    eigenvalue_clip_kwargs = get_recurse(
        cov_params, "clip_params", eigenvalue_clip_method, default={}
    )

    for month in range(1, 13):
        # Solve filenames
        parmfile = basepath / parmfile_template.format(month=month)
        outfile = basepath / outfile_180_template.format(month=month)
        print(f"{month = }")
        print(f"{parmfile = }")
        print(f"{outfile = }")
        outdir = outfile.parent
        if not outdir.isdir():
            outdir.mkdir(parents=True, exist_ok=True)

        # Create covariance
        parm_ds = xr.open_dataset(parmfile)
        Lx_da = parm_ds["Lx"]
        Lx = parm_ds["Lx"].values
        Ly = parm_ds["Ly"].values
        theta = parm_ds["theta"].values
        stdev = parm_ds["standard_deviation"].values
        mask = np.logical_or(np.isnan(Lx), Lx < 0.0)
        coords = Lx_da.coords

        out_ds = generate_output_ds(coords)

        # Handle specific bad-value cases
        if variable == "sst":
            Lx = resolve_svalbard(Lx, Ly, month=month, max_allowed=max_allowed_lx)
        elif variable == "lsat":
            Lx = resolve_kergulen(Lx, Ly, month=month, max_allowed=max_allowed_lx)

        start_time = datetime.now()
        spatial_cov = EllipseCovarianceBuilder(
            np.ma.masked_where(mask, Lx),
            np.ma.masked_where(mask, Ly),
            np.ma.masked_where(mask, theta),
            np.ma.masked_where(mask, stdev),
            coords["latitude"].values,
            coords["longitude"].values,
            v=ellipse.v,
            max_dist=max_dist_cov,
            covariance_method="batched",
            batch_size=100_000,
        )

        current_diag_values = np.diag(spatial_cov.cov_ns)
        diag_fill_value = (
            float(np.mean(current_diag_values))
            if diag_fill_value_ is None
            else diag_fill_value_
        )
        print(f"{diag_fill_value = }")
        if fill_before_clipping:
            spatial_cov = expand_covariance(
                spatial_cov, diag_fill_value, off_diag_fill_value
            )
            spatial_cov = adjust_positive_def(
                spatial_cov, eigenvalue_clip_method, **eigenvalue_clip_kwargs
            )
        else:
            spatial_cov = adjust_positive_def(
                spatial_cov, eigenvalue_clip_method, **eigenvalue_clip_kwargs
            )
            spatial_cov = expand_covariance(
                spatial_cov, diag_fill_value, off_diag_fill_value
            )

        summary = summarise_cov(spatial_cov)
        for k, v in summary.items():
            if k == "eigvals":
                continue
            print(f"Final: {k} = {v}")
        end_time = datetime.now()
        print(f"Month {month} took {end_time - start_time}")

        # Create correlation matrix
        spatial_cov.calculate_cor()

        out_ds["covariance"] = (
            ["index_1", "index_2"],
            spatial_cov.cov_ns,
            {"long_name": "covariance", "units": "K**2"},
        )
        out_ds["correlation"] = (
            ["index_1", "index_2"],
            spatial_cov.cor_ns,
            {"long_name": "correlation", "units": "1"},
        )

        out_ds["nu"] = v
        out_ds["nu"].attrs = {"long_name": "mattern_nu", "units": "1"}

        out_ds["diag_fill_value"] = diag_fill_value
        out_ds["diag_fill_value"].attrs = {
            "long_name": "diagonal_fill_value_for_matrix_expansion",
            "units": "K**2",
        }
        out_ds["smallest_eigv"] = summary["smallest_eigv"]
        out_ds["smallest_eigv"].assign_attrs(
            long_name="smallest_eigenvalue", units="K**2"
        )
        out_ds["largest_eigv"] = summary["largest_eigv"]
        out_ds["largest_eigv"].assign_attrs(
            long_name="largest_eigenvalue", units="K**2"
        )
        out_ds["trace"] = summary["trace"]
        out_ds["trace"].assign_attrs(long_name="total_variance", units="K**2")
        out_ds["pd_check"] = True
        out_ds["pd_check"].assign_attrs(long_name="positive_semidefinite_check_enabled")

        print(f"{month}: Saving to {outfile}")
        out_ds.to_netcdf(outfile)

        # Remaps the file covariance to 0 - 360 and save that as well
        outfile_360 = basepath / outfile_360_template.format(month=month)
        print(f"{outfile_360 = }")
        out_ds_360 = out_ds.copy()
        out_ds_360.coords["longitude_1"] = out_ds_360.coords["longitude_1"] % 360
        out_ds_360.coords["longitude_2"] = out_ds_360.coords["longitude_2"] % 360
        out_ds_360 = out_ds_360.sortby(["latitude_1", "longitude_1"]).sortby(
            ["latitude_2", "longitude_2"]
        )
        out_ds_360.coords["index_1"] = np.arange(spatial_cov.cov_ns.shape[0])
        out_ds_360.coords["index_2"] = np.arange(spatial_cov.cov_ns.shape[0])

        out_ds_360.to_netcdf(outfile_360)

    print("Complete")


if __name__ == "__main__":
    main()
