#!/usr/bin/env python3
"""
Prepare NetCDF Release Files.
----------------------------

Low-memory streaming ensemble processing:
 - Reads one file per ensemble member
 - Computes ensemble mean and spread per time slice across ensemble
 - Writes:
     * stats_out: single NetCDF containing mean, std, p05, p95 across ensemble members
     * per-member files (each with original member data but CF metadata from YAML)
 - Reads metadata from YAML (general, coordinates and variables)
 - Output filenames use general.name and general.version
 - Command-line arguments:
     * Only --config and --out_dir are mandatory
     * The other values override values provided in the yaml file
"""

import glob
import calendar
import argparse
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm
import re
from itertools import islice


def batch(iterable, batch_size):
    """Yield successive batches of size `batch_size` from `iterable`."""
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def compute_bounds(coord_vals):
    """Auto calculate coordinate bounds."""
    bounds = np.zeros((len(coord_vals), 2), dtype=coord_vals.dtype)
    diffs = np.diff(coord_vals) / 2.0
    bounds[1:, 0] = coord_vals[:-1] + diffs
    bounds[:-1, 1] = coord_vals[:-1] + diffs
    bounds[0, 0] = coord_vals[0] - diffs[0]
    bounds[-1, 1] = coord_vals[-1] + diffs[-1]
    return bounds


def get_month_bounds(dt):
    """Auto calculate time bounds."""
    if isinstance(dt, np.datetime64):
        dt = pd.to_datetime(str(dt))
    year, month = dt.year, dt.month
    start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)
    return start, end


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Process NetCDF ensemble with CF-1.7 metadata and stats."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--input_glob", help="Override input file glob pattern.")
    parser.add_argument("--varname", help="Override new variable name.")
    parser.add_argument("--nbatch", help="Length of batch sizes.")
    parser.add_argument("--varname_old", help="Override original variable name.")
    return parser.parse_args()


def main():
    """Process input files, and write out reformatted files."""
    args = parse_args()
    out_dir = args.out_dir

    # Load YAML config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg["general"]["name"]
    old_name = args.varname_old or cfg["variable"]["old_name"]
    version = cfg["general"]["version"]
    varname = args.varname or cfg["variable"]["name"]
    var_attrs = cfg["variable"].get("attributes", {})
    aux_attrs = cfg.get("aux", {})
    coord_attrs = cfg.get("coordinates", {})
    setup_attrs = cfg.get("setup", {})
    calendar = coord_attrs["time"]["calendar"]
    nbatch = args.nbatch or setup_attrs["nbatch"]
    nbatch = int(nbatch)
    coord_long_name = {"lat": "latitude", "lon": "longitude"}
    global_attrs = cfg.get("general", {})
    global_attrs["history"] = datetime.now().strftime("Built on %Y-%m-%d at %H:%M:%S")

    input_glob = args.input_glob or setup_attrs["input_glob"]
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found matching: {input_glob}")

    # Open initial file to get time range, dims and auxiliary variables
    ds0 = xr.open_dataset(files[0])
    time_vals = ds0["time"].values
    ntimes = len(time_vals)
    spatial_dims = [d for d in ds0[old_name].dims if d not in ["time", "member"]]

    out_stats_path = out_dir + f"/{name}_{version}_mean_spread.nc"
    nc_stats = Dataset(out_stats_path, "w")
    nc_stats.setncatts(global_attrs)

    nc_stats.createDimension("time", None)
    for d in spatial_dims:
        nc_stats.createDimension(d, len(ds0[d]))
    nc_stats.createDimension("bnds", 2)

    # Set the time axis
    time_units = coord_attrs["time"]["units"]
    time_origin = re.sub(r"[a-zA-Z]", "", time_units)
    time_origin = pd.Timestamp(time_origin)

    time_var = nc_stats.createVariable("time", "f8", ("time",))
    time_bnds_var = nc_stats.createVariable("time_bnds", "f8", ("time", "bnds"))
    time_var.units = time_units
    time_var.calendar = calendar
    time_var.bounds = "time_bnds"
    if "time" in coord_attrs:
        time_var.setncatts(coord_attrs["time"])

    tstart, tend = [], []
    for dt in time_vals:
        start, end = get_month_bounds(dt)
        t0 = (
            start - datetime(time_origin.year, time_origin.month, time_origin.day)
        ).total_seconds() / 86400.0
        t1 = (
            end - datetime(time_origin.year, time_origin.month, time_origin.day)
        ).total_seconds() / 86400.0
        tstart.append(t0)
        tend.append(t1)

    time_var[:] = (pd.to_datetime(time_vals) - time_origin) / pd.Timedelta("1D")
    time_bnds_var[:, 0] = tstart
    time_bnds_var[:, 1] = tend

    # Set coordinate variables
    for d in spatial_dims:
        vals = ds0[d].values
        dtype = "f8" if np.issubdtype(vals.dtype, np.floating) else "i4"
        var = nc_stats.createVariable(d, dtype, (d,))
        var[:] = vals
        if d in coord_attrs:
            var.setncatts(coord_attrs[d])
        bname = f"{d}_bnds"
        if vals.ndim == 1 and len(vals) > 1:
            bvals = compute_bounds(vals)
            bvar = nc_stats.createVariable(bname, "f8", (d, "bnds"))
            bvar[:, :] = bvals
            var.bounds = bname
            long_name = coord_long_name[d]
            bvar.setncattr("standard_name", f"{long_name}_bounds")

    ds0.close()

    # Set-up containers for ensemble stats
    v_mean = nc_stats.createVariable(
        f"{varname}_mean", "f8", ("time", *spatial_dims), zlib=True
    )
    v_std = nc_stats.createVariable(
        f"{varname}_std", "f8", ("time", *spatial_dims), zlib=True
    )
    v_p05 = nc_stats.createVariable(
        f"{varname}_p05", "f8", ("time", *spatial_dims), zlib=True
    )
    v_p95 = nc_stats.createVariable(
        f"{varname}_p95", "f8", ("time", *spatial_dims), zlib=True
    )

    # Stats Variable attributes
    stat_vars = [v_mean, v_std, v_p05, v_p95]
    stats = ["mean", "standard deviation", "5th percentile", "95th percentile"]
    mean_method = "time mean; area mean; realization "

    for v, sname in zip(stat_vars, stats):
        v.setncatts(var_attrs)
        v.setncattr("cell_methods", mean_method + sname)

    # Stats realization dim
    nc_stats.createDimension("realization", 1)
    ens_stats = nc_stats.createVariable("realization", "i4", ("realization"))
    ens_stats.units = "1"
    ens_stats.standard_name = "realization"
    ens_stats.realization_bound = "realization_bnds"
    ens_stats_bnds = nc_stats.createVariable("realization_bnds", "i4", ("bnds"))
    ens_stats_bnds[:] = [1, len(files)]

    # Loop through time slices and create stats
    # -----------------------------------------
    nc_files = [xr.open_dataset(f) for f in files]

    nbatches = ntimes // nbatch + (ntimes % nbatch > 0)

    for t_idx in tqdm(
        batch(range(ntimes), nbatch), total=nbatches, desc="Running stats calculations"
    ):
        slices = []

        for dsi in nc_files:
            arr = dsi[old_name].isel(time=t_idx).values.astype("f8")
            slices.append(arr)

        stack = np.stack(slices, axis=0)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        p05 = np.nanpercentile(stack, 5.0, axis=0)
        p95 = np.nanpercentile(stack, 95.0, axis=0)

        v_mean[t_idx, ...] = mean.astype("f8")
        v_std[t_idx, ...] = std.astype("f8")
        v_p05[t_idx, ...] = p05.astype("f8")
        v_p95[t_idx, ...] = p95.astype("f8")

    for f in nc_files:
        f.close()

    nc_stats.close()

    # Loop through ensemble files
    # ----------------------------
    for i, f in enumerate(tqdm(files, desc="Processing ensemble files")):
        dsm = xr.open_dataset(f)
        out_path = out_dir + f"/{name}_{version}_member_{i + 1:03d}.nc"
        nc = Dataset(out_path, "w")
        nc.setncatts(global_attrs)

        nc.createDimension("time", None)
        for d in spatial_dims:
            nc.createDimension(d, len(dsm[d]))
        nc.createDimension("bnds", 2)

        tv = nc.createVariable("time", "f8", ("time",))
        tv_bnds = nc.createVariable("time_bnds", "f8", ("time", "bnds"))
        tv.units = time_units
        tv.calendar = calendar
        tv.bounds = "time_bnds"
        if "time" in coord_attrs:
            tv.setncatts(coord_attrs["time"])

        for d in spatial_dims:
            vals = dsm[d].values
            dtype = "f8" if np.issubdtype(vals.dtype, np.floating) else "i4"
            var = nc.createVariable(d, dtype, (d,))
            var[:] = vals
            if d in coord_attrs:
                var.setncatts(coord_attrs[d])
            bname = f"{d}_bnds"
            if vals.ndim == 1 and len(vals) > 1:
                bvals = compute_bounds(vals)
                bvar = nc.createVariable(bname, "f8", (d, "bnds"))
                bvar[:, :] = bvals
                var.bounds = bname
                long_name = coord_long_name[d]
                bvar.setncattr("standard_name", f"{long_name} bounds")

        dsm.close()

        dv = nc.createVariable(varname, "f8", ("time", *spatial_dims), zlib=True)
        dv.setncatts(var_attrs)
        dv.setncattr("cell_methods", "time mean; area mean")

        nc.createDimension("realization", 1)
        ens = nc.createVariable("realization", "i4", ("realization"))
        ens.units = "1"
        ens.standard_name = "realization"
        ens[:] = i + 1

        tv[:] = (pd.to_datetime(time_vals) - time_origin) / pd.Timedelta("1D")
        dv[:] = dsm[old_name]
        tv_bnds[:, 0] = tstart
        tv_bnds[:, 1] = tend
        nc.close()

    # Create auxiliary variables file

    aux_path = out_dir + f"/{name}_{version}_diagnostics.nc"

    nc = Dataset(aux_path, "w")
    nc.setncatts(global_attrs)

    nc.createDimension("time", None)
    for d in spatial_dims:
        nc.createDimension(d, len(ds0[d]))
    nc.createDimension("bnds", 2)

    tv = nc.createVariable("time", "f8", ("time",))
    tv_bnds = nc.createVariable("time_bnds", "f8", ("time", "bnds"))
    tv.units = time_units
    tv.calendar = calendar
    tv.bounds = "time_bnds"
    if "time" in coord_attrs:
        tv.setncatts(coord_attrs["time"])

    for d in spatial_dims:
        vals = ds0[d].values
        dtype = "f8" if np.issubdtype(vals.dtype, np.floating) else "i4"
        var = nc.createVariable(d, dtype, (d,))
        var[:] = vals
        if d in coord_attrs:
            var.setncatts(coord_attrs[d])
        bname = f"{d}_bnds"
        if vals.ndim == 1 and len(vals) > 1:
            bvals = compute_bounds(vals)
            bvar = nc.createVariable(bname, "f8", (d, "bnds"))
            bvar[:, :] = bvals
            var.bounds = bname
            long_name = coord_long_name[d]
            bvar.setncattr("standard_name", f"{long_name} bounds")

    tv[:] = (pd.to_datetime(time_vals) - time_origin) / pd.Timedelta("1D")
    tv_bnds[:, 0] = tstart
    tv_bnds[:, 1] = tend

    aux = ["sst_alpha", "lsat_alpha", "sst_n_obs", "lsat_n_obs", "weight"]

    for v in aux:
        var_attrs = aux_attrs[v]

        try:
            new_name = var_attrs["new_name"]
            var_attrs.pop("new_name")
        except KeyError:
            new_name = v

        if v == "weight":
            wgt_nc = xr.open_dataset(aux_attrs["weights_path"])
            dv = nc.createVariable(v, "f8", ("time", *spatial_dims), zlib=True)
            data = wgt_nc[v]
            wgt_nc.close()
        else:
            dv = nc.createVariable(new_name, "f8", ("time", *spatial_dims), zlib=True)
            data = ds0[v]

        dv.setncatts(var_attrs)
        dv[:] = data

    nc.close()
    ds0.close()


if __name__ == "__main__":
    main()
