#!/usr/bin/env python

"""Append new data."""

import argparse
import os
from pathlib import Path
import xarray as xr
import yaml


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Append data from concat to base stream, saving to out stream."
    )
    parser.add_argument(
        "--yaml",
        required=True,
        help="yaml file with stream information.",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "append_new_data.yaml"),
    )
    parser.add_argument(
        "-s",
        "--stream",
        help="data stream / variable to process.",
        type=str,
        choices=["sst", "lsat", "tas", "super", "ts"],
        default="tas",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="dry run; does not save output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more debugging",
    )
    return parser.parse_args()


def main():
    """Append data update to old data."""
    args = parse_args()
    yaml_file = Path(args.yaml)
    stream = args.stream
    dry_run = args.dry_run
    verbose = args.verbose

    if verbose:
        print(yaml_file)
        print(stream)
        print(dry_run)

    with open(yaml_file, "r") as io:
        config: dict = yaml.safe_load(io)

    filename_pattern = config.get("filename_pattern", "DCENT_I_1.1.0.0_*.nc")
    base_path = Path(config["base"].get(stream))
    base_files = sorted(base_path.glob(filename_pattern))

    update_path = Path(config["update"].get(stream))
    out_path = Path(config["out"].get(stream))

    for i, base_file in enumerate(base_files):
        basename = base_file.parts[-1]

        update_file = update_path / basename
        out_file = out_path / basename

        if not update_file.is_file():
            raise FileNotFoundError(f"{update_file} not found")

        if verbose:
            print(f"{i} : {base_file = }")
            print(f"{i} : {update_file = }")
            print(f"{i} : {out_file = }")

        base_ds = xr.open_dataset(base_file)
        update_ds = xr.open_dataset(update_file)
        updated_ds = xr.merge([base_ds, update_ds])
        # Ensure variables are 'tas'
        if stream == "tas":
            if "mean_spread" in basename:
                updated_ds = updated_ds.rename(
                    {"tas_" + d: "ts_" + d for d in ["mean", "std", "p05", "p95"]}
                )
            elif "member" in basename:
                updated_ds = updated_ds.rename({"tas": "ts"})
            elif "diagnostics" in basename:
                pass
            else:
                raise IOError(f"Unknown file {base_file}")

        if dry_run:
            print(updated_ds)
            continue
        encoding = {
            varname: {"zlib": True, "complevel": 9} for varname in updated_ds.keys()
        }
        updated_ds.to_netcdf(out_file, encoding=encoding)


if __name__ == "__main__":
    main()
