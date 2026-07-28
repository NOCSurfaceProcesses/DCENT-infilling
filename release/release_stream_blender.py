#!/usr/bin/env python
"""
Merge tas, sst, and lsat streams into a single "super" stream.

The original post-processing code process SST, lsat, and their
weighted average separately, and save them and their summary
statistics and auxiliaries to separate directories.

This code puts them back into a single "super" directory.
"""
import argparse
import os
import shutil
import xarray as xr


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Organise DCENT-I NetCDF ensemble and their summary stats "
        "and diagnostic files, patches CF-1.7 metadata if necessary."
    )
    parser.add_argument(
        "--tas_stream",
        required=True,
        help="Path to tas files.",
    )
    parser.add_argument(
        "--sst_stream",
        required=True,
        help="Path to sst files.",
    )
    parser.add_argument(
        "--lsat_stream",
        required=True,
        help="Path to lsat files.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def file_check(file_name) -> None:
    """Check file existence."""
    check = os.path.isfile(file_name)
    print(f"{file_name}: check {check}")
    if not check:
        raise IOError('Required file is not found.')


def get_file_list(
        file_path: str,
        variable_name: str = "tas",
        emin: int = 1,
        emax: int = 200,
    ) -> tuple[list, dict]:
    """
    Returns a list and dict of files needed for processing.

    Parameters
    ----------
    file_path: str
        The path where the files to be processed are
    variable_name: str
        The variable to be processed
    emin: int
        The first (minimum) ensemble member ID
    emax: int
        The last (maximum) ensemble member ID

    Returns
    -------
    ens_files: list
        A list of ensemble member files with its existence confirmed
    files_to_cp: dict
        Dictionary of files to be copied; this only applies to:
        - file that is used store ensemble spread
        - basic diagnostic like land-sea weights and the Met Office alpha parameter
    """
    #
    # Populate ensemble member filenames of variable stream into a list
    es = range(emin, emax + 1)
    ens_files = [f"{file_path}/DCENT_I_1.1.0.0_member_{e:03}.nc" for e in es]
    for ens_file in ens_files:
        file_check(ens_file)
    #
    # Dictionary of ensemble-mean/spread and diagnostic files
    # to be copied and their new post-copied filenames
    #
    # For SST and LSAT, files_to_cp only has 1 key
    # as only ensemble-mean/spread file is copied
    #
    # Only for ts/tas, it has two keys.
    #
    # Key is original filename with its original path;
    # value is the new filename within output path.
    files_to_cp = {}
    #
    # Ensemble spread and mean (applies to all 3 streams)
    # Note: tas >>> ts
    prefix = 'DCENT_I_1.1.0.0_mean_spread'
    file_check(f'{file_path}/{prefix}.nc')
    variable_name_ = "ts" if variable_name == "tas" else variable_name
    files_to_cp[f'{file_path}/{prefix}.nc'] = f'{prefix}_{variable_name_}.nc'
    #
    # The diagnostic file; copying only applies to tas/ts
    if variable_name == "tas":
        prefix = 'DCENT_I_1.1.0.0_diagnostics'
        file_check(f'{file_path}/{prefix}.nc')
        files_to_cp[f'{file_path}/{prefix}.nc'] = f"{prefix}.nc"
    #
    return ens_files, files_to_cp


def main():
    """
    MAIN
    The command line options are listed in function parse_args.
    """
    args = parse_args()
    out_dir = args.out_dir
    tas_files, tas_summaries = get_file_list(args.tas_stream, "tas")
    sst_files, sst_summaries = get_file_list(args.sst_stream, "sst")
    lsat_files, lsat_summaries = get_file_list(args.lsat_stream, "lsat")
    zipper = zip(tas_files, sst_files, lsat_files)
    for tas_file, sst_file, lsat_file in zipper:
        tas_ds = xr.open_dataset(tas_file)
        tas_ds['tas'].attrs['long_name'] = 'Surface Temperature Anomaly'
        tas_ds['tas'].attrs['standard_name'] = 'surface_temperature'
        tas_ds = tas_ds.rename({'tas': 'ts'})
        sst_ds = xr.open_dataset(sst_file)
        lsat_ds = xr.open_dataset(lsat_file)
        lsat_ds['lsat'].attrs['standard_name'] = 'surface_temperature'
        out_ds = tas_ds.merge(sst_ds).merge(lsat_ds)
        out_file = f"{out_dir}/{os.path.basename(tas_file)}"
        print(f"{tas_file}, ...: {out_file}")
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in out_ds.data_vars}
        out_ds.to_netcdf(out_file, encoding=encoding)
    #
    for summary in [tas_summaries, sst_summaries, lsat_summaries]:
        for f2c in summary:
            print(f"{f2c}: {out_dir}/{summary[f2c]}")
            out_file = f"{out_dir}/{summary[f2c]}"
            shutil.copyfile(f2c, out_file)
    #
    # Other adhoc attribute changes
    #
    # For tas/ts:
    ds = xr.open_dataset(f"{out_dir}/DCENT_I_1.1.0.0_mean_spread_tas.nc")
    for mean_spread_varname in ['tas_mean', 'tas_std', 'tas_p05', 'tas_p95']:
        ds[mean_spread_varname].attrs['long_name'] = 'Surface Temperature Anomaly'
        ds[mean_spread_varname].attrs['standard_name'] = 'surface_temperature'
        ds = ds.rename({mean_spread_varname: mean_spread_varname.replace('tas', 'ts')})
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{out_dir}/temp_out.nc", encoding=encoding)
    del ds
    shutil.move(
        f"{out_dir}/temp_out.nc",
        f"{out_dir}/DCENT_I_1.1.0.0_mean_spread_tas.nc",
    )
    #
    # For lsat:
    ds = xr.open_dataset(f"{out_dir}/DCENT_I_1.1.0.0_mean_spread_lsat.nc")
    for mean_spread_varname in ['lsat_mean', 'lsat_std', 'lsat_p05', 'lsat_p95']:
        ds[mean_spread_varname].attrs['standard_name'] = 'surface_temperature'
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{out_dir}/temp_out.nc", encoding=encoding)
    del ds
    shutil.move(
        f"{out_dir}/temp_out.nc",
        f"{out_dir}/DCENT_I_1.1.0.0_mean_spread_lsat.nc",
    )
    #
    print('Finish')


if __name__ == "__main__":
    main()
