# DCENT-I Recipe

## Set-up

Get this repository:

```bash
git clone https://github.com/NOCSurfaceProcesses/DCENT-infilling /path/to/DCENT-infilling
```

Create a python environment, and install required dependencies

```bash
cd /path/to/DCENT-infilling
python -m venv venv  # Create virtual environment
source ./venv/bin/activate  # load the virtual environment
pip install --editable .  # Install the project and its dependencies into the virtual environment
pip install matplotlib  # Option for plotting
```

## Masks

Adjustments to the file-locations/inputs etc can be set in the file
`./land-ice-mask/mask_config.yaml`

### Land Mask

The land-mask is generated from an instance of a high resolution (1/20th degree) analysed SST field
(which contains a `mask` field). The `mask` is a _bit-mask_ where the 2nd bit indicates that the
pixel is land.

This file is combined with the ArcLake data (at the same resolution) and can be accessed from
[https://www.laketemp.net/home_ARCLake/index.php](https://www.laketemp.net/home_ARCLake/index.php),
where version 1.1.2 is used. Along with a mapping table which maps the `lakeid` field to a lake
name.

The lake data is filtered to include only a selection of lakes.

To generate the land-mask at the desired resolution run

```bash
cd /path/to/DCENT-infilling
source ./venv/bin/activate
./land-ice-mask/create_land_mask.py --config ./land-ice-mask/mask_config.yaml
```

### Sea-Ice Mask

Convolves the higher resolution HadISST2 sea-ice-concentration to a target resolution (aligning with
the land-mask). A threshold value can be set in the config to tune the sea-ice concentration limits.
Levels above this value are assumed to be 100% ice in this pixel, otherwise 0% ice.

Further, the sea-ice-concentration file can be _extended_ to the end of a target year (if required).

These two jobs can be done in either order - although the land mask step above must be done before
as this is used to determine the output grid.

```bash
cd /path/to/DCENT-infilling
source ./venv/bin/activate
./land-ice-mask/sea_ice_to_5x5.py --config ./land-ice-mask/mask_config.yaml
./land-ice-mask/extend_sea_ice.py --config ./land-ice-mask/mask_config.yaml
```

## Error Covariance Matrices

This converts the _table_ error covariance format into a matrix (grid size * grid_size) for
convenience in the infilling process.

The infilling script can be amended to do this during the process. See
`dcent_infilling.error_covariance.process_error_table` which takes an input path and resolution. It
returns the resulting matrix.

To perform this as a pre-processing step:

```bash
cd /path/to/DCENT-infilling
source ./venv/bin/activate
./pre-processing/process_error_covariance_files.py --config ./pre-processing/config_error_cov.yaml
```

Update the config file as required.

NOTE: this is not the most efficient way to do this procedure, and could benefit from another look.

## Interpolation Covariance Matrices

### SST

### LSAT

## Infilling

### SST

### LSAT

## Blending

## Release

## Update
