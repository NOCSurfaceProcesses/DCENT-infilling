# DCENT Gridding

Scripts for the production of an in-filled DCENT dataset.

## Configuration Files

Configuration files passed to the scripts are in YAML format. The file-paths are generic and the
user will need to replace them with correct paths.

## Input Data

### Ellipse-based Covariance

#### SST

The ESA-CCI SST anomalies 1982-2022 (baseline 1982-2010) are re-gridded to a 5 x 5 degree, monthly,
resolution grid.

A mask is generated from the 5 x 5 ESA-CCI SST internal land-mask and a sea-ice mask computed from
grid-boxes with 15% sea-ice-fraction (ESA-CCI Sea-ice).

Only unmasked data-complete (has values for all time-points) grid-boxes are used in the estimation
of ellipse parameters and interpolation covariance matrix, using the `ellipse` module from
[GloMarGridding](https://pypi.org/project/glomar_gridding/) for SST.

#### Air Temperature

The ERA5 Reanalysis HRES 2 metre air temperature anomalies 1979-2023 (baseline 1981-2010) are
re-gridded to to a 5 x 5 degree, monthly, resolution grid.

A 5 x 5 degree land-fraction mask is computed from the ESA-CCI SST original resolution land-mask.
Grid-boxes are then included if

- If the land-fraction value is greater than 0.01,
- Or the grid-box is masked in the SST ellipse-parameter data input (to account for the
  sea-ice-fraction component).

Only unmasked data-complete (has values for all time-points) grid-boxes are used in the estimation
of ellipse parameters and interpolation covariance matrix, using the `ellipse` module from
[GloMarGridding](https://pypi.org/project/glomar_gridding/) for LSAT.

---

### Infilling

- DCENT version 2.0 SST and LSAT ensemble members can be obtained from: [Harvard
  Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NU4UGW).
- DCENT Error Covariances available on request.

---

### Blending

The weightings used to blend the SST and LSAT fields is the combination of the land-fraction and
sea-ice-fraction arrays (`land-fraction` + (1 - `land-fraction`) * `sea-ice-fraction`).
