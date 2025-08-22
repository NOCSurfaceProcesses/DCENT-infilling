# DCENT Gridding

Scripts for the production of an in-filled DCENT dataset

## Configuration Files

Configuration files passed to the scripts are in YAML format. The file-paths are generic and the
user will need to replace them with correct paths.

## Input Data

### Ellipse-based Covariance

#### SST

- ESA-CCI SST re-gridded to 5 x 5 degree, monthly resolution
- Filtered to ocean grid-points using the original data mask (add details)

#### Air Temperature

- ERA5 re-gridded to 5 x 5 degree, monthly resolution
- Add: Details about mask

### Infilling

- DCENT version 2.0 SST and LSAT ensemble members can be obtained from: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NU4UGW).
- DCENT Error Covariances available from .... (add details)

### Blending

The weightings used to blend the SST and LSAT fields (add details)
