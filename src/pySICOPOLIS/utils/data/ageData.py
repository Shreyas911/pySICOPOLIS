import netCDF4 as nc
import numpy as np
import xarray as xr

from pySICOPOLIS.backend.types import Dataset, DataArray
from pySICOPOLIS.backend.types import Optional, OptionalList
from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy

from pySICOPOLIS.utils.data.dataCleaner import *

__all__ = ['correctAgeDataset', 'interpToModelGrid']

def correctAgeDataset(ds_age: Dataset,
                      path: Optional[str] = None,
                      filename: Optional[str] = None,
                      zetaLevels: int = 26,
                      unCorrupt: bool = False) -> Dataset:
    
    """
    Create corrected age dataset, with z-coord from bottom to top.
    The coords for the dataArrays are generally x and y instead of indices.
    Parameters
    ----------
    ds_age : Dataset
        Raw age-layer Dataset by Macgregor et. al
    path : str or None
        Absolute path to where to export corrected Dataset as nc file
    filename : str or None
        File name of nc file
    zetaData : int
        Number of vertical levels in data, current version has 25 + 1 for 0 age
    unCorrupt: bool
        Bool to decide if uncorrupt age_uncert field, default False
    """

    ds_age = ds_age.rename({"age_norm": "age_c", "age_norm_uncert": "age_c_uncert", "thick": "H"})

    x = ds_age['x'].transpose("number of grid points in y-direction", 
                              "number of grid points in x-direction").data
    y = ds_age['y'].transpose("number of grid points in y-direction", 
                              "number of grid points in x-direction").data
    H = ds_age['H'].transpose("number of grid points in y-direction", 
                                      "number of grid points in x-direction").data

    jData = np.arange(x.shape[0])
    iData = np.arange(x.shape[1])
    zetaData = np.arange(zetaLevels)

    # delta_z for each column, thickness dependent
    delta_z = H / (zetaLevels-1)
    # z co-ord within each column, thickness dependent
    z_minus_zb = np.array([delta_z*i for i in range(zetaLevels)])

    # DataArray for x coordinates
    da_x = xr.DataArray(
        data = x,
        dims = ["jData", "iData"],
        coords = dict(
            jData = jData,
            iData = iData
        ),
        attrs = dict(description="x in kms"),
    )

    # DataArray for y coordinates
    da_y = xr.DataArray(
        data = y,
        dims = ["jData", "iData"],
        coords = dict(
            jData = np.arange(y.shape[0]),
            iData = np.arange(y.shape[1])
        ),
        attrs = dict(description="y in kms"),
    )

    # DataArray for thickness
    da_H = xr.DataArray(
        data = H,
        coords = dict(
            yData = da_y.data[:,0],
            xData = da_x.data[0]
        ),
        dims = ["yData", "xData"],
        attrs = dict(description="Ice thickness in metres"),
    )

    # DataArray for z co-ord within each column, thickness dependent
    da_z_minus_zb = xr.DataArray(
        data = z_minus_zb,
        coords = dict(
            zetaData = zetaData,
            yData = da_y.data[:,0],
            xData = da_x.data[0]
        ),
        dims = ["zetaData", "yData", "xData"],
        attrs = dict(description="z-zb in metres"),
    )

    # Load age DataArray and reverse, z co-ord now bottom to top
    age = ds_age['age_c'].transpose("number of vertical layers",
                                    "number of grid points in y-direction", 
                                    "number of grid points in x-direction")[::-1]
    old_age_shape = age.data.shape
    # Concatenate 0 age layer at the top
    age = np.concatenate((age,np.zeros((1,jData.shape[0],iData.shape[0]), dtype = np.float64)), axis = 0)

    # DataArray for age
    da_age = xr.DataArray(
        data = age,
        coords=dict(
            z_minus_zbData = (["zetaData", "yData", "xData"], da_z_minus_zb.data),
            yData = da_y.data[:,0],
            xData = da_x.data[0]
        ),
        dims = ["zetaData", "yData", "xData"],
        attrs=dict(description="Age in years, from bottom to top"),
    )

    age_uncert_fake = 0.1*age
    age_uncert_fake[age_uncert_fake <= 1.0] = 1.0

    # DataArray for fake age uncertainty since the real one is corrupted
    da_age_uncert_fake = xr.DataArray(
        data = age_uncert_fake,
        coords=dict(
            z_minus_zbData = (["zetaData", "yData", "xData"], da_z_minus_zb.data),
            yData = da_y.data[:,0],
            xData = da_x.data[0]
        ),
        dims = ["zetaData", "yData", "xData"],
        attrs=dict(description="Fake age uncertainty in years, from bottom to top"),
    )

    # If uncorrupt, add age_uncert data as well
    if unCorrupt:

        # Load age uncertainty DataArray and reverse, z co-ord now bottom to top
        age_uncert = ds_age['age_c_uncert'].transpose("number of vertical layers",
                                                    "number of grid points in y-direction", 
                                                    "number of grid points in x-direction")[::-1]

        age_uncert_clean = corruptionToNum(age_uncert, old_age_shape, replace_with = np.nan)
        age_uncert_clean[age_uncert_clean == 0.0] = np.nan

        # Concatenate 0 age_uncert layer at the top, but can't divide by 0
        # So assign uncert 1.0 years which is quite small
        age_uncert_clean = np.concatenate((age_uncert_clean,
                                           np.ones((1,jData.shape[0],iData.shape[0]), 
                                                   dtype = np.float64)), 
                                           axis = 0)
        safe_age = (age > 0) & (age_uncert_clean > 0)
        ratio = np.zeros_like(age, dtype=float)
        ratio[safe_age] = age_uncert_clean[safe_age] / age[safe_age]
        age_uncert_clean[(safe_age) & (ratio <= 0.1)] = 0.1 * age[(safe_age) & (ratio <= 0.1)]

        # DataArray for age uncertainty
        da_age_uncert = xr.DataArray(
            data = age_uncert_clean,
            coords=dict(
                z_minus_zbData = (["zetaData", "yData", "xData"], da_z_minus_zb.data),
                yData = da_y.data[:,0],
                xData = da_x.data[0]
            ),
            dims = ["zetaData", "yData", "xData"],
            attrs=dict(description="Age uncertainty in years, from bottom to top"),
        )

    
    # Collect all DataArrays in a Dataset
    ds_age_correct = xr.Dataset()
    ds_age_correct = ds_age_correct.assign(xMesh          = da_x,
                                           yMesh          = da_y,
                                           H              = da_H,
                                           z_minus_zbData = da_z_minus_zb,
                                           age_c            = da_age,
                                           age_c_uncert     = da_age_uncert_fake)

    # If uncorrupt, add age_uncert data as well
    if unCorrupt:
        ds_age_correct = ds_age_correct.assign(age_c_uncert_real = da_age_uncert)

    # Write Dataset to NetCDF file
    if path and filename:
        ds_age_correct.to_netcdf(path+filename, mode='w')

    return ds_age_correct

def interpToModelGrid(ds_age_correct: Dataset,
                      xModel: VectorNumpy,
                      yModel: VectorNumpy,
                      sigma_levelModel: VectorNumpy,
                      hor_interp_method: str = 'nearest',
                      ver_interp_method: str = 'linear',
                      replace_nans_with: float = -999.0,
                      path: Optional[str] = None,
                      filename: Optional[str] = None,
                      **kwargs) -> Dataset:

    """
    Create corrected age dataset, with z-coord from bottom to top.
    The coords for the dataArrays are generally x and y instead of indices.
    Parameters
    ----------
    ds_age_correct : Dataset
        Corrected age layer dataset
    xModel : numpy 1D array
        x co-ordinates for model
    yModel : numpy 1D array
        y co-ordinates for model
    sigma_levelModel : numpy 1D array
        Normalized z-co-ordinates for model
    hor_interp_method : str
        Method for horizontal interpolation, default 'nearest'
    ver_interp_method : str
        Method for vertical interpolation, default 'nearest'
    replace_nans_with : float
        Replace NaNs with this float
    path : str or None
        Absolute path to where to export model Dataset as nc file
    filename : str or None
        File name of nc file
    kwargs
        Passed to scipy.ndimage.gaussian_filter
    """

    # Interpolate horizontally on to model grid
    ds_model = ds_age_correct.interp(xData=xModel, 
                                     yData=yModel,
                                     method = hor_interp_method)
    # Rename x and y dimensions
    ds_model = ds_model.rename({'xData':'xModel', 
                                'yData':'yModel', 
                                'z_minus_zbData': 'z_minus_zbModel'})
    
    # DataArray for x-coordinate
    da_x = xr.DataArray(
        data = np.tile(xModel, (yModel.shape[0],1)),
        dims = ["jModel", "iModel"],
        coords = dict(
            jModel = np.arange(yModel.shape[0]),
            iModel = np.arange(xModel.shape[0])
        ),
        attrs = dict(description="x in kms"),
    )

    # DataArray for y-coordinate
    da_y = xr.DataArray(
        data = np.tile(yModel, (xModel.shape[0],1)).T,
        dims = ["jModel", "iModel"],
        coords = dict(
            jModel = np.arange(yModel.shape[0]),
            iModel = np.arange(xModel.shape[0])
        ),
        attrs = dict(description="y in kms"),
    )

    ds_model = ds_model.assign(xMesh = da_x, yMesh = da_y)
    ds_model = ds_model.drop_dims(['jData','iData'])

    # Get scaling of zetaData
    temp = ds_age_correct['zetaData'].shape[0]-1

    # Interpolate to zeta_c grid

    ## First interpolate the in-between NaNs
    ds_model = ds_model.interpolate_na(dim="zetaData", method = ver_interp_method)
    ## Interpolate on to zeta_c grid
    ds_model = ds_model.interp(zetaData=sigma_levelModel*temp, method = ver_interp_method)
    ## Rename z-dimension
    ds_model = ds_model.rename({'zetaData':'sigma_levelModel'})
    ## Scale back to between 0 and 1
    ds_model['sigma_levelModel'] = ds_model['sigma_levelModel'] / temp

    # Smooth age data in 2D fashion
    age_smooth2D = np.zeros(ds_model['age_c'].data.shape)

    for k in range(age_smooth2D.shape[0]):
        age_smooth2D[k] = gaussian_filter_withNaNs(ds_model['age_c'].data[k],
                                                   **kwargs)
    
    # DataArray for smoothed (2D fashion) age layer data
    da_age_smooth2D = xr.DataArray(
        data = age_smooth2D,
        dims = ["sigma_levelModel", "yModel", "xModel"],
        coords = dict(
            sigma_levelModel = ds_model["sigma_levelModel"].data,
            yModel      = ds_model["yModel"].data,
            xModel      = ds_model["xModel"].data
        ),
        attrs = dict(description="Age smoothed 2D-wise",
                     metadata=str(kwargs)),
    )

    # Smooth age data, whole 3D field at once, tends to be less realistic
    age_smooth3D = gaussian_filter_withNaNs(ds_model['age_c'].data,
                                            **kwargs)

    # DataArray for smoothed (3D fashion) age layer data
    da_age_smooth3D = xr.DataArray(
        data = age_smooth3D,
        dims = ["sigma_levelModel", "yModel", "xModel"],
        coords = dict(
            sigma_levelModel = ds_model["sigma_levelModel"].data,
            yModel      = ds_model["yModel"].data,
            xModel      = ds_model["xModel"].data
        ),
        attrs = dict(description="Age smoothed 3D field at once",
                     metadata=str(kwargs)),
    )

    age_uncert_smooth2D_fake = 0.1*age_smooth2D
    age_uncert_smooth2D_fake[age_uncert_smooth2D_fake <= 1.0] = 1.0
    age_uncert_smooth3D_fake = 0.1*age_smooth3D
    age_uncert_smooth3D_fake[age_uncert_smooth3D_fake <= 1.0] = 1.0

    # DataArray for smoothed (2D fashion) age uncertainty data
    da_age_uncert_smooth2D = xr.DataArray(
        data = age_uncert_smooth2D_fake,
        dims = ["sigma_levelModel", "yModel", "xModel"],
        coords = dict(
            sigma_levelModel = ds_model["sigma_levelModel"].data,
            yModel      = ds_model["yModel"].data,
            xModel      = ds_model["xModel"].data
        ),
        attrs = dict(description="Fake age uncertainty smoothed 2D-wise",
                     metadata=str(kwargs)),
    )

    # DataArray for smoothed (2D fashion) age uncertainty data
    da_age_uncert_smooth3D = xr.DataArray(
        data = age_uncert_smooth3D_fake,
        dims = ["sigma_levelModel", "yModel", "xModel"],
        coords = dict(
            sigma_levelModel = ds_model["sigma_levelModel"].data,
            yModel      = ds_model["yModel"].data,
            xModel      = ds_model["xModel"].data
        ),
        attrs = dict(description="Fake age uncertainty smoothed 3D-wise",
                     metadata=str(kwargs)),
    )

    # Assign smoothed age fields to ds_model
    ds_model = ds_model.assign(age_c_smooth2D = da_age_smooth2D, 
                               age_c_smooth3D = da_age_smooth3D,
                               age_c_uncert_smooth2D = da_age_uncert_smooth2D,
                               age_c_uncert_smooth3D = da_age_uncert_smooth3D,)

    # Replace all NaNs with -999.0
    ds_model = ds_model.fillna(replace_nans_with)

    # Write Dataset to NetCDF file
    if path and filename:
        ds_model.to_netcdf(path+filename, mode='w')

    return ds_model

def interpToModelGrid2D(ds: Dataset,
                        xModel: VectorNumpy,
                        yModel: VectorNumpy,
                        hor_interp_method: str = 'nearest',
                        replace_nans_with: float = -999.0,
                        path: Optional[str] = None,
                        filename: Optional[str] = None) -> Dataset:

    """
    Create corrected age dataset, with z-coord from bottom to top.
    The coords for the dataArrays are generally x and y instead of indices.
    Parameters
    ----------
    ds : Dataset
        Corrected 2D dataset, for example BedMachine
    xModel : numpy 1D array
        x co-ordinates for model
    yModel : numpy 1D array
        y co-ordinates for model
    sigma_levelModel : numpy 1D array
        Normalized z-co-ordinates for model
    hor_interp_method : str
        Method for horizontal interpolation, default 'nearest'
    replace_nans_with : float
        Replace NaNs with this float
    path : str or None
        Absolute path to where to export model Dataset as nc file
    filename : str or None
        File name of nc file
    """

    ds = ds.rename({"thickness": "H", "errbed": "H_uncert"})

    # Interpolate horizontally on to model grid
    ds_model = ds.interp(x=xModel, 
                         y=yModel,
                         method = hor_interp_method)
    # Rename x and y dimensions
    ds_model = ds_model.rename({'x':'xModel', 
                                'y':'yModel'})
    
    # DataArray for x-coordinate
    da_x = xr.DataArray(
        data = np.tile(xModel, (yModel.shape[0],1)),
        dims = ["jModel", "iModel"],
        coords = dict(
            jModel = np.arange(yModel.shape[0]),
            iModel = np.arange(xModel.shape[0])
        ),
        attrs = dict(description="x in kms"),
    )

    # DataArray for y-coordinate
    da_y = xr.DataArray(
        data = np.tile(yModel, (xModel.shape[0],1)).T,
        dims = ["jModel", "iModel"],
        coords = dict(
            jModel = np.arange(yModel.shape[0]),
            iModel = np.arange(xModel.shape[0])
        ),
        attrs = dict(description="y in kms"),
    )

    ds_model = ds_model.assign(xMesh = da_x, yMesh = da_y)

    # Replace all NaNs with -999.0
    ds_model = ds_model.fillna(replace_nans_with)

    # Write Dataset to NetCDF file
    if path and filename:
        ds_model.to_netcdf(path+filename, mode='w')

    return ds_model

