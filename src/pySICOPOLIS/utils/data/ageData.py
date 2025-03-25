import netCDF4 as nc
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from pySICOPOLIS.backend.types import Dataset, DataArray
from pySICOPOLIS.backend.types import Optional, OptionalList
from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy

from pySICOPOLIS.utils.data.dataCleaner import *

__all__ = ['correctAgeDataset', 'interpToModelGrid', 'interpToModelGrid2D', 'find_alpha', 'interpolate_nans']

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
        age_uncert_clean[(safe_age) & (ratio <= 0.04)] = 0.04 * age[(safe_age) & (ratio <= 0.04)]

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
                      hor_interp_method: str,
                      ver_interp_method: str,
                      bool_gausian_smoothing_before: bool = False,
                      sigma_gs: float = 1.25,
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
        Method for horizontal interpolation
    ver_interp_method : str
        Method for vertical interpolation
    bool_gausian_smoothing_before : bool
        Gaussian smoothing before interpolation?
    sigma_gs: float
        Sigma for Gaussian smoothing, default value is such that +- 4 sigma is +- 5kms
    replace_nans_with : float
        Replace NaNs with this float
    path : str or None
        Absolute path to where to export model Dataset as nc file
    filename : str or None
        File name of nc file
    kwargs
        Passed to scipy.ndimage.gaussian_filter
    """

    def gaussian_filter_nan(array):
        """Applies Gaussian filter while ignoring NaNs."""
        nan_mask = np.isnan(array)  # Identify NaNs

        # Replace NaNs with 0 temporarily
        array_filled = np.where(nan_mask, 0, array)

        # Apply Gaussian filter to the data only along x and y axes
        if len(array.shape) == 3:
            filtered = gaussian_filter(array_filled, sigma=[0, sigma_gs, sigma_gs], mode="nearest")
        elif len(array.shape) == 2:
            filtered = gaussian_filter(array_filled, sigma=sigma_gs, mode="nearest")

        # Create a weight mask (1 where valid data, 0 where NaNs)
        weight_mask = np.where(nan_mask, 0, 1).astype(float)  # Converts True/False → 1.0/0.0
        if len(array.shape) == 3:
            weight = gaussian_filter(weight_mask, sigma=[0, 1.25, 1.25], mode="nearest")
        elif  len(array.shape) == 2:
            weight = gaussian_filter(weight_mask, sigma=1.25, mode="nearest")

        # Normalize the filtered data
        filtered /= weight
        filtered[nan_mask] = np.nan  # Restore NaNs

        return filtered

    if bool_gausian_smoothing_before:
        for var in ds_age_correct.data_vars:
            if ds_age_correct[var].dtype.kind in "fi":  # Process only float/int data (skip categorical or boolean)
                ds_age_correct[var].data = gaussian_filter_nan(ds_age_correct[var].data.copy())

    # Interpolate horizontally on to model grid
    ds_model = ds_age_correct.interp(xData=xModel,
                                     yData=yModel,
                                     method = hor_interp_method)

    if hor_interp_method == "linear" and ver_interp_method == "linear":

        ds_model["age_c_uncert_manual"] = ds_model["age_c_uncert"].copy()
        ds_model["age_c_uncert_real_manual"] = ds_model["age_c_uncert_real"].copy()
        ds_model["age_c_manual"] = ds_model["age_c"].copy()

        # Rename x and y dimensions
        ds_model = ds_model.rename({'xData':'xModel',
                                    'yData':'yModel',
                                    'z_minus_zbData': 'z_minus_zbModel'})

        for j in range(len(ds_model["yModel"].data)):
            for i in range(len(ds_model["xModel"].data)):

                j_data, alpha_y = find_alpha(ds_age_correct["yData"].data, ds_model["yModel"].data[j])
                i_data, alpha_x = find_alpha(ds_age_correct["xData"].data, ds_model["xModel"].data[i])

                if alpha_x is not None and alpha_y is not None:
                    ds_model["age_c_uncert_manual"].data[:, j, i] = np.sqrt(alpha_y**2*alpha_x**2*ds_age_correct["age_c_uncert"].data[:, j_data+1, i_data+1]**2\
                                                                  + (1-alpha_y)**2*alpha_x**2*ds_age_correct["age_c_uncert"].data[:, j_data, i_data+1]**2\
                                                                  + alpha_y**2*(1-alpha_x)**2*ds_age_correct["age_c_uncert"].data[:, j_data+1, i_data]**2\
                                                                  + (1-alpha_y)**2*(1-alpha_x)**2*ds_age_correct["age_c_uncert"].data[:, j_data, i_data]**2)
                    ds_model["age_c_uncert_real_manual"].data[:, j, i] = np.sqrt(alpha_y**2*alpha_x**2*ds_age_correct["age_c_uncert_real"].data[:, j_data+1, i_data+1]**2\
                                                                       + (1-alpha_y)**2*alpha_x**2*ds_age_correct["age_c_uncert_real"].data[:, j_data, i_data+1]**2\
                                                                       + alpha_y**2*(1-alpha_x)**2*ds_age_correct["age_c_uncert_real"].data[:, j_data+1, i_data]**2\
                                                                       + (1-alpha_y)**2*(1-alpha_x)**2*ds_age_correct["age_c_uncert_real"].data[:, j_data, i_data]**2)
                    ds_model["age_c_manual"].data[:, j, i] = alpha_y*alpha_x*ds_age_correct["age_c"].data[:, j_data+1, i_data+1]\
                                                           + (1-alpha_y)*alpha_x*ds_age_correct["age_c"].data[:, j_data, i_data+1]\
                                                           + alpha_y*(1-alpha_x)*ds_age_correct["age_c"].data[:, j_data+1, i_data]\
                                                           + (1-alpha_y)*(1-alpha_x)*ds_age_correct["age_c"].data[:, j_data, i_data]

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

    if hor_interp_method == "linear" and ver_interp_method == "linear":

        for j in range(len(ds_model["yModel"].data)):
            for i in range(len(ds_model["xModel"].data)):
                ds_model["age_c_uncert_manual"].data[:, j, i] = interpolate_nans(ds_model["age_c_uncert_manual"].data[:, j, i], bool_uncert = True)
                ds_model["age_c_uncert_real_manual"].data[:, j, i] = interpolate_nans(ds_model["age_c_uncert_real_manual"].data[:, j, i], bool_uncert = True)
                ds_model["age_c_manual"].data[:, j, i] = interpolate_nans(ds_model["age_c_manual"].data[:, j, i], bool_uncert = False)

    ## Interpolate on to zeta_c grid
    ds_model_old = ds_model.copy()
    ds_model = ds_model.interp(zetaData=sigma_levelModel*temp, method = ver_interp_method)

    if hor_interp_method == "linear" and ver_interp_method == "linear":

        for kc in range(len(ds_model["zetaData"].data)):

            kc_old, alpha_z = find_alpha(ds_model_old["zetaData"].data, ds_model["zetaData"].data[kc])

            if kc_old is not None:
                ds_model["age_c_uncert_manual"].data[kc] = np.sqrt(alpha_z**2*ds_model_old["age_c_uncert"].data[kc_old+1]**2\
                                                         + (1-alpha_z)**2*ds_model_old["age_c_uncert"].data[kc_old]**2)
                ds_model["age_c_uncert_real_manual"].data[kc] = np.sqrt(alpha_z**2*ds_model_old["age_c_uncert_real"].data[kc_old+1]**2\
                                                              + (1-alpha_z)**2*ds_model_old["age_c_uncert_real"].data[kc_old]**2)
                ds_model["age_c_manual"].data[kc] = alpha_z*ds_model_old["age_c"].data[kc_old+1]\
                                                  + (1-alpha_z)*ds_model_old["age_c"].data[kc_old]

    ## Rename z-dimension
    ds_model = ds_model.rename({'zetaData':'sigma_levelModel'})
    ## Scale back to between 0 and 1
    ds_model['sigma_levelModel'] = ds_model['sigma_levelModel'] / temp

    # Replace all NaNs with -999.0
    ds_model = ds_model.fillna(replace_nans_with)

    # Write Dataset to NetCDF file
    if path and filename:
        ds_model.to_netcdf(path+filename, mode='w')

    return ds_model

def interpToModelGrid2D(ds: Dataset,
                        xModel: VectorNumpy,
                        yModel: VectorNumpy,
                        hor_interp_method: str,
                        bool_gausian_smoothing_before: bool = False,
                        sigma_gs: float = 12.5,
                        replace_nans_with: float = -999.0,
                        path: Optional[str] = None,
                        filename: Optional[str] = None) -> Dataset:

    """
    For BM5 dataset, interpolation function.
    Parameters
    ----------
    ds : Dataset
        Corrected 2D dataset, for example BedMachine
    xModel : numpy 1D array
        x co-ordinates for model
    yModel : numpy 1D array
        y co-ordinates for model
    hor_interp_method : str
        Method for horizontal interpolation
    bool_gausian_smoothing_before : bool
        Gaussian smoothing before interpolation?
    sigma_gs: float
        Sigma for Gaussian smoothing, default value is such that +- 4 sigma is +- 5kms
    replace_nans_with : float
        Replace NaNs with this float
    path : str or None
        Absolute path to where to export model Dataset as nc file
    filename : str or None
        File name of nc file
    """

    ds = ds.rename({"thickness": "H", "errbed": "H_uncert", "bed": "zl", "surface": "zs"})
    ds["zl_uncert"] = ds["H_uncert"].copy()
    ds["zs_uncert"] = ds["H_uncert"].copy()*0.0 + 10.0

    if bool_gausian_smoothing_before:
        for var in ds.data_vars:
            if ds[var].dtype.kind in "fi":  # Process only float/int data (skip categorical or boolean)
                ds[var].data = gaussian_filter(ds[var].data.copy(), sigma=sigma_gs, mode="nearest")

    # Interpolate horizontally on to model grid
    ds_model = ds.interp(x=xModel, 
                         y=yModel,
                         method = hor_interp_method)

    if hor_interp_method == "linear":

        ds_model["H_uncert_manual"] = ds_model["H_uncert"].copy()
        ds_model["zs_uncert_manual"] = ds_model["zs_uncert"].copy()
        ds_model["zl_uncert_manual"] = ds_model["zl_uncert"].copy()
        ds_model["H_manual"] = ds_model["H"].copy()
        ds_model["zs_manual"] = ds_model["zs"].copy()
        ds_model["zl_manual"] = ds_model["zl"].copy()

        # Rename x and y dimensions
        ds_model = ds_model.rename({'x':'xModel',
                                    'y':'yModel'})

        for j in range(len(ds_model["yModel"].data)):
            for i in range(len(ds_model["xModel"].data)):

                j_data, alpha_y = find_alpha(ds["y"].data, ds_model["yModel"].data[j])
                i_data, alpha_x = find_alpha(ds["x"].data, ds_model["xModel"].data[i])

                if alpha_x is not None and alpha_y is not None:
                    ds_model["H_uncert_manual"].data[j, i] = np.sqrt(alpha_y**2*alpha_x**2*ds["H_uncert"].data[j_data+1, i_data+1]**2\
                                                           + (1-alpha_y)**2*alpha_x**2*ds["H_uncert"].data[j_data, i_data+1]**2\
                                                           + alpha_y**2*(1-alpha_x)**2*ds["H_uncert"].data[j_data+1, i_data]**2\
                                                           + (1-alpha_y)**2*(1-alpha_x)**2*ds["H_uncert"].data[j_data, i_data]**2)
                    ds_model["H_manual"].data[j, i] = alpha_y*alpha_x*ds["H"].data[j_data+1, i_data+1]\
                                                    + (1-alpha_y)*alpha_x*ds["H"].data[j_data, i_data+1]\
                                                    + alpha_y*(1-alpha_x)*ds["H"].data[j_data+1, i_data]\
                                                    + (1-alpha_y)*(1-alpha_x)*ds["H"].data[j_data, i_data]
                    ds_model["zs_manual"].data[j, i] = alpha_y*alpha_x*ds["zs"].data[j_data+1, i_data+1]\
                                                    + (1-alpha_y)*alpha_x*ds["zs"].data[j_data, i_data+1]\
                                                    + alpha_y*(1-alpha_x)*ds["zs"].data[j_data+1, i_data]\
                                                    + (1-alpha_y)*(1-alpha_x)*ds["zs"].data[j_data, i_data]
                    ds_model["zl_manual"].data[j, i] = alpha_y*alpha_x*ds["zl"].data[j_data+1, i_data+1]\
                                                    + (1-alpha_y)*alpha_x*ds["zl"].data[j_data, i_data+1]\
                                                    + alpha_y*(1-alpha_x)*ds["zl"].data[j_data+1, i_data]\
                                                    + (1-alpha_y)*(1-alpha_x)*ds["zl"].data[j_data, i_data]

        ds_model["zl_uncert_manual"] = ds_model["H_uncert_manual"].copy()

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

def find_alpha(arr: VectorNumpy, x: float):
    """
    Given a sorted increasing/decreasing array and a value x,
    find the two values that straddle x and compute alpha.

    Parameters:
        arr (list): A sorted list of numbers.
        x (float): The target value.

    Returns:
        (int, float): Tuple containing (lower_idx, alpha).
    """

    for i in range(len(arr) - 1):
        if arr[i] <= x <= arr[i + 1] or arr[i] >= x >= arr[i + 1]:
            x_low, x_high = arr[i], arr[i + 1]
            alpha = (x - x_low) / (x_high - x_low)
            return i, alpha

    return None, None  # Can reach here near the boundary edges

def interpolate_nans(arr, bool_uncert):
    """
    Linearly interpolates NaN values in a NumPy array without extrapolating.

    Parameters:
        arr (numpy.ndarray): Input array containing NaN values.
        bool_uncert (bool): Is the arr an uncert field? If so, variances are additive, not std. deviations.

    Returns:
        tuple: (Interpolated array, Dictionary of interpolation weights)
    """

    arr = np.asarray(arr, dtype=np.float64)  # Ensure it's a float array
    x = np.arange(len(arr))
    valid = ~np.isnan(arr)
    nan_indices = np.where(np.isnan(arr))[0]

    arr_interp = np.copy(arr)

    for i in nan_indices:
        # Find the nearest valid points on both sides
        left = np.max(x[valid & (x < i)], initial=-1)
        right = np.min(x[valid & (x > i)], initial=len(arr))

        # Ensure both left and right exist for interpolation
        if left != -1 and right != len(arr):
            y1, y2 = arr[left], arr[right]
            w1 = (right - i) / (right - left)
            w2 = (i - left) / (right - left)

            # Apply interpolation
            if not bool_uncert:
                arr_interp[i] = w1 * y1 + w2 * y2
            else:
                arr_interp[i] = np.sqrt(w1**2*y1**2 + w2**2*y2**2)

    return arr_interp
