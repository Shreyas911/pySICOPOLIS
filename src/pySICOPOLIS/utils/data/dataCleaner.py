import numpy as np
import xarray as xr

# Required for xarray interpolation feature
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from pySICOPOLIS.backend.types import Dataset, DataArray
from pySICOPOLIS.backend.types import Optional, OptionalList
from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy

__all__ = ['corruptionToNum', 'exp_sigma_level',
           'gaussian_filter_withNaNs']

def corruptionToNum(da_field : DataArray,
                    replace_with : float = np.nan) -> TensorNumpy:
      
    """
    Replace corrupted parts of a DataArray with a float
    So far only used for ds_age["age_uncert"] field
    Returns uncorrupted numpy tensor
    Parameters
	----------
	da_field : DataArray
        Corrupted DataArray
    replace_with : float
        Replace corrupted stuff with this float, default np.nan
    """

    da_field_clean = np.zeros(age_uncert.shape)

    for layer in range(len(age_uncert)):

        try:
            age_uncert_clean[layer] = age_uncert[layer]
        except:
            for j in range(age_uncert.shape[1]):
                try:
                    age_uncert_clean[layer, j] = age_uncert[layer, j]
                except:
                    for i in range(age_uncert.shape[2]):
                        try:
                            age_uncert_clean[layer, j, i] = age_uncert[layer, j, i]
                        except:
                            age_uncert_clean[layer, j, i] = np.nan
                        
    return age_uncert_clean

def exp_sigma_level(zeta: VectorNumpy,
                    exponent : float) -> VectorNumpy:
    """
    Convert uniform z-grid to exponential z-grid
    Returns exponential z-grid
    Parameters
	----------
    zeta : 1D numpy array
        Uniform grid of zeta_c between 0 and 1
    exponent : float
        Hyperparameter for exponential grid
    """
    
    return (np.exp(exponent*zeta) - 1) / (np.exp(exponent) - 1)

def gaussian_filter_withNaNs(field : TensorNumpy,
                             **kwargs) -> TensorNumpy:
    
    """
    Do smoothing with a Gaussian kernel in presence of NaNs
    See here for source - 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    Returns smoothed field
    Parameters
	----------
    field : N-dimensional numpy array
        field to be smoothed
    kwargs
        Pass to scipy.ndimage.gaussian_filter
    """

    maskNaNs = np.isnan(field)

    V = field.copy()
    V[np.isnan(field)] = 0
    VV = gaussian_filter(V, **kwargs)

    W = 0*field.copy()+1
    W[np.isnan(field)] = 0
    WW = gaussian_filter(W, **kwargs)

    smooth_field = VV/WW

    # Return field but only in those regions where original field is not NaN
    return np.where(maskNaNs, np.nan, smooth_field)
