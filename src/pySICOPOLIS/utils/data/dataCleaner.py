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

def corruptionToNum(field : DataArray,
                    field_shape : VectorNumpy,
                    replace_with : float = np.nan) -> TensorNumpy:
      
    """
    Replace corrupted parts of a DataArray with a float
    So far only used for ds_age["age_uncert"] field
    Returns uncorrupted numpy tensor
    Parameters
	----------
	field : DataArray
        Corrupted DataArray
    field_shape: 1D numpy array
        Shape of field (which is sometimes not readable using field.shape for a corrupt field)
    replace_with : float
        Replace corrupted stuff with this float, default np.nan
    """

    field_clean = np.zeros(field.shape)

    for layer in range(field_shape[0]):

        try:
            field_clean[layer] = field[layer].data
        except:
            print(f"Uncorrupt z={layer}.")
            for j in range(field_shape[1]):
                try:
                    field_clean[layer, j] = field[layer, j].data
                except:
                    for i in range(field_shape[2]):
                        try:
                            field_clean[layer, j, i] = field[layer, j, i].data
                        except:
                            field_clean[layer, j, i] = np.nan
                        
    return field_clean

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
