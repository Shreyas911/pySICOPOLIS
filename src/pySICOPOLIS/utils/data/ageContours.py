import numpy as np
import xarray as xr

from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy

__all__ = ['get_column_iso_index', 'get_age_isochrone']

def get_age_column_iso_index(age_c_column: VectorNumpy,
                             age_iso_value: float) -> int:

    """
    Return the smallest index in age column with age lesser than age_iso_value.
    ----------
    age_c_column : VectorNumpy
        One column of some age field, either model or data
    age_iso_value : float
        The age value that is either the threshold or we need the isochrone for
    """

    # This line is necessary if the age value comes from data where NaNs are -999.0
    age_c_column[age_c_column == -999.0] = 1000000 
    
    if age_c_column[-1] > age_iso_value:
        return -100
    elif age_c_column[0] < age_iso_value:
        return -100
    else:
        return np.argmax(age_c_column < age_iso_value)

def get_age_isochrone(sigma_level_c: VectorNumpy, 
                      H: MatrixNumpy, 
                      age_c: TensorNumpy, 
                      age_iso_value: float) -> MatrixNumpy:

    """
    Return the 2D age isochrone field for a given age_iso_value and age field.
    ----------
    sigma_level_c : VectorNumpy
        Vertical levels in the data, normalized between 0 and 1
    H : MatrixNumpy
        Ice sheet thickness field
    age_c : TensorNumpy
        Age field
    age_iso_value : float
        The age value that we need the isochrone for
    """

    if age_iso_value == 0.0:
        return H

    iso_indices = np.zeros(age_c.shape[1:], dtype = int)
    sigma_c_level_isochrone = np.zeros(age_c.shape[1:])
    H_isochrone = np.zeros(age_c.shape[1:])

    for j in range(age_c.shape[1]):
        for i in range(age_c.shape[2]):
        
            iso_indices[j, i] = get_age_column_iso_index(age_c[:, j, i], age_iso_value)

            if iso_indices[j, i] == -100:
                sigma_c_level_isochrone[j, i] = np.nan
            else:
                # Linear interpolation using age field to get correct interpolated sigma level
                sigma1 = sigma_level_c[iso_indices[j, i] - 1]
                sigma2 = sigma_level_c[iso_indices[j, i]]
                age1 = age_c[iso_indices[j, i] - 1, j, i]
                age2 = age_c[iso_indices[j, i], j, i]
                alpha = (age2 - age_iso_value) / (age2 - age1)
                
                sigma_c_level_isochrone[j, i] = sigma1*alpha + sigma2*(1-alpha)

    return H*sigma_c_level_isochrone
