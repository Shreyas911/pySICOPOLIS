import netCDF4 as nc
import numpy as np
import xarray as xr

from pySICOPOLIS.backend.types import Dataset, DataArray
from pySICOPOLIS.backend.types import Optional, OptionalList
from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy, TensorNumpy

from pySICOPOLIS.utils.data.dataCleaner import *

__all__ = ['correctModelDataset']

def correctModelDataset(ds_model: Dataset,
                        path: Optional[str] = None,
                        filename: Optional[str] = None
                       ) -> Dataset:
    
    """
    Create corrected model dataset, with some modification of dims and coords.
    Parameters
    ----------
    ds_model : Dataset
        Raw model dataset output by normal SICOPOLIS run
    path : str or None
        Absolute path to where to export corrected Dataset as nc file
    filename : str or None
        File name of nc file
    """

    ds_model = ds_model.rename({'x':'xModel', 
                                'y':'yModel'})
    
    # z co-ord within each column, thickness dependent
    ds_model['z_minus_zb'] = ds_model['sigma_level_c']*ds_model['H']
    
    # Write Dataset to NetCDF file
    if path and filename:
        ds_model.to_netcdf(path+filename, mode='w')
        
    return ds_model
