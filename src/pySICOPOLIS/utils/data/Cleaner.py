import netCDF4 as nc
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

class DataCleaner3D:

	def __init__(self, 
		     ds : xarrayDataset) -> None:
		self.ds = ds

	def	
