""" Custom type hinting for XAIRT """

from typing import Optional, Sequence, Union, Tuple, TypeAlias, TypeVar, TypedDict, Dict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import Axes

# Dummy for NotRequired since it is only available in Python 3.11 and onwards
# from typing import NotRequired # use once you have Python 3.11 env
NotRequired: TypeAlias = Optional # Optional is the closest in meaning to NotRequired for now

T = TypeVar("T")
TNumpy = TypeVar("TNumpy", bound=np.generic, covariant=True)

# Alias for numpy arrays, matrices and tensors
VectorNumpy: TypeAlias = np.ndarray[Tuple[float], np.dtype[TNumpy]]
MatrixNumpy: TypeAlias = np.ndarray[Tuple[float, float], np.dtype[TNumpy]]
TensorNumpy: TypeAlias = np.ndarray[Tuple[float, ...], np.dtype[TNumpy]]

# Alias for arguments that can either be a scalar or a list
OptionalList: TypeAlias = Union[T, list[T]]
OptionalSequence: TypeAlias = Union[T, Sequence[T]]

# Alias for plotting stuff
Figure: TypeAlias = mpl.figure.Figure
Axes: TypeAlias = plt.Axes
Colormap: TypeAlias = Union[str, mpl.colors.Colormap]

# Typing hints for xarray datasets and dataarraysxarray.core.dataset.Dataset
Dataset: TypeAlias = xr.core.dataset.Dataset
DataArray: TypeAlias = xr.core.dataarray.DataArray