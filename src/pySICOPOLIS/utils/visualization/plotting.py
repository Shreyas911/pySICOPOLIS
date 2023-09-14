import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pySICOPOLIS.backend.types import Figure, Colormap, Axes
from pySICOPOLIS.backend.types import VectorNumpy, MatrixNumpy
from pySICOPOLIS.backend.types import Optional, OptionalList

__all__ = ["plot_normal", "plot_log"]

def get_cmap_rgb(cmap: Colormap,
		 n_colors: int = 256) -> Colormap:

	"""
	Enter a matplotlib colormap name, return rgb array
	Parameters
	----------
	cmap : str or colormap object or None
	    if matplotlib color, should be string to make sure
	    n_colors is selected correctly. If cmocean, pass the object
	    e.g.
	        matplotlib :  get_cmap_rgb('viridis',10)
	        cmocean : get_cmap_rgb(cmocean.cm.thermal,10)
	n_colors : int, optional
	    number of color levels in color map
	"""

	return cm.get_cmap(cmap,n_colors)(range(n_colors))

def plot_normal(x: VectorNumpy,
		y: VectorNumpy,
		data: MatrixNumpy,
		cmap: Colormap = 'RdBu_r',
		nbins: int = 100,
		fig: Optional[Figure] = None,
		ax: Optional[Axes] = None,
		cbar_label: Optional[str] = None,
		**kwargs) -> Optional[Axes]:

	"""
	x : 1D array of x-axis co-ordinates
	y : 1D array of y-axis co-ordinates
	data : The field that gets plotted, a 2D array
	fig : matplotlib.figure, optional
	ax : matplotlib.axes, optional
	    to make plot at
	cmap : str, optional
	    specifies colormap
	cbar_label : str, optional
	    label for colorbar, default grabs units from DataArray
	kwargs
	    passed to matpotlib.pyplot.contourf
	Returns
	-------
	ax : matplotlib.axes
	    if one is not provided
	""" 

	return_ax = False
	if ax is None:
		fig, ax = plt.subplots()
		return_ax = True

	im = ax.contourf(x, y, data, nbins, cmap = plt.set_cmap(cmap), **kwargs)

	### Locate ax and apend axis cax to it
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)

	### Mount colorbar on cax
	colorbar = fig.colorbar(im, cax=cax)

	if cbar_label is not None:
		colorbar.set_label(str(cbar_label))
	else:
		colorbar.set_label("Color scheme")

	if return_ax:
		return ax

def plot_log(x: VectorNumpy,
	     y: VectorNumpy,
	     data: MatrixNumpy,
	     cmap: Colormap = 'RdBu_r',
	     nbins: int = 100,
	     bin_edges: Optional[OptionalList[float]] = None,
	     fig: Optional[Figure] = None,
	     ax: Optional[Axes] = None,
	     cbar_label: Optional[str] = None,
	     **kwargs) -> Optional[Axes]:

	"""
	x : 1D array of x-axis co-ordinates
	y : 1D array of y-axis co-ordinates
	data : The field that gets plotted, a 2D array
	nbins : int, optional
	    number of colored bin (centers) positive and negative values
	    i.e. we get 2*nbins+1, bins. one is neutral (middle)
	bin_edges : array-like, optional
	    exclusive with nbins, specify bin edges (positive only)
	fig : matplotlib.figure, optional
	ax : matplotlib.axes, optional
	    to make plot at
	cmap : str, optional
	    specifies colormap
	cbar_label : str, optional
	    label for colorbar, default grabs units from DataArray
	kwargs
	    passed to matpotlib.pyplot.contourf
	Returns
	-------
	ax : matplotlib.axes
	    if one is not provided
	"""

	return_ax = False
	if ax is None:
		fig, ax = plt.subplots()
		return_ax = True	
	if nbins is not None and bin_edges is not None:
		raise TypeError('one or the other')

	log = np.log10(np.abs(data))
	log = np.where((~np.isnan(log)) & (~np.isinf(log)), log, 0.)

	if nbins is not None:
		_,bin_edges = np.histogram(log,bins=nbins)
	else:
		nbins = len(bin_edges)-1

	logbins=np.round(bin_edges)

	# determine if colorbar will be extended
	maxExtend = np.any((data>10**logbins[-1]))
	minExtend = np.any((data<-10**logbins[-1]))
	extend='neither'
	if minExtend and maxExtend:
	    extend='both'
	elif maxExtend:
	    extend='max'
	elif minExtend:
	    extend='min'
	# determine number of colors, adding one for each extension
	# and always one extra, the middle color bin
	ncolors=2*nbins+1
	ncolors = ncolors+1 if maxExtend else ncolors
	ncolors = ncolors+1 if minExtend else ncolors
	# if only one end is extended,
	# chop off the extreme value from the other end to fit
	# in the middle (neutral) colorbin
	if extend in ['min' ,'max']:
		cmap = get_cmap_rgb(cmap,ncolors+1)
		bot = np.arange(1,nbins+1) if extend=='max' else np.arange(0,nbins+1)
		top = np.arange(ncolors-nbins,ncolors) if extend=='min' else np.arange(ncolors-nbins,ncolors+1)
		index = list(bot)+[nbins+1]+list(top)
		cmap = cmap[index,:]
	else:
		cmap=get_cmap_rgb(cmap,ncolors)

	# levels and plot
	levels=10**logbins
	levels = np.concatenate([-levels[::-1],levels],axis=0)
	im=ax.contourf(x, y, data, levels=levels, colors=cmap, extend=extend, **kwargs)

	### Locate ax and apend axis cax to it
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)

	### Mount colorbar on cax
	colorbar = fig.colorbar(im, cax=cax)

	if cbar_label is not None:
		colorbar.set_label(str(cbar_label))
	else:
		colorbar.set_label("Color scheme")

	ticklabels = [f'-10^{b:.0f}' for b in logbins[::-1]]
	ticklabels += [f'10^{b:.0f}' for b in logbins]
	colorbar.set_ticklabels(ticklabels)

	if return_ax:
		return ax

def plot_1D_depth_profile(field: VectorNumpy,
	     depth: VectorNumpy,
		 uncertField: Optional[VectorNumpy] = None,
		 invert_yaxis: Optional[bool] = True,
		 gridLines: Optional[bool] = True,
	     fig: Optional[Figure] = None,
	     ax: Optional[Axes] = None,
	     **kwargs) -> Optional[Axes]:

	"""
	field : 1D array depth-wise (top to bottom) of field
	depth : 1D array of depths
	uncertField : 1D array depth-wise (top to bottom) of 
	            uncertainty in the field, optional
	invert_yaxis : bool to invert_yaxis, optional 
	fig : matplotlib.figure, optional
	ax : matplotlib.axes, optional
	    to make plot at
	kwargs
	    passed to matpotlib.pyplot.plot
	Returns
	-------
	ax : matplotlib.axes
	    if one is not provided
	""" 

	return_ax = False
	if ax is None:
		fig, ax = plt.subplots()
		return_ax = True

	ax.plot(field, depth, **kwargs)
	ax.fill_betweenx(depth, field-uncertField, field+uncertField, alpha=0.8)

	if invert_yaxis:
		ax.invert_yaxis()

	if gridLines:
		ax.grid()
	
	if return_ax:
		return ax