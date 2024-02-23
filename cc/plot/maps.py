from .. import util
from ..tools import field
import cartopy.crs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.ticker
import numpy
import xarray


def maplot(data:xarray.DataArray,
           ax=None,
           plotmode='contourf',
           transform=cartopy.crs.PlateCarree(),
           zmin=-1., zmax=1., nbins=21,
           cmap='seismic',
           colors='k',
           significance = None,
           mask = None,
           shape_show = False, shape = None,
           cyclic = False,
           hide_ax_labels=False,
           zorder=None,
           **kwargs):
    '''
    
    Parameters
    ----------
    ...
    significance : None or xarray.DataArray or tuple
        If None, no significance is shown.
        If xarray.DataArray, significance field, significance level and hatches parameter as set by default.
        If tuple:
            - the first item is the significance field,
            - the second item is a float that indicates the significance level to plot,
            - the third and optional item is the hatches parameter, that is set by default if missing.
    '''

    util.check_DataArray(data)
    _data = field.check_lat_lon_names(data)
    ### check that data only has lat and lon as dimensions!

    #_lon = _data.lon; _lat = _data.lat
    #if len(_lon)

    _ax = ax or matplotlib.pyplot.gca()
    _cmap = matplotlib.pyplot.get_cmap(cmap)

    if significance is not None:
        if isinstance(significance, xarray.DataArray):
            _signi = field.check_lat_lon_names(significance)
            field.check_lat_lon_coords(_data, _signi, warning = True) ### maybe not needed
            _signi_lvl = .05
            _signi_htc = '...'
            _signi_zrd = None
        elif isinstance(significance, tuple):
            _signi = field.check_lat_lon_names(significance[0])
            field.check_lat_lon_coords(_data, _signi, warning = True) ### maybe not needed
            _signi_lvl = significance[1]
            _signi_htc = significance[2] or '....'
            _signi_zrd = significance[3] or None

    if mask is not None:
        _mask = field.check_lat_lon_names(mask)
        field.check_lat_lon_coords(_data, _mask)
        _data = field.mask_land(_data, _mask)

    if cyclic:
        _data = field.add_cyclic_point(_data)

    _shading = None if len(_data.lon.dims) == 2 else 'auto'
    if plotmode == 'contourf':
        _levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        _plot = _ax.contourf(_data.lon,_data.lat,_data,transform=transform,levels=_levels,
                            cmap=_cmap,extend='both',zorder=zorder,**kwargs)
    elif plotmode == 'contour':
        _levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        _plot = _ax.contour(_data.lon,_data.lat,_data,transform=transform,levels=_levels,
                            colors = colors,zorder=zorder,**kwargs)
    elif plotmode == 'raster':
        _levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        _norm = matplotlib.colors.BoundaryNorm(_levels, ncolors=_cmap.N, clip=True)
        _plot = _ax.pcolor(_data.lon,_data.lat,_data,transform=transform,cmap=_cmap,norm=_norm,shading=_shading,zorder=zorder,**kwargs)

    if significance is not None:
        if plotmode == 'raster':
            _ax.pcolor(_data.lon,_data.lat,xarray.where(_signi>_signi_lvl, 1., numpy.nan),transform=transform,alpha=0,hatch=_signi_htc,zorder=_signi_zrd,shading=_shading,**kwargs)
        else:
            _ax.contourf(_data.lon,_data.lat,_signi,transform=transform,levels=[0,_signi_lvl,100],colors='none',hatches=[None,_signi_htc],zorder=_signi_zrd,**kwargs)

    ### add shape plot

    if not hide_ax_labels:
        ### Add parameter for lat and lon spacers (default:5)
        _ax.set_xticks(numpy.linspace(_data.lon.min(), _data.lon.max(), 5), crs=cartopy.crs.PlateCarree())
        _ax.set_yticks(numpy.linspace(_data.lat.min(), _data.lat.max(), 5), crs=cartopy.crs.PlateCarree())
        _ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        _ax.yaxis.set_major_formatter(LatitudeFormatter())

    return _plot


def plot_shape(
    shape=None,
    shape_file=None,
    name=None,
    disp_name='c',
    ax = None,
    color='k',
    linestyle='-',
    linewidth=3,
    fontsize=None,
    zorder=None,
):
    """
    """

    if shape is None and (shape_file is None or name is None):
        raise InputError('A shape or a shaphe_file name and shape name must be given as input(s).')
    _ax = ax or matplotlib.pyplot.gca()
    _shape = shape or field.get_shape(shape_file, name)
    _x_lon = numpy.zeros((len(_shape.points), 1))
    _y_lat = numpy.zeros((len(_shape.points), 1))
    for ip in range(len(_shape.points)):
        _x_lon[ip] = _shape.points[ip][0]
        _y_lat[ip] = _shape.points[ip][1]
    _ax.plot(_x_lon, _y_lat, color=color, linestyle=linestyle, linewidth = linewidth, zorder=zorder)
    if disp_name in ('c', 'center'):
        _points = numpy.array(_shape.points)
        _lat, _lon = _points.mean(axis=0)
        _ax.text(_lat, _lon, name, va = 'center', ha = 'center', fontsize=fontsize)
    elif disp_name in ('tl', 'top-left') or disp_name is True:
        _points = numpy.array(_shape.points)
        _lat = _points[:, 0].min(); _lon = _points[:, 1].min()
        for _pos in _points:
            if _pos[0] == _lat:
                _lon = max(_lon, _pos[1])
        _ax.text(_lat+0.5, _lon-0.5, name, verticalalignment = 'top', fontsize=fontsize)


'''
def old_maplot(lon, lat, z, ax=None, fillmode='contour', zmin=-1., zmax=1., nbins=21, cmap='seismic', signishow = None, signilevel=.05, hatches='...', shapeshow = False, shape = None, maskland = False, maskfile = None, cyclic = False):
    """
    Create a map plot.

    Parameters
    ----------

    lon, lat, z : array-like, shape (nLat, nLon)
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    Returns
    -------

    .

    """

    from cartopy.util import add_cyclic_point
    import line.tools as ltls

    ax = ax or matplotlib.pyplot.gca()
    cmap = matplotlib.pyplot.get_cmap(cmap)
    if cyclic:
        z, lon = add_cyclic_point(z, coord = lon)

    if (maskland):
        z = ltls.mask_land_OLD(z, mask_file = maskfile, cyclic = cyclic)

    if (fillmode == 'contour'):
        #levels = numpy.arange(zmin,zmax+step,step)
        levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        out = ax.contourf(lon,lat,z,levels=levels,cmap=cmap,extend='both')
    elif (fillmode == 'raster'):
        levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
        norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        out = ax.pcolor(lon,lat,z,vmin=zmin,vmax=zmax,cmap=cmap,norm=norm,shading='auto')

    if not (signishow is None):
        ax.contourf(lon,lat,signishow,levels=[0,signilevel,100],colors='none',hatches=[hatches,None])

    if (shapeshow and not (shape is None)):
        x_lon = np.zeros((len(shape.points),1))
        y_lat = np.zeros((len(shape.points),1))
        for ip in range(len(shape.points)):
            x_lon[ip] = shape.points[ip][0]
            y_lat[ip] = shape.points[ip][1]

        plt.plot(x_lon,y_lat,'k',linewidth=3)

    return out
'''