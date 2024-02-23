from .. import util
import cartopy.util
import gridfill
import matplotlib.path
import numpy
import pandas
import shapefile
import sklearn.impute
import sklearn.linear_model
import xarray


def check_lat_lon_names(data):
    _dict = {
        'longitude': 'lon',
        'Longitude': 'lon',
        'latitude': 'lat',
        'Latitude': 'lat'
    }
    for _dim in data.dims:
        if _dim in _dict.keys():
            data = data.rename({_dim: _dict[_dim]})
    return data


def check_lat_lon_names_OLD(data):
    '''
    Depreciated.
    '''
    if 'longitude' in data.dims:
        data = data.rename({'longitude': 'lon'})
    if 'latitude' in data.dims:
        data = data.rename({'latitude': 'lat'})
    return data


def check_lat_lon_coords(A, B, warning=False, preci=1e-13):
    util.check_Xarray(A); util.check_Xarray(B);
    A = check_lat_lon_names(A); B : check_lat_lon_names(A)
    if len(A.lat.values) != len(B.lat.values):
        if warning: print('WARNING: Latitudes are not the same.')
        else: raise TypeError('Latitudes must be the same.')
    else:
        if numpy.all(numpy.absolute(A.lat.values) - numpy.absolute(B.lat.values) < preci):
            B = B.assign_coords({'lat' : A.lat})
        else:
            if warning: print('WARNING: Latitudes are not the same.')
            else: raise TypeError('Latitudes must be the same.')
    if len(A.lon.values) != len(B.lon.values):
        if warning: print('WARNING: Longitudes are not the same.')
        else: raise TypeError('Longitudes must be the same.')
    else:
        if numpy.all(numpy.absolute(A.lon.values) - numpy.absolute(B.lon.values) < preci):
            B = B.assign_coords({'lon' : A.lon})
        else:
            if warning: print('WARNING: Longitudes are not the same.')
            else: raise TypeError('Longitudes must be the same.')
    return A, B


def check_lat_lon_coords_SAVE(A, B, warning=False):
    util.check_Xarray(A); util.check_Xarray(B);
    if numpy.any(check_lat_lon_names(A).lat.values != check_lat_lon_names(B).lat.values):
        if warning:
            print('WARNING: Latitudes are not the same.')
        else:
            #print(A.lat)
            #print(B.lat)
            #print(A.lat.values - B.lat.values)
            raise TypeError('Latitudes must be the same.')
    if numpy.any(check_lat_lon_names(A).lon.values != check_lat_lon_names(B).lon.values):
        if warning:
            print('WARNING: Longitudes are not the same.')
        else:
            raise TypeError('Longitudes must be the same.')


#def field_avg(data):
#    weights = numpy.cos(numpy.deg2rad(data.lat))
#    weights.name = 'weights'
#    out = check_lat_lon(data).weighted(weights).mean(('lon', 'lat'), keep_attrs = True)
#    out.attrs = data.attrs
#    return out


def field_avg(data):
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')
    _weights = numpy.cos(numpy.deg2rad(data.lat)); _weights.name = 'weights';
    _avgs = []
    for _da in _das:
        if 'lat' in _da.dims and 'lon' in _da.dims:
            _avgs.append(check_lat_lon_names(_da).weighted(_weights).mean(('lon', 'lat'), keep_attrs = True))
        else:
            _avgs.append(_da)
        _avgs[-1].attrs = _da.attrs
    if _to_ds:
        out = xarray.merge(_avgs)
        #out.attrs = data.attrs
    else:
        out = _avgs[0]
    out.attrs = data.attrs

    return out


def field_sum(data, area=None):
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')
    if isinstance(area, xarray.DataArray):
        _area = area
    elif isinstance(area, str):
        _area = xarray.open_mfdataset('/data/scratch/globc/line/Data/Masks/areacella_fx_'+area+'_*.nc').load().areacella
    else:
        _area = xarray.open_dataset('/data/scratch/globc/line/Data/Masks/areacella_fx_CNRM-CM6-1_gr.nc').areacella
    _sums = []
    for _da in _das:
        if 'lat' in _da.dims and 'lon' in _da.dims:
            _da_sfx = _da * _area
            _da_sfx.attrs = _da.attrs
            _da_sfx.attrs['units'] = _area.attrs['units']
            _sums.append(check_lat_lon_names(_da_sfx).sum(('lon', 'lat'), keep_attrs = True))
        else:
            _sums.append(_da)
        _sums[-1].attrs = _da.attrs
    if _to_ds:
        out = xarray.merge(_sums)
        #out.attrs = data.attrs
    else:
        out = _sums[0]
    out.attrs = data.attrs

    return out


def field_median(data):
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')
    _weights = numpy.cos(numpy.deg2rad(data.lat)); _weights.name = 'weights';
    _avgs = []
    for _da in _das:
        if 'lat' in _da.dims and 'lon' in _da.dims:
            _avgs.append(check_lat_lon_names(_da).weighted(_weights).median(('lon', 'lat'), keep_attrs = True))
        else:
            _avgs.append(_da)
        _avgs[-1].attrs = _da.attrs
    if _to_ds:
        out = xarray.merge(_avgs)
        #out.attrs = data.attrs
    else:
        out = _avgs[0]
    out.attrs = data.attrs

    return out


def lon_flip(data):
    data = check_lat_lon_names(data)
    flipped = util.check_bool_attribute(data, 'lon_flip')
    if not flipped or data.lon[0] == 0.:
        #
        #
        if len(data.lon) == 2:
            _fixed_lons = data.lon.copy()
            for i, start in enumerate(numpy.argmax(numpy.abs(numpy.diff(data.lon)) > 180, axis=1)):
                _fixed_lons[i, start+1:] += 360
            out = data.assign_coords(lon = _fixed_lons)
            out.attrs['lon_flip'] = 'True'
        else:
            out = data.assign_coords(lon = (((data.lon + 180.) % 360.) - 180.)).sortby('lon')
            out.attrs['lon_flip'] = 'True'
    else:
        out = data
        out.attrs['lon_flip'] = 'True'
    if isinstance(out, xarray.Dataset):
        for _var in out.data_vars:
            if 'lon' in out.get(_var).dims:
                out.get(_var).attrs['lon_flip'] = 'True'
    return out


def lon_flip_OLD(data):
    data = check_lat_lon_names(data)
    flipped = util.check_bool_attribute(data, 'lon_flip')
    if not flipped or data.lon[0] == 0.:
        #
        #
        out = data.assign_coords(lon = (((data.lon + 180.) % 360.) - 180.)).sortby('lon')
        out.attrs['lon_flip'] = 'True'
    else:
        out = data
        out.attrs['lon_flip'] = 'True'
    if isinstance(out, xarray.Dataset):
        for _var in out.data_vars:
            if 'lon' in out.get(_var).dims:
                out.get(_var).attrs['lon_flip'] = 'True'
    return out


def lon_flip_VERY_OLD(data, igreenwich = None):
    """
    """
    data = check_lat_lon_names(data)
    flipped = util.check_bool_attribute(data, 'lon_flip')
    if not flipped or data.lon[0] == 0.:
        i = int(len(data.lon)/2.) if igreenwich is None else igreenwich
        print(data)
        _tmp = data.roll(lon = i)
        print(_tmp)
        _lon = (((_tmp.lon + 180.) % 360.) - 180.)
        print(_lon)
        _lon.attrs = _tmp.lon.attrs
        out = _tmp.assign_coords(lon = _lon)
        out.attrs['lon_flip'] = 'True'
        out = out.sortby('lon')
    else:
        out = data
    if isinstance(out, xarray.Dataset):
        for _var in out.data_vars:
            if 'lon' in out.get(_var).dims:
                out.get(_var).attrs['lon_flip'] = 'True'
    return out


def add_cyclic_point(data):
    # usefull tool before ploting or before interpolating
    util.check_Xarray(data)
    _data = check_lat_lon_names(data)
    if isinstance(_data, xarray.DataArray):
        _ds = _data.to_dataset(); _ds.attrs = _data.attrs; _to_da = True
    else:
        _ds = _data; _to_da = False
    cycled = util.check_bool_attribute(_ds, 'add_cyclic_point')
    if not cycled:
        _ds_out = xarray.Dataset(data_vars = {'empty': [0]}, attrs = _ds.attrs)
        _lon = xarray.DataArray(numpy.insert(_ds.lon.values, len(_ds.lon.values+1), _ds.lon.values[0] + 360.),
                                dims = 'lon',
                                coords = {'lon': numpy.insert(_ds.lon.values, len(_ds.lon.values+1), _ds.lon.values[0] + 360.)},
                                attrs = _ds.lon.attrs)
        for _var in _ds.variables:
            if _var == 'lon':
                _ds_out = _ds_out.assign({'lon': _lon})
            elif 'lon' in _ds.get(_var).dims:
                _tmp, _ = cartopy.util.add_cyclic_point(_ds.get(_var), coord = _ds.lon.values) # axis???
                _ds_out = _ds_out.assign({_var: xarray.DataArray(_tmp,
                                                                 dims = _ds.get(_var).dims,
                                                                 coords = _ds.get(_var).drop('lon').coords.merge({'lon': _lon}).coords,
                                                                 attrs = _ds.get(_var).attrs)})
            else:
                _ds_out = _ds_out.assign({_var: _ds.get(_var)})
        _ds_out = _ds_out.drop('empty')
        _ds_out.attrs['add_cyclic_point'] = True
    else:
        _ds_out = _ds
    return _ds_out.to_array(name=_data.name).isel(variable=0).drop('variable') if _to_da else _ds_out


def interpolate_to_regular_earth_grid(data:xarray.DataArray, 
                                      degree = None, lon0 = -180.,
                                      lat = None, lon = None,
                                      sftlf = None,
                                      method = 'linear', cyclic = True):
    ### Change the name of the function: make evident that coastline points might not be correctly computed.
    if degree is None and (lat is None or lon is None):
        raise ValueError('Degree and lat or lon are missing. The destination grid is unknown.')
    _data = check_lat_lon_names(data)
    if degree is not None or lon[0] != 0:
        _data = lon_flip(_data)
    if isinstance(degree, float) or isinstance(degree, int):
        lat = numpy.arange(-90., 90. + degree, degree)
        lon = numpy.arange(lon0, lon0 + 360. + degree, degree)
    #if cyclic:
    #    if not isinstance(_data, xarray.Dataset):
    #        name = _data.name; _data = _data.to_dataset(); _ds_to_da = True;
    #    _data = add_cyclic_point(_data)
    #    if _ds_to_da:
    #        _data = _data.get(name)
    _data = _flank_longitudes(_data)
    return _data.interp(lat = lat, lon= lon, method = method)


def interpolate_coast(data:xarray.DataArray, data_sftlf:xarray.DataArray, sftlf:xarray.DataArray,
                      method = 'linear', land_only = False, ocean_only = False,
                      eps = 1e-4, relax = 0.6, itermax = 1e4, initzonal = False, cyclic = False, verbose = False,
                      out_print=False,
                     ):
    _data = lon_flip(check_lat_lon_names(data))
    if data_sftlf is not None:
        _data_sftlf = lon_flip(check_lat_lon_names(data_sftlf))
    _sftlf = lon_flip(check_lat_lon_names(sftlf))
    _kw = dict(eps=eps, relax=relax, itermax=itermax, initzonal=initzonal, cyclic=cyclic, verbose=verbose)
    # LAND
    if not ocean_only:
        if not land_only:
            _land = _data.where(_data_sftlf > 50.)
        else:
            _land = _data
        # Fill missing data on ocean with values derived from solving Poisson's equation via relaxation
        _filled, _converged = gridfill.fill(_land.to_masked_array(), _land.dims.index('lat'), _land.dims.index('lon'), **_kw)
        if not all(i for i in _converged):
            print('WARNING: At least one grid filling did not converge!')
        _land = _flank_longitudes(xarray.DataArray(_filled, dims = _data.dims, coords = _data.coords)).interp(lat = sftlf.lat, lon = sftlf.lon, method = method)
        if out_print:
            print('Land extra- and interpolated.')
    # OCEAN
    if not land_only:
        if not ocean_only:
            _ocean = _data.where(_data_sftlf < 50.)
        else:
            _ocean = _data
        # Fill missing data on ocean with values derived from solving Poisson's equation via relaxation
        _filled, _converged = gridfill.fill(_ocean.to_masked_array(), _ocean.dims.index('lat'), _ocean.dims.index('lon'), **_kw)
        if not all(i for i in _converged):
            print('WARNING: At least one grid filling did not converge!')
        _ocean = _flank_longitudes(xarray.DataArray(_filled, dims = _data.dims, coords = _data.coords)).interp(lat = sftlf.lat, lon = sftlf.lon, method = method)
        if out_print:
            print('Ocean extra- and interpolated.')
    # Join land and ocean
    if land_only:
        _out = _land.where(_sftlf > 50.)
    elif ocean_only:
        _out = _ocean.where(_sftlf < 50.)
    else:
        _out = _land.fillna(0.) * _sftlf / 100. + _ocean.fillna(0.) * (100. - _sftlf) / 100.
    try:
        _out = _out.drop('type')
    except:
        pass
    if out_print:
        print('Coastal interpolation completed.')
    return _out


def _flank_longitudes(data):
    _data = check_lat_lon_names(data)
    _left = _data.sel(lon = _data.lon[-1])
    _left.coords['lon'] = xarray.DataArray(_left.lon.values-360., dims = _left.lon.dims, coords = _left.lon.coords, attrs = _left.lon.attrs)
    _right = _data.sel(lon = _data.lon[0])
    _right.coords['lon'] = xarray.DataArray(_right.lon.values+360., dims = _right.lon.dims, coords = _right.lon.coords, attrs = _right.lon.attrs)
    return xarray.concat([_left, _data, _right], dim = 'lon')


def mask_land(data, mask = None): # should keep attrs! # mask_ocean
    """
    Add option for lon_flip
    """
    util.check_Xarray(data)
    _data = check_lat_lon_names(data)
    _mask = xarray.open_dataset('/data/scratch/globc/dcom/FIXED_FIELDS/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc').sftlf if mask is None else mask
    _mask = check_lat_lon_names(_mask)
    _data, _mask = check_lat_lon_coords(_data, _mask)
    if isinstance(_data, xarray.Dataset):
        _das = [_data.get(_var) for _var in _data.data_vars]; _to_ds = True;
    elif isinstance(_data, xarray.DataArray):
        _das = [_data]; _to_ds = False;
    _out = []
    for _da in _das:
        if 'lat' in _da.dims and 'lon' in _da.dims:
            _tmp = _da.where(_mask > 50.)
            _tmp.attrs = _da.attrs
            _tmp.attrs['mask_land'] = 'True'
            _out.append(_tmp)
        else:
            _out.append(_da)
        #_out[-1].attrs = _da.attrs
    if _to_ds:
        out = xarray.merge(_out)
        out.attrs = data.attrs
    else:
        out = _out[0]
    return out


def mask_sea(data, mask = None): # should keep attrs! # mask_ocean
    """
    """
    util.check_Xarray(data)
    _data = check_lat_lon_names(data)
    _mask = xarray.open_dataset('/data/scratch/globc/dcom/FIXED_FIELDS/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc').sftlf if mask is None else mask
    _mask = check_lat_lon_names(_mask)
    _data, _mask = check_lat_lon_coords(_data, _mask)
    if isinstance(_data, xarray.Dataset):
        _das = [_data.get(_var) for _var in _data.data_vars]; _to_ds = True;
    elif isinstance(_data, xarray.DataArray):
        _das = [_data]; _to_ds = False;
    _out = []
    for _da in _das:
        if 'lat' in _da.dims and 'lon' in _da.dims:
            _tmp = _data.where(_mask < 50.)
            _tmp.attrs = _data.attrs
            _tmp.attrs['mask_sea'] = 'True'
            _out.append(_tmp)
    return xarray.merge(_out) if _to_ds else _out[0]


def reg_lat_lon(x, y, fit_intercept = False): ### a reprendre de maniere propre !
    # x[time]
    # y[time, lat, lon]
    ### check if len(Y.shape) <= 3
    _sklinreg = sklearn.linear_model.LinearRegression(fit_intercept = fit_intercept)
    _shape = numpy.array(y).shape
    _X = numpy.array(x)[:, numpy.newaxis]
    _Y = numpy.array(y).reshape(_shape[0], -1)
    _imputer = sklearn.impute.SimpleImputer(keep_empty_features=True)
    _X = _imputer.fit_transform(_X)
    _sklinreg.fit(_X, _Y)
    _out = xarray.DataArray(_sklinreg.coef_.reshape(_shape[1:]),
                            dims = ('lat', 'lon'),
                            coords = {'lat': y.lat, 'lon': y.lon})
    return _out

def reg_lat_lon_TO_TRY(x, y, fit_intercept = False): ### a reprendre de maniere propre !
    from sklearn.impute import SimpleImputer, MissingIndicator
    #sklinreg = sklearn.linear_model.LinearRegression(fit_intercept = fit_intercept)
    transformer = FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy='mean')),
            ('indicators', MissingIndicator())])
    shape = numpy.array(y).shape
    X = numpy.array(x)[:, numpy.newaxis]
    Y = numpy.array(y).reshape(shape[0], -1)
    #sklinreg.fit(X, Y)
    transformer = transformer.fit(X, Y)
    out = xarray.DataArray(transformer.coef_.reshape(shape[1:]),
                           dims = ('lat', 'lon'),
                           coords = {'lat': y.lat, 'lon': y.lon})
    return out


def get_shape(shape_file, name): #(shape_file:str, name:str)
    """
    """

    sf = shapefile.Reader(shape_file)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pandas.DataFrame(columns = fields, data = records)
    df = df.assign(coords = shps)
    ind = df[df.Acronym == name].index[0]

    return sf.shape(ind)


def shape_to_mask(shape, lat, lon, lon_flipped=False):
    """
    INSPIRED FROM: https://stackoverflow.com/questions/34585582/how-to-mask-the-specific-array-data-based-on-the-shapefile
    
    Create mask from outline contour of shape.

    Parameters
    ----------
    shape: shape polygon #line: array-like (N, 2)
    lat, lon: 1-D grid coordinates (input for meshgrid)

    Returns
    -------
    mask: 2-D boolean array (True inside)
    
    Work remaining
    --------------
    auto-determine is lonFlip is needed or not
    build lon and lat if they are not given? n_lat and n_lon?
    """
    
    _line = numpy.zeros((len(shape.points),2))
    for _ip in range(len(shape.points)):
        _line[_ip,0] = shape.points[_ip][0]
        _line[_ip,1] = shape.points[_ip][1]

    if lon_flipped:
        _lon_remove = 0.
    else:
        _lon_remove = 180.
    _lon, _lat = numpy.meshgrid(lon - _lon_remove, lat)
    _points = numpy.array((_lon.flatten(), _lat.flatten())).T
    _mask = matplotlib.path.Path(_line).contains_points(_points).reshape(_lat.shape)
    #_ind = numpy.concatenate((numpy.arange(len(lon)/2,len(lon),dtype=int),numpy.arange(0,len(lon)/2,dtype=int)))
    
    ### attention : il faut éventuellement unflip... à creuser !

    return _mask#[:,_ind]


def average_in_shape(data, shape): #, mask_land = False, mask_ocean = False, mask_file = None):
    """
    mask_land: keep only land points
    mask_ocean: keep only ocean points
    """
    ### CAUTION: data should not be lonFlipped...
    util.check_Xarray(data)
    _data = check_lat_lon_names(data)

    #if mask_land:
    #    mask_file = mask_file or '/data/scratch/globc/dcom/FIXED_FIELDS/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'
    #    land = xarray.open_dataset(mask_file)
    #    data = data.where(land['sftlf'] > 50.)
    #elif mask_ocean:
    #    mask_file = mask_file or '/data/scratch/globc/dcom/FIXED_FIELDS/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'
    #    land = xarray.open_dataset(mask_file)
    #    data = data.where(land['sftlf'] <= 50.)

    flipped = util.check_bool_attribute(data, 'lon_flip')

    out = field_avg(_data.where(shape_to_mask(shape, _data.lat, _data.lon, lon_flipped = flipped)))
    out.attrs = data.attrs

    return out


def sum_in_shape(data, shape, area=None):
    ### CAUTION: data should not be lonFlipped...
    util.check_Xarray(data)
    _data = check_lat_lon_names(data)

    flipped = util.check_bool_attribute(data, 'lon_flip')

    out = field_sum(_data.where(shape_to_mask(shape, _data.lat, _data.lon, lon_flipped = flipped)), area=area)
    out.attrs = data.attrs

    return out








