from . import util
from .tools.time import get_season
from .tools.field import check_lat_lon_names, field_avg, lon_flip, reg_lat_lon
from .stats.scipy import get_trend

import numpy
import xarray
import eofs.xarray
import statsmodels.api

"""
Need to be specified in DataArray outputs: long_name units standard_name description
! In the Python code, copy and add Dataset information before saving: original_file ...
"""

def _eof(mode, data, season='JFM',
         solver=None,
         north=80,east=40,south=20,west=-90, # north=80,east=30,south=20,west=-90,
         start_year=None, end_year=None):

    _tmp = get_season(lon_flip(check_lat_lon_names(data)).sel(lat = slice(south, north), lon = slice(west, east)), season)
    _time = _tmp.time; _tmp = _tmp.dropna(dim='time', how='all')

    if solver is None:
        if None not in [start_year, end_year]:
            _tmp = _tmp.sel(time = slice(start_year, end_year))
        # solver = eofs.xarray.Eof(_tmp.transpose(..., 'lat'), weights = numpy.sqrt(numpy.cos(numpy.deg2rad(_tmp.lat))))
        solver = eofs.xarray.Eof(_tmp.transpose(..., 'lat'), weights = numpy.cos(numpy.deg2rad(_tmp.lat)))
        _index = solver.pcs(npcs=mode+1, pcscaling = 1)
    else:
        _index = solver.projectField(_tmp.transpose(..., 'lat'), neofs=mode+1, eofscaling=2)
        _index = (_index - _index.mean(dim = 'time', keep_attrs = True)) / _index.std(dim = 'time', keep_attrs = True)
    _pattern = solver.eofs(neofs=mode+1, eofscaling = 2)

    _index = _index.isel(mode=mode).drop('mode')
    if _time[0] != _index.time[0]:
        _tmp = _time.sel(time=slice(None,_index.time[0]))[:-1]
        _list=list()
        for _t in _tmp:
            _list.append(xarray.full_like(_index.isel(time=0), fill_value=numpy.nan).assign_coords(time=_t))
        _index = xarray.concat([xarray.concat(_list, dim='time'), _index], dim='time')
    ### Do the same thing on the other side!
    _pattern = _pattern.isel(mode=mode).drop('mode').transpose(..., 'lon')
    _variance_fraction = solver.varianceFraction(neigs=mode+1)
    _index.attrs['variance_fraction'] = _variance_fraction.values[mode]
    _pattern.attrs['variance_fraction'] = _variance_fraction.values[mode]

    _index.attrs['units'] = ''

    if 'member' in data.coords:
        _index = _index.assign_coords({'member': data.member.values})

    return _index, _pattern, solver


def nao(data, season='JFM',
        solver=None, get_solver=False,
        north=80,east=40,south=20,west=-90,
        start_year=None, end_year=None):

    util.check_DataArray(data)

    _mode = 0

    _index, _pattern, _solver = _eof(_mode, data, season=season,
                                     solver=solver,
                                     north=north,east=east,south=south,west=west,
                                     start_year=start_year, end_year=end_year)

    _index = _index.rename('nao')
    _index.attrs['name'] = 'NAO'
    _index.attrs['long_name'] = 'North Atlantic Oscillation'
    _index.attrs['mode'] = _mode

    if _pattern.sel(lat=64., lon=-22., method="nearest") > 0.:
        _pattern *= -1.
        _index *= -1.
    _pattern = _pattern.rename('nao_pattern')
    _pattern.attrs['name'] = 'NAO'
    _pattern.attrs['long_name'] = 'Pattern of the North Atlantic Oscillation'
    _pattern.attrs['mode'] = _mode

    out = [_index, _pattern]
    if get_solver :
        out.append(_solver)
    return tuple(out)


def OLD_nao(data, season='JFM',
        solver=None, get_solver=False, #solver_id=None, -> in the python code saving files!
        north=80,east=40,south=20,west=-90,
        start_year=None, end_year=None):

    util.check_DataArray(data)

    _tmp = get_season(lon_flip(check_lat_lon_names(data)).sel(lat = slice(south, north), lon = slice(west, east)), season)

    if solver is None:
        if None not in [start_year, end_year]:
            _tmp = _tmp.sel(time = slice(start_year, end_year))
        solver = eofs.xarray.Eof(_tmp * numpy.cos(numpy.deg2rad(_tmp.lat)))
        _index = solver.pcs(npcs=1, pcscaling = 1)
    else:
        _index = solver.projectField(_tmp, neofs=1, eofscaling=2)
        _index = (_index - _index.mean(dim = 'time', keep_attrs = True)) / _index.std(dim = 'time', keep_attrs = True)
    _index = _index.rename('nao')
    _index.attrs['name'] = 'NAO'
    _index.attrs['long_name'] = 'North Atlantic Oscillation'
    _index.attrs['mode'] = _index.mode.values[0]
    _index = _index.isel(mode=0).drop('mode')

    _pattern = solver.eofs(neofs=1, eofscaling = 2)
    if _pattern.sel(lat=64., lon=-22., method="nearest") > 0.:
        _pattern *= -1.
        _index *= -1.
    _pattern = _pattern.rename('nao_pattern')
    _pattern.attrs['name'] = 'NAO'
    _pattern.attrs['long_name'] = 'Pattern of the North Atlantic Oscillation'
    _pattern.attrs['mode'] = _pattern.mode.values[0]
    _pattern = _pattern.isel(mode=0).drop('mode')

    _variance_fraction = solver.varianceFraction(neigs=1)
    _index.attrs['variance_fraction'] = _variance_fraction.values[0]
    _pattern.attrs['variance_fraction'] = _variance_fraction.values[0]

    out = [_index, _pattern]
    if get_solver :
        out.append(solver)
    return tuple(out)


def eap(data, season='JFM',
        solver=None, get_solver=False,
        north=80,east=40,south=20,west=-90,
        start_year=None, end_year=None):

    util.check_DataArray(data)

    _mode = 1

    _index, _pattern, _solver = _eof(_mode, data, season=season,
                                     solver=solver,
                                     north=north,east=east,south=south,west=west,
                                     start_year=start_year, end_year=end_year)

    _index = _index.rename('eap')
    _index.attrs['name'] = 'EAP'
    _index.attrs['long_name'] = 'East Atlantic Pattern'
    _index.attrs['mode'] = _mode

    if _pattern.sel(lat=50., lon=-25., method="nearest") > 0.:
        _pattern *= -1.
        _index *= -1.
    _pattern = _pattern.rename('eap_pattern')
    _pattern.attrs['name'] = 'EAP'
    _pattern.attrs['long_name'] = 'Pattern of the East Atlantic Pattern'
    _pattern.attrs['mode'] = _mode

    out = [_index, _pattern]
    if get_solver :
        out.append(_solver)
    return tuple(out)


def bl(data, season='JFM',
        solver=None, get_solver=False,
        north=80,east=40,south=20,west=-90,
        start_year=None, end_year=None):

    util.check_DataArray(data)

    _mode = 2

    _index, _pattern, _solver = _eof(_mode, data, season=season,
                                     solver=solver,
                                     north=north,east=east,south=south,west=west,
                                     start_year=start_year, end_year=end_year)

    _index = _index.rename('bl')
    _index.attrs['name'] = 'BL'
    _index.attrs['long_name'] = 'Blocking'
    _index.attrs['mode'] = _mode

    if _pattern.sel(lat=60., lon=-44., method="nearest") > 0.:
        _pattern *= -1.
        _index *= -1.
    _pattern = _pattern.rename('bl_pattern')
    _pattern.attrs['name'] = 'BL'
    _pattern.attrs['long_name'] = 'Pattern of the Blocking'
    _pattern.attrs['mode'] = _mode

    out = [_index, _pattern]
    if get_solver :
        out.append(_solver)
    return tuple(out)

def amv(data:xarray.DataArray, method='Trenberth & Shea',
        north_atlantic_basin={'north': 60.,'east': 0.,'south': 0.,'west': -80.}, # 'north': 65.
        global_basin={'north': 60.,'east': 180.,'south': -60.,'west': -180.},
        temporality='yr',
        sftlf:xarray.DataArray=None,
        forcing:xarray.DataArray=None,
        pattern:xarray.DataArray=None,
        lowess_filter=None,
        get_pattern=False):
    """
    Computes the standardized AMV index based upon the average anomalies of sea surface temperatures in the North Atlantic basin. #detrended

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    method : str
        'Enfeld et al.', 'linear': Linear detrending is used in the traditional defnition of the AMV index, which corresponds to the average over the entire North Atlantic basin of the linearly detrended SST yearly anomalies, as proposed by Enfeld et al. (2001).
        'Trenberth & Shea', 'trshea' (DEFAULT): In Trenberth and Shea (2006), the annual observed near-global-mean (60N-60S) SST anomaly time series is subtracted from the observed annual North Atlantic spatially averaged time series to obtain the raw unfltered AMV index.
        'demeaned': estimated forcing is given as an imput, typically an ensemble average from model(s)
    north_atlantic_basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed
        (mask and shape still need to be added to the function)
    global_basin : dict or mask or shape, optional
        Indication of the global basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    forcing : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
        Must have the same grid as 'data'.
    pattern : DataArray (lat, lon)
        Sea surface temperature average pattern.
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
        Traditionaly, a 10-year running mean is applied.
    get_pattern : bool, optional
        If true, returns the regression of the AMV index on sea surface temperature in the North Atlantic basin.
    """

    _name = 'amv'

    util.check_DataArray(data); _data = check_lat_lon_names(data);

    # convert data to celcius temperature unit
    _data = util.to_celcius(_data)

    # keep only non frozen cells and flip longitude indexes
    _data = lon_flip(_data.where(_data > -1.8))

    if temporality == 'yr':
        # get annual average
        _data = _data.resample(time = 'Y').mean(dim = 'time', keep_attrs = True)
    # elif temporality == 'mon':
    #     _attrs = _data.attrs
    #     _data = _data.groupby('time.month') - _data.sel(time=slice(ref_start,ref_stop)).groupby('time.month').mean('time'); _data.attrs = _attrs; _data.attrs['units'] = 'C'

    # keep only ocean cells
    if sftlf is not 'Ignore':
        util.check_DataArray(sftlf); _sftlf = lon_flip(check_lat_lon_names(sftlf));
        _data = _data.where(_sftlf < 50.)
        _data = util.drop(_data, 'type')

    # remove the estimated externally forced signal
    if method in ('Enfeld et al.', 'linear'):
        _forcing = get_trend(_data.fillna(0))
        _name += 'Linear'
    if method in ('Trenberth & Shea', 'Trenberth and Shea', 'trshea'):
        if isinstance(global_basin, dict):
            _forcing = field_avg(_data.sel(lat = slice(global_basin['south'], global_basin['north']),
                                           lon = slice(global_basin['west'], global_basin['east'])))
            _name += 'Trshea'
        else:
            raise OptionNotCodedYet
    elif method in ('demeaned'):
        util.check_DataArray(forcing)
        _forcing = lon_flip(util.to_celcius(check_lat_lon_names(forcing)))
        _forcing = lon_flip(_forcing.where(_forcing > -1.8))
        _name += 'Demeaned'
    elif method in (None, 'nasst'):
        _forcing = None
        _name = 'nasst'
    _resi = _data - _forcing if forcing is not None else _data

    # restrict the calculation to the North Atlantic basin
    if isinstance(north_atlantic_basin, dict):
        _resi = _resi.sel(lat = slice(north_atlantic_basin['south'], north_atlantic_basin['north']),
                          lon = slice(north_atlantic_basin['west'], north_atlantic_basin['east']))
    else:
        raise OptionNotCodedYet

    # compute the AMV index (standardized)
    if pattern is not None:
        util.check_DataArray(pattern)
        _pattern = check_lat_lon_names(pattern)
        _name += 'Projected'
        # project sea surface temperature on the AMV input pattern
        _un_stded = _resi.fillna(0).dot(_pattern.fillna(0))
    else:
        _un_stded = field_avg(_resi)
    if method != 'nasst':
        _amv = (_un_stded - _un_stded.mean(dim = 'time')) / _un_stded.std(dim = 'time')
    else:
        _amv = _un_stded
    _amv = xarray.DataArray(
        data = _amv,
        coords = _amv.coords, dims = _amv.dims,
        name = _name, attrs = _data.attrs)

    # optionally apply a lowess filter 
    if lowess_filter is not None:
        lowess = statsmodels.api.nonparametric.lowess
        _amv = xarray.DataArray(
            data = lowess(_amv, _amv.time.dt.year,
                          return_sorted = False, frac = lowess_filter/len(_amv.time)),
            coords = _amv.coords, dims = _amv.dims, name = _amv.name, attrs = _amv.attrs)
        _amv.attrs['low-pass filter'] = str(lowess_filter)+' years'

    # optionally get the pattern of regression between the AMV index and sea surface temperature
    if get_pattern:
        _data = _data.fillna(0)
        _pattern = reg_lat_lon(_amv, _data - get_trend(_data)) # add informations to the DataArray
        _pattern.name = 'pattern'
        _pattern.attrs['long_name'] = 'Regression pattern of the AMV index on SST'
        _pattern.attrs['unit'] = 'C'
        if sftlf is not 'Ignore':
            _pattern = _pattern.where(_sftlf < 50.)
            _pattern = util.drop(_pattern, 'type')
        out = (_amv, _pattern)
    else:
        out = _amv

    return out


def _nasst(data:xarray.DataArray,
        basin={'north': 65,'east': 0,'south': 0,'west': -80},
        sftlf:xarray.DataArray=None,
        lowess_filter=None,
        get_pattern=False):
    """
    Computes the AMV index based upon the average anomalies of sea surface temperatures in the North Atlantic basin.

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
    get_pattern : bool, optional
        If true, returns the regression of the AMV index on sea surface temperature in the North Atlantic basin.
    """

    util.check_DataArray(data); _data = check_lat_lon_names(data);

    # convert to one temperature unit: celcius
    _data = util.to_celcius(_data)

    # keep only non frozen cells, get annual average, convert to celcius, and flip longitude indexes
    _data = lon_flip(_data.where(_data > -1.8).resample(time = 'Y').mean(dim = 'time', keep_attrs = True))

    # restrict the calculation to the North Atlantic basin
    if isinstance(basin, dict):
        if sftlf is not 'Ignore':
            util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf);
            _data = _data.where(_sftlf < 50.)
        _data = _data.sel(lat = slice(basin['south'], basin['north']), lon = slice(basin['west'], basin['east']))
    else:
        raise OptionNotCodedYet

    # compute the average
    _amv = field_avg(_data)
    try:
        _amv = _amv.drop('type')
    except:
        pass

    ### add relevant informations to the DataArray

    # optionally apply a lowess filter 
    if lowess_filter is not None:
        lowess = statsmodels.api.nonparametric.lowess
        _amv = xarray.DataArray(data = lowess(_amv, _amv.time.dt.year,
                                              return_sorted = False, frac = lowess_filter/len(_amv.time)),
                                coords = _amv.coords, dims = _amv.dims, name = _amv.name, attrs = _amv.attrs)

    # optionally get the pattern of regression between the AMV index and sea surface temperature
    if get_pattern:
        _pattern = reg_lat_lon((_amv-_amv.mean())/_amv.std(), _data.fillna(0)) # add informations to the DataArray
        if sftlf is not 'Ignore':
            util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf);
            _pattern = _pattern.where(_sftlf < 50.)
            try:
                _pattern = _pattern.drop('type')
            except:
                pass
        out = (_amv, _pattern)
    else:
        out = _amv

    return out


def amv_projected(data:xarray.DataArray,
                  pattern:xarray.DataArray,
                  basin={'north': 65,'east': 0,'south': 0,'west': -80},
                  sftlf: xarray.DataArray=None,
                  lowess_filter=None):
    """
    Computes the AMV index based upon an input average anomalies of sea surface temperatures pattern in the North Atlantic basin.

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    pattern : DataArray (lat, lon)
        Sea surface temperature average pattern.
    basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
    """

    util.check_DataArray(data); util.check_DataArray(pattern); _data = check_lat_lon_names(data); _pattern = check_lat_lon_names(pattern);

    _data = lon_flip(util.to_celcius(_data.resample(time = 'Y').mean(dim = 'time', keep_attrs = True)))

    if isinstance(basin, dict):
        util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf);
        _data = _data.where(_sftlf < 50.).sel(lat = slice(basin['south'], basin['north']), lon = slice(basin['west'], basin['east']))
    else:
        raise OptionNotCodedYet

    # project sea surface temperature on the AMV input pattern, then normalize
    _un_stded = _data.where(_data > -1.8).fillna(0).dot(_pattern.fillna(0))
    _amv_projected = xarray.DataArray(data = _un_stded / _un_stded.std(dim = 'time'),
                                      coords = _un_stded.coords, dims = _un_stded.dims,
                                      name = 'amv_projected', attrs = _data.attrs)

    # optionally apply a lowess filter 
    if lowess_filter is not None:
        lowess = statsmodels.api.nonparametric.lowess
        _amv_projected = xarray.DataArray(data = lowess(_amv_projected, _amv_projected.time.dt.year,
                                                        return_sorted = False, frac = lowess_filter/len(_amv_projected.time)),
                                          coords = _amv_projected.coords, dims = _amv_projected.dims,
                                          name = _amv_projected.name, attrs = _amv_projected.attrs)

    return _amv_projected


def amv_trshea(data: xarray.DataArray,
               north_atlantic_basin={'north': 65.,'east': 0.,'south': 0.,'west': -80.},
               global_basin={'north': 60.,'east': 180.,'south': -60.,'west': -180.},
               sftlf: xarray.DataArray=None,
               lowess_filter=None,
               get_pattern=False,
               extra_outputs=False):
    """
    Computes the AMV index as recommended by Trenberth and Shea (2006).

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    north_atlantic_basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed
        (mask and shape still need to be added to the function)
    global_basin : dict or mask or shape, optional
        Indication of the global basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
    get_pattern : bool, optional
        If true, returns the regression of the AMV index on sea surface temperature in the North Atlantic basin.
    extra_outputs : bool, optional
        If True, the North Atlantic sea surface temperature average (NASST)
        and the global sea surface temperature average (GloSST) variables are returned.
    """
 
    util.check_DataArray(data)

    _nasst = _nasst(data, north_atlantic_basin, sftlf, lowess_filter)
    _glosst = _nasst(data, global_basin, sftlf, lowess_filter)

    _amv_trshea = xarray.DataArray(data = _nasst.values - _glosst.values,
                                   coords = _nasst.coords, dims = _nasst.dims, name = 'amv_trshea', attrs = _nasst.attrs)

    _out = [_amv_trshea]

    # optionally get the pattern of regression between the AMV index and sea surface temperature
    if get_pattern:
        util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf); _data = check_lat_lon_names(data);
        
        # convert to one temperature unit: celcius
        _data = util.to_celcius(_data)

        # keep only non frozen cells, get annual average, convert to celcius, and flip longitude indexes
        _data = lon_flip(util.to_celcius(_data.where(_data > -1.8).resample(time = 'Y').mean(dim = 'time', keep_attrs = True)))
        # restrict the calculation to the North Atlantic basin
        if isinstance(north_atlantic_basin, dict):
            util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf);
            _data = _data.where(_sftlf < 50.).sel(lat = slice(north_atlantic_basin['south'], north_atlantic_basin['north']), lon = slice(north_atlantic_basin['west'], north_atlantic_basin['east']))
        else:
            raise OptionNotCodedYet
        _pattern = reg_lat_lon((_amv_trshea-_amv_trshea.mean())/_amv_trshea.std(), _data.fillna(0)).where(_sftlf < 50.) # add informations to the DataArray
        try:
            _pattern = _pattern.drop('type')
        except:
            pass
        _out.append(_pattern)

    if extra_outputs:
        _out.append(_nasst)
        _out.append(_glosst)

    return tuple(_out) if len(_out) > 1 else _out[0]


def amv_demeaned(data:xarray.DataArray,
                 forcing:xarray.DataArray,
                 basin={'north': 65,'east': 0,'south': 0,'west': -80},
                 sftlf:xarray.DataArray=None,
                 lowess_filter=None,
                 get_pattern=False):
    """
    Computes the AMV index based upon the average anomalies of sea surface temperatures (SST) in the North Atlantic basin,
    removing a pattern of SST corresponding to the estimation of the forcing impact.

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    forcing : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
    get_pattern : bool, optional
        If true, returns the regression of the AMV index on sea surface temperature in the North Atlantic basin.
    """

    util.check_DataArray(data); util.check_DataArray(forcing);
    _data = lon_flip(util.to_celcius(check_lat_lon_names(data))); _forcing = lon_flip(util.to_celcius(check_lat_lon_names(forcing)));

    return _nasst(data=xarray.DataArray(_data.values - _forcing.values,
                                        dims = _data.dims, coords = _data.coords, attrs = _data.attrs),
               basin=basin, sftlf=sftlf, lowess_filter=lowess_filter, get_pattern=get_pattern)

def amv_demeaned_projected(data:xarray.DataArray,
                           forcing:xarray.DataArray,
                           pattern:xarray.DataArray,
                           basin={'north': 65,'east': 0,'south': 0,'west': -80},
                           sftlf: xarray.DataArray=None,
                           lowess_filter=None):
    """
    Computes the AMV index based upon an input average anomalies of sea surface temperatures pattern in the North Atlantic basin. The projected SST is demeaned by a temporal pattern given as input.

    Parameters
    ----------
    data : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    forcing : DataArray (time, lat, lon)
        Sea surface temperature (or equivalent).
    pattern : DataArray (lat, lon)
        Sea surface temperature average pattern.
    basin : dict or mask or shape, optional
        Indication of the North Atlantic basin delimitation.
        If dict, then 'sftlf' is needed.
        (mask and shape still need to be added to the function)
    sftlf : DataArray (lat, lon), optional under conditions
        Percentage of the grid cell occupied by land (including lakes).
        Must have the same grid as 'data'.
    lowess_filter : int, optional
        If not None, applies a Lowess filter of the corresponding years.
    """

    util.check_DataArray(data); util.check_DataArray(forcing); util.check_DataArray(pattern);
    _data = util.to_celcius(check_lat_lon_names(data)); _forcing = util.to_celcius(check_lat_lon_names(forcing)); _pattern = check_lat_lon_names(pattern);

    _data = xarray.DataArray(_data.values - _forcing.values,
                             dims = _data.dims, coords = _data.coords, attrs = _data.attrs)

    _data = lon_flip(_data.resample(time = 'Y').mean(dim = 'time', keep_attrs = True))

    if isinstance(basin, dict):
        util.check_DataArray(sftlf); _sftlf = check_lat_lon_names(sftlf);
        _data = _data.where(_sftlf < 50.).sel(lat = slice(basin['south'], basin['north']), lon = slice(basin['west'], basin['east']))
    else:
        raise OptionNotCodedYet

    try:
        _data = _data.drop('type')
    except:
        pass

    # project sea surface temperature on the AMV input pattern, then normalize
    _un_stded = _data.where(_data > -1.8).fillna(0).dot(_pattern.fillna(0))
    _amv_demeaned_projected = xarray.DataArray(data = _un_stded / _un_stded.std(dim = 'time'),
                                               coords = _un_stded.coords, dims = _un_stded.dims,
                                               name = 'amv_demeaned_projected', attrs = _data.attrs)

    try:
        _amv_demeaned_projected = _amv_demeaned_projected.drop('type')
    except:
        pass

    # optionally apply a lowess filter 
    if lowess_filter is not None:
        lowess = statsmodels.api.nonparametric.lowess
        _amv_demeaned_projected = xarray.DataArray(data = lowess(_amv_demeaned_projected, _amv_demeaned_projected.time.dt.year,
                                                          return_sorted = False, frac = lowess_filter/len(_amv_demeaned_projected.time)),
                                                   coords = _amv_demeaned_projected.coords, dims = _amv_demeaned_projected.dims,
                                                   name = _amv_demeaned_projected.name, attrs = _amv_demeaned_projected.attrs)

    return _amv_demeaned_projected


def gnb(data:xarray.DataArray,
        mask:xarray.DataArray=None,
        area: xarray.DataArray=None):

    if mask is None:
        ds_mask = xarray.open_dataset('/archive/globc/cassou/evian/CNRM_CM6/Masks/mask_GNB_ORCA1.nc')
        mask = ds_mask.get('mask_GNB')
        mask = mask.assign_coords({'nav_lon': ds_mask.nav_lon, 'nav_lat': ds_mask.nav_lat})

    if area is None:
        area = xarray.open_dataset('/data/scratch/globc/dcom/FIXED_FIELDS/CMIP6/DECK/CNRM-CM6-1_historical_r1i1p1f2/areacello_Ofx_CNRM-CM6-1_historical_r1i1p1f2_gn.nc').get('areacello')

    out = (data * mask * area / 100.).sum(dim=('x', 'y')) / 1000. ** 2
    out.attrs = data.attrs
    out.attrs['name'] = 'GNB Ice'
    out.attrs['long_name'] = 'Greenland-Norway-Batlands Ice Cover'
    out.attrs['units'] = 'km2'
    out = out.drop('type')

    return out


def enso(data:xarray.DataArray,
         index='3.4',
         rolling:int=None,
         starting_month:int=None,
         normalise=True, detrend=True, unforced=None,
         ref_start='1960', ref_end='1989'):
    _lat = data.lat.values; _lat.sort()
    _out = data.sel(lat=_lat)
    if index == 'ONI':
        _name = 'ONI'; _index_region = '3.4'
        _out = _out.sel(lat=slice(-5,5),lon=slice(360-170,360-120))
        rolling = rolling or 3
    elif index == '1+2':
        _name = 'NINO1+2'; _index_region = '1+2'
        _out = _out.sel(lat=slice(-10,0),lon=slice(360-90,360-80))
    elif index == '3':
        _name = 'NINO3'; _index_region = '3'
        _out = _out.sel(lat=slice(-5,5),lon=slice(360-150,360-90))
    elif index == '3.4':
        _name = 'NINO3.4'; _index_region = '3.4'
        _out = _out.sel(lat=slice(-5,5),lon=slice(360-170,360-120))
        rolling = rolling or 5
    elif index == '4':
        _name = 'NINO4'; _index_region = '4'
        _out = _out.sel(lat=slice(-5,5),lon=slice(160,360-150))
    _out = field_avg(_out)
    if unforced is not None:
        _out = _out - field_avg(data.sel(lat=_lat).sel(lat=slice(unforced[0], unforced[1])))
    _out = _out.rolling(time=rolling, center=False, min_periods=rolling).mean(dim='time', keep_attrs=True).dropna('time')
    if normalise: ### inclure ici la période de référence ?
        ### méthode à sauver quelque part !
        _out = (_out.groupby('time.month') - _out.groupby('time.month').mean()).groupby('time.month') / _out.groupby('time.month').std()
        _out = _out.drop('month')
        ###
        _out.attrs['units'] = ''
    if detrend:
        _out = _out - get_trend(_out)
    _out.name = _name
    _out.attrs['long_name'] = 'El Niño Sourthern Oscillation SST '+index+' Index'
    _out.attrs['dataset'] = 'NOAA Extended Reconstructed SST V5'
    _out.attrs['var_desc'] = 'Sea Surface Temperature'
    _out.attrs['index_region'] = _index_region
    _out.attrs['rolling'] = rolling
    if starting_month is not None:
        _out = _out.groupby('time.month')[starting_month]
        _out.attrs['starting_month'] = starting_month
    _out.attrs['actual_range'] = [float(_out.min()), float(_out.max())]
    return _out


def tni(data:xarray.DataArray):
    _lat = data.lat.values; _lat.sort()
    _data = data.sel(lat=_lat)
    _out = enso(_data, index='4') - enso(_data, index='1+2')
    _out.attrs['long_name'] = 'Trans-Niño Index'
    return _out