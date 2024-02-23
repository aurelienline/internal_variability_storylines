from .. import util
from ..tools import field
import gridfill


def open_cmip6(variable, experiment, period):
    pass


def open_obs(variable, dataset, period):
    pass


def reggrid(data, grid,
            method='linear'):
    '''
    Dummy function.
    '''
    util.check_Xarray(data)
    _data = field.check_lat_lon_names(data); _grid = field.check_lat_lon_names(grid);
    return _data.interp(lat = _grid.lat, lon = _grid.lon, method = method)


def reggrid_land_ocean(data, sftlf_ini, sftlf_target, #=None,
                       method='linear',
                       eps=1e-4,relax=0.6,itermax=1e4,initzonal=False,cyclic=False,verbose=False):

    ### check cyclic option!

    _kw = dict(eps=eps, relax=relax, itermax=itermax, initzonal=initzonal, cyclic=cyclic, verbose=verbose)

    _sftlf_target = sftlf_target #or 

    util.check_Xarray(data);
    util.check_DataArray(sftlf_ini); util.check_DataArray(_sftlf_target);
    _data = field.check_lat_lon_names(data)
    _sftlf_ini = field.check_lat_lon_names(_sftlf_ini); _sftlf_target = field.check_lat_lon_names(_sftlf_target);

    ### loop on da/ds?

    _das = []
    for _sub_data in [_data.where(_sftlf > 50.), _data.where(_sftlf < 50.)]:
        _filled, _converged = gridfill.fill(_sub_data.to_masked_array(), _sub_data.dims.index('lat'), _sub_data.dims.index('lon'), **_kw)
        if not all(i for i in _converged):
            print('ERROR: At least one grid filling did not converge!')
        _da_filled = xr.DataArray(_filled, dims = _data.dims, coords = _data.coords, attrs = _data.attrs)
        _das.append(_da_filled.interp(lat = _sftlf_target.lat, lon = _sftlf_target.lon, method = method))

    _da = _das[0].where(_sftlf_target > 50.) * _sftlf_target / 100. + _das[1].where(_sftlf_target < 50.) * _sftlf_target / 100.
    #_da.attrs[''] = #informations on the grid

    return _da


cmip6_dict = {}


dcom_obs_dict = {}