from ..krepidoma.numpy import *

import numpy
import scipy.stats
import xarray

def anova(data, weight=None, dims=None, noise_dim=None, confidence=.9):
    '''
    Add third dimension calculations.
    '''
    import scipy.stats

    if noise_dim is not None:
        if noise_dim in data.dims:
            data = data.transpose(..., noise_dim)
    else:
        noise_dim = data.dims[-1]
    data = data.sortby(noise_dim)

    _len = len(data.shape)
    _dims = data.dims if dims is None else dims; _dims = list(_dims)
    if _len > 1:
        _conf_factor = scipy.stats.t.ppf(1.-(1.-confidence)/2., len(data.coords[noise_dim]))

        weight = xarray.ones_like(data.coords[_dims[numpy.argmax(_len)]], dtype='float') if weight is None else weight
        weight = weight.sortby(noise_dim)

        _mu = data.weighted(weight).mean(dim=_dims)
        _signal = {'signal': _mu}

        _var_tot = 0.; _F = 0.

        _first = dict(); _var_first = dict()
        for _dim1 in _dims[:-1]:
            _tmp = _dims.copy(); _tmp.remove(_dim1)
            _first[_dim1] = data.weighted(weight).mean(dim=_tmp) - _mu
            _var_first[_dim1] = _first[_dim1].var(dim=(_dim1))#.values
            _var_tot = _var_tot + _var_first[_dim1]
            _F = _F + _var_first[_dim1] ** .5
        _signal = {**_signal, **_first}

        if _len > 2:
            _second = dict(); _var_second = dict()
            for i, _dim1 in enumerate(_dims[:-1]):
                for j, _dim2 in enumerate(_dims[1+i:-1]):
                    _tmp = _dims.copy()
                    for _dim in (_dim1, _dim2):
                        try:
                            _tmp.remove(_dim)
                        except:
                            pass
                    _second[_dim1+'-'+_dim2] = data.weighted(weight).mean(dim=_tmp) + _mu - _first[_dim1] - _first[_dim2]
                    _var_second[_dim1+'-'+_dim2] = _second[_dim1+'-'+_dim2].var(dim=(_dim1, _dim2))#.values
                    _var_tot = _var_tot +  _var_second[_dim1+'-'+_dim2]
                    _F = _F +  _var_second[_dim1+'-'+_dim2] ** .5
            _signal = {**_signal, **_second}

        if _len > 3:
            #print('Add third dimension calculations.')
            _third = dict(); _var_third = dict()
            for i, _dim1 in enumerate(_dims[:-1]):
                for j, _dim2 in enumerate(_dims[1+i:-1]):
                    for k, _dim3 in enumerate(_dims[2+i+j:-1]):
                        if _dim2 != _dim3:
                            _tmp = _dims.copy()
                            for _dim in (_dim1, _dim2, _dim3):
                                try:
                                    _tmp.remove(_dim)
                                except:
                                    pass
                            _third[_dim1+'-'+_dim2+'-'+_dim3] = data.weighted(weight).mean(dim=_tmp) + _mu - _first[_dim1] - _first[_dim2] - _first[_dim3] - _second[_dim1+'-'+_dim2] - _second[_dim1+'-'+_dim3] - _second[_dim2+'-'+_dim3]
                            _var_third[_dim1+'-'+_dim2+'-'+_dim3] = _third[_dim1+'-'+_dim2+'-'+_dim3].var(dim=(_dim1, _dim2, _dim3))#.values
                            _var_tot = _var_tot + _var_third[_dim1+'-'+_dim2+'-'+_dim3]
                            _F = _F +  _var_third[_dim1+'-'+_dim2+'-'+_dim3] ** .5
            _signal = {**_signal, **_third}

        _nu = data - data.weighted(weight).mean(dim=_dims[-1])
        _var_nu = _nu.weighted(weight).var(dim=_dims)#.values
        _var_tot = _var_tot + _var_nu
        _F = _F + _var_nu ** .5
        _std = data.weighted(weight).std(dim=_dims)
        #_noise = {'noise': _conf_factor * _std.values}; _F = _F / _std.values
        _noise = {'noise': _conf_factor * _var_tot ** .5}; _F = _F / _var_tot ** .5
        _var = {'total': _var_tot, 'member': _var_nu}

        _F_nu = {noise_dim: _conf_factor * _var_nu ** .5 / _F}
        _noise = {**_noise, **_F_nu}

        _F_first = dict()
        for _dim1 in _dims[:-1]:
            _F_first[_dim1] = _conf_factor * _var_first[_dim1] ** .5 / _F
        _noise = {**_noise, **_F_first}
        _var = {**_var, **_var_first}

        if _len > 2:
            _F_second = dict()
            for i, _dim1 in enumerate(_dims[:-1]):
                for _dim2 in _dims[1+i:-1]:
                    _F_second[_dim1+'-'+_dim2] = _conf_factor * _var_second[_dim1+'-'+_dim2] ** .5 / _F
            _noise = {**_noise, **_F_second}
            _var = {**_var, **_var_second}

        if _len > 3:
            _F_third = dict()
            for i, _dim1 in enumerate(_dims[:-1]):
                for _dim2 in _dims[1+i:-1]:
                    for _dim3 in _dims[2+i+j:-1]:
                        if _dim2 != _dim2:
                            _F_third[_dim1+'-'+_dim2+'-'+_dim3] = _conf_factor * _var_third[_dim1+'-'+_dim2+'-'+_dim3] ** .5 / _F
            _noise = {**_noise, **_F_third}
            _var = {**_var, **_var_third}

        return _signal, _noise, _var