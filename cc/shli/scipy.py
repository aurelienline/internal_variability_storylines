import collections
import numpy
import pandas
import scipy.signal
import scipy.special
import scipy.stats
import xarray

from ..shli.numpy import *
from .. import util


ftest = collections.namedtuple('ftest', ['statistic', 'pvalue'])
def ftest_ind(a, b, alternative='two-sided', weights=None):
    '''
    Calculate the F-test for the variances of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical variance (expected) values.

    (inspirated from scipy.stats.ttest_ind)

    Parameters
    ----------
    a, b: array_like
        The arrays must have the same shape, except in the dimension
        corresponding to 'axis' (the first, by default).
    (axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, 'a', and 'b'.)
    alternative: {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    weights: list of array_like or None

    Returns
    -------
    statistic : float or array
        The calculated f-statistic.
    pvalue : float or array
        The two-tailed p-value.

    Notes
    -----
    if alternative == 'two-sided':
        Null hypothesis is: 'Sample variances are significantly different.'
    elif alternative == 'less':
        Null hypothesis is: 'First sample variance is significantly less than second sample variance.'
    elif alternative == 'greater':
        Null hypothesis is: 'First sample variance is significantly greater than second sample variance.'

    if ftest_ind(a, b).pvalue > alpha:
        Null hypothesis is rejected.

    '''
    A = numpy.array(a); B = numpy.array(b)
    if weights is None:
        #w_A = None; w_B = None
        A = A - numpy.mean(A, axis=0); B = B - numpy.mean(B, axis=0)
        statistic = numpy.var(A, axis=0) / numpy.var(B, axis=0)
    else:
        w_A = weights[0]; w_B = weights[1]
        A = A - mean(A, weights=w_A); B = B - mean(B, weights=w_B)
        statistic = var(A, weights=w_A) / var(B, weights=w_B)
    if alternative == 'two-sided':
        pvalue = numpy.where(statistic > 1., 2. - 2. * scipy.stats.f.cdf(statistic, A.size-1, B.size-1), 2. * scipy.stats.f.cdf(statistic, A.size-1, B.size-1))
        #if statistic > 1.:
        #    pvalue = 2. - 2. * scipy.stats.f.cdf(statistic, A.size, B.size)
        #else :
        #    pvalue = 2. * scipy.stats.f.cdf(statistic, A.size, B.size)
    elif alternative == 'less':
        pvalue = scipy.stats.f.cdf(statistic, A.size, B.size)
    elif alternative == 'greater':
        pvalue = 1. - scipy.stats.f.cdf(statistic, A.size, B.size)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    return ftest(statistic, pvalue)


ttest = collections.namedtuple('ttest', ['statistic', 'pvalue'])
def ttest_ind(a, b, alternative='two-sided', equal_var = None, weights=None): #, dim=None
    '''
    Inspired from scipy.stats.ttest_ind
    '''
    ### get axis_ind from dim
    # permutations
    A = numpy.array(a); B = numpy.array(b)
    if weights is None:
        #w_A = None; w_B = None
        _m_A = numpy.mean(A, axis=0); _m_B = numpy.mean(B, axis=0)
        _v_A = numpy.var(A, axis=0); _v_B = numpy.var(B, axis=0)
    else:
        w_A = weights[0]; w_B = weights[1]
        _m_A = mean(A, weights=w_A); _m_B = mean(B, weights=w_B)
        _v_A = var(A, weights=w_A); _v_B = var(B, weights=w_B)
    _n_A = A.size; _n_B = B.size
    _var_test = _v_A / _v_B
    if equal_var is None :
        equal_var = True if _var_test > .5 and _var_test < 2. else False
    if equal_var:
        _df = _n_A + _n_B - 2
        statistic = (_m_A - _m_B) * numpy.sqrt(_df) / numpy.sqrt(1. / _n_A + 1. / _n_B) / numpy.sqrt((_n_A - 1) * _v_A + (_n_B - 1) * _v_B)
    else:
        #print('Un-equal variances.')
        _vn_A = _v_A / _n_A; _vn_B = _v_B / _n_B
        with numpy.errstate(divide='ignore', invalid='ignore'):
            _df = (_vn_A + _vn_B) ** 2. / (_vn_A ** 2. / (_n_A - 1) + _vn_B ** 2. / (_n_B - 1))
        if numpy.isnan(_df):
            _df = 1.
        statistic = (_m_A - _m_B) / numpy.sqrt(_vn_A + _vn_B)
    if alternative == 'less':
        pvalue = scipy.special.stdtr(_df, statistic)
    elif alternative == 'greater':
        pvalue = scipy.special.stdtr(_df, - statistic)
    elif alternative == 'two-sided':
        pvalue = scipy.special.stdtr(_df, - numpy.abs(statistic)) * 2.
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    return ttest(statistic, pvalue)


pearsonr = collections.namedtuple('pearson', ['statistic', 'pvalue'])
def pearson(a, b, weights=None):
    '''
    Weighted correlation
    '''

    n = len(a)
    if n != len(b):
        raise ValueError('a and b must have the same length.')

    if weights is not None:
        statistic = cov(numpy.array([a, b]).T, weights = weights)[1, 0] / (std(a, weights=weights) * std(b, weights=weights))
        statistic = min(max(statistic, -1.), 1.)
        if statistic not in (-1., 1.):
            #t = statistic * numpy.sqrt((n - 2.) / (1. - statistic ** 2.))
            #pvalue = 2 * scipy.stats.t.cdf(t, df = n - 2)
            dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc = -1, scale = 2)
            pvalue = 2. * dist.cdf(-abs(statistic))
        else:
            pvalue = 0.
    else:
        statistic, pvalue = scipy.stats.pearsonr(a, b)
    ### caution with pvalue, might need some work (see ftest_ind)
    return pearsonr(statistic, pvalue)


def get_trend(data, dim='time', groupby='time.month', **kwargs):
    '''
    Computes the linear trend on a given dimension.
    '''
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')

    _trend = []
    for _da in _das:
        if dim in _da.dims and dim not in _da.name:
            _tmp_da = _da.transpose(..., dim)
            if dim == 'time' and groupby is not None:
                _da_trend = xarray.concat(
                    [titi[1] - xarray.DataArray(name = titi[1].name,
                        data = scipy.signal.detrend(titi[1], **kwargs),
                        dims = titi[1].dims,
                        coords = titi[1].coords) for titi in _tmp_da.groupby(groupby)], dim='time')
            else:
                _da_trend = _tmp_da - xarray.DataArray(name = _tmp_da.name,
                    data = scipy.signal.detrend(_tmp_da, **kwargs),
                    dims = _tmp_da.dims,
                    coords = _tmp_da.coords)
            for _dim in _da.dims:
                _da_trend = _da_trend.transpose(..., _dim)
            _da_trend = _da_trend.reindex_like(_da)
            _da_trend.attrs = _da.attrs
        _trend.append(_da_trend)

    if _to_ds:
        out = xarray.merge(_trend)
        out.attrs = data.attrs
    else:
        out = _trend[0]

    return out


def detrend(data, dim='time', groupby='time.month', **kwargs):
    '''
    Removes the linear trend on a given dimension.
    
    time.dayofyear?
    '''

    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')

    _dtrnd = []
    for _da in _das:
        if _da.name is None:
            _da.name = ''
        if dim in _da.dims and dim not in _da.name:
            _tmp_da = _da.transpose(..., dim)
            if dim == 'time' and groupby is not None:
                _da_detrended = xarray.concat(
                    [titi[1].mean(dim = dim) + xarray.DataArray(name = titi[1].name,
                        data = scipy.signal.detrend(titi[1], **kwargs),
                        dims = titi[1].dims,
                        coords = titi[1].coords) for titi in _tmp_da.groupby(groupby)], dim='time')
            else:
                _da_detrended = _tmp_da.mean(dim = dim) + xarray.DataArray(name = _tmp_da.name,
                    data = scipy.signal.detrend(_tmp_da, **kwargs),
                    dims = _tmp_da.dims,
                    coords = _tmp_da.coords)
            for _dim in _da.dims:
                _da_detrended = _da_detrended.transpose(..., _dim)
            _da_detrended = _da_detrended.reindex_like(_da)
            _da_detrended.attrs = _da.attrs
        _dtrnd.append(_da_detrended)

    if _to_ds:
        out = xarray.merge(_dtrnd)
        out.attrs = data.attrs
    else:
        out = _dtrnd[0]

    return out


def get_trend_TIME(data, **kwargs):
    '''
    Computes the linear trend.
    '''
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')

    _trend = []
    for _da in _das:
        if 'time' in _da.dims and 'time' not in _da.name:
            _tmp_da = _da.transpose(..., 'time')
            _tmp = list()
            for i, titi in enumerate(_tmp_da.groupby('time.month')):
                _tmp.append(titi[1] - xarray.DataArray(name = titi[1].name,
                    data = scipy.signal.detrend(titi[1], **kwargs),
                    dims = titi[1].dims,
                    coords = titi[1].coords))
            _da_trend = xarray.concat(_tmp, 'time')
            for _dim in _da.dims:
                _da_trend = _da_trend.transpose(..., _dim)
            _da_trend = _da_trend.reindex_like(_da)
            _da_trend.attrs = _da.attrs
        _trend.append(_da_trend)

    if _to_ds:
        out = xarray.merge(_trend)
        out.attrs = data.attrs
    else:
        out = _trend[0]

    return out


def detrend_TIME(data, groupby='time.month', **kwargs):
    '''
    Removes the linear trend.
    
    time.dayofyear?
    '''

    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')

    _dtrnd = []
    for _da in _das:
        if _da.name is None:
            _da.name = ''
        if 'time' in _da.dims and 'time' not in _da.name:
            _tmp_da = _da.transpose(..., 'time')
            _detrended = list()
            for i, titi in enumerate(_tmp_da.groupby(groupby)):
                _detrended.append(titi[1].mean(dim = 'time') + xarray.DataArray(name = titi[1].name,
                    data = scipy.signal.detrend(titi[1], **kwargs),
                    dims = titi[1].dims,
                    coords = titi[1].coords))
            _da_detrended = xarray.concat(_detrended, 'time')
            for _dim in _da.dims:
                _da_detrended = _da_detrended.transpose(..., _dim)
            _da_detrended = _da_detrended.reindex_like(_da)
            _da_detrended.attrs = _da.attrs
        _dtrnd.append(_da_detrended)

    if _to_ds:
        out = xarray.merge(_dtrnd)
        out.attrs = data.attrs
    else:
        out = _dtrnd[0]

    return out

def pearsonr_OLD(A, B, methods=['raw'], print_cor=False):
    # check A and B
    if isinstance(methods, str):
        methods = [methods]
    if 'all' in methods:
        methods = ['raw', 'normalized', 'detrended']
    _out = []
    for _method in methods:
        if _method == 'raw':
            _A = A; _B = B;
        elif _method == 'normalized':
            _A = (A - A.mean()) / A.std(); _B = (B - B.mean()) / B.std();
        elif _method == 'detrended':
            _A = scipy.signal.detrend(A); _B = scipy.signal.detrend(B);
        _out.append(scipy.stats.pearsonr(_A, _B)[0])
        if print_cor:
            print('Correlation ('+_method+'): '+'{0:.3f}'.format(_out[-1]))
    return xarray.DataArray(_out, dims='method', coords={'method':methods})


# ftest = collections.namedtuple('ftest', ['statistics', 'pvalue'])
def ftest_ind_OLD(a, b, alternative='two-sided'):
    '''
    Calculate the F-test for the variances of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical variance (expected) values.

    Parameters
    ----------
    a, b: array_like
        The arrays must have the same shape, except in the dimension
        corresponding to 'axis' (the first, by default).
    (axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, 'a', and 'b'.)
    alternative: {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    Returns
    -------
    statistic : float or array
        The calculated f-statistic.
    pvalue : float or array
        The two-tailed p-value.

    Notes
    -----
    if alternative == 'two-sided':
        Null hypothesis is: 'Sample variances are significantly different.'
    elif alternative == 'less':
        Null hypothesis is: 'First sample variance is significantly less than second sample variance.'
    elif alternative == 'greater':
        Null hypothesis is: 'First sample variance is significantly greater than second sample variance.'

    if ftest_ind(a, b).pvalue > alpha:
        Null hypothesis is rejected.

    '''
    A = numpy.array(a); B = numpy.array(b)
    A = A - A.mean(); B = B - B.mean()
    statistics = numpy.var(A)/numpy.var(B)
    if alternative == 'two-sided':
        if statistics > 1.:
            pvalue = 2. - 2. * scipy.stats.f.cdf(statistics, A.size, B.size)
        else :
            pvalue = 2. * scipy.stats.f.cdf(statistics, A.size, B.size)
    elif alternative == 'less':
        pvalue = scipy.stats.f.cdf(statistics, A.size, B.size)
    elif alternative == 'greater':
        pvalue = 1. - scipy.stats.f.cdf(statistics, A.size, B.size)
    return ftest(statistics, pvalue)


def ftest_ind_OLD2(a, b):
    A = numpy.array(a); B = numpy.array(b)
    statistics = numpy.var(A, ddof=1)/numpy.var(B, ddof=1)
    pvalue = 1. - scipy.stats.f.cdf(statistics, A.size-1, B.size-1)
    #pvalue = 2. * scipy.stats.f.cdf(1. / statistics if statistics > 1. else statistics, A.size-1, B.size-1)
    #pvalue = 2. * (1. - scipy.stats.f.cdf(statistics, A.size-1, B.size-1))
    #pvalue = 1. - 2. * scipy.stats.f.cdf(statistics, A.size-1, B.size-1)
    #pvalue = 2. * scipy.stats.f.cdf(statistics, A.size-1, B.size-1)
    #pvalue = scipy.stats.f.sf(statistics, A.size-1, B.size-1)
    #pvalue = scipy.stats.f.cdf(statistics, A.size-1, B.size-1)
    return ftest(statistics, pvalue)

