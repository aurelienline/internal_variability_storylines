from .. import util
import numpy
import xarray
import pandas


def ratio(data, dim=None):
    util.check_Xarray(data)
    out = data.mean(dim=dim) / data.std(dim=dim)
    out.attrs = data.attrs
    return out


def time_running_anomaly_snr(data, dim=None, number_of_months:int=3,
                               reference_start:str='1995', reference_end:str='2014',
                               interest_start:str='2020', interest_end:str='2039'):
    util.check_Xarray(data)
    out = list()
    _rolled_data = data.rolling(time = number_of_months, center = True).mean(dim = 'time')
    if number_of_months == 3:
        _labels = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
    for i in range(1, 13):
        _season = _rolled_data.sel(time = numpy.in1d(_rolled_data['time.month'], i))
        _anomaly = (_season.sel(time = slice(interest_start, interest_end)) - _season.sel(time = slice(reference_start, reference_end)).mean('time')).mean('time')
        out.append(ratio(_anomaly, dim=dim))
    return xarray.concat(out, dim = pandas.Index(_labels, name='time'))
