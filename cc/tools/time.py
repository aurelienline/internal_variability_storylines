from .. import cmip6

import numpy
import pandas
import xarray

season2m_dict = {'DJ':1,'JF':2,'FM':3,'MA':4,'AM':5,'MJ':6,'JJ':7,'JA':8,'AS':9,'SO':10,'ON':11,'ND':12}
season3m_dict = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
season4m_dict = {'ONDJ':1,'NDJF':2,'DJFM':3,'JFMA':4,'FMAM':5,'MAMJ':6,'AMJJ':7,'MJJA':8,'JJAS':9,'JASO':10,'ASON':11,'SOND':12}
season5m_dict = {'NDJFM':1,'DJFMA':2,'JFMAM':3,'FMAMJ':4,'MAMJJ':5,'AMJJA':6,'MJJAS':7,'JJASO':8,'JASON':9,'ASOND':10,'SONDJ':11,'ONDJF':12}
season6m_dict = {'ASONDJ':1,'SONDJF':2,'ONDJFM':3,'NDJFMA':4,'DJFMAM':5,'JFMAMJ':6,'FMAMJJ':7,'MAMJJA':8,'AMJJAS':9,'MJJASO':10,'JJASON':11,'JASOND':12}

def get_season(data, season):
    if season == 'yr':
        return data.resample(time='1Y').mean(dim='time', keep_attrs = True, skipna=False)
    elif len(season) <= 3 and season[-1] == 'm':
        return data.rolling(time = int(season[:-1]), center = True).mean(dim='time', keep_attrs = True, skipna=False)
    elif len(season) <= 3 and season[0] == 'm':
        return data.groupby('time.month')[int(season[1:])]
    elif season in season2m_dict.keys():
        _seasons = data.rolling(time = 2, center = False).mean(dim='time', keep_attrs = True, skipna=False)
        return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season2m_dict[season]))
    elif season in season3m_dict.keys():
        _seasons = data.rolling(time = 3, center = True).mean(dim='time', keep_attrs = True, skipna=False)
        return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season3m_dict[season]))
    elif season in season4m_dict.keys():
        _seasons = data.rolling(time = 4, center = False).mean(dim='time', keep_attrs = True, skipna=False)
        return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season4m_dict[season]))
    elif season in season5m_dict.keys():
        _seasons = data.rolling(time = 5, center = True).mean(dim='time', keep_attrs = True, skipna=False)
        return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season5m_dict[season]))
    elif season in season6m_dict.keys():
        _seasons = data.rolling(time = 6, center = False).mean(dim='time', keep_attrs = True, skipna=False)
        return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season6m_dict[season]))
    elif isinstance(season, int):
        try:
            return data.groupby('time.month')[season]
        except:
            print('Error; Nothing happened')
            return data
    else:
        print('Error; Nothing happened')
        return data


def get_season_OLD(data, season):
    if season == 'yr':
        return data.resample(time='1Y').mean(dim='time', keep_attrs = True, skipna=False)
    if season in season3m_dict.keys():
        ##if season in ['DJF', 'MAM', 'JJA', 'SON']: # !!!
        ##    return data.groupby('time.season').mean('time', keep_attrs = True)
        ##el
        if season in ['JFM', 'FMA', 'AMJ', 'MJJ', 'JAS', 'ASO',
                      'MAM', 'JJA', 'SON']:
            #if season in list(season3m_dict.keys()):
            #          'MAM', 'JJA', 'SON']:
            _months = data.groupby('time.month')
            _years = xarray.concat([data.isel(time=_months.groups[season3m_dict[season]-1]),
                                    data.isel(time=_months.groups[season3m_dict[season]]),
                                    data.isel(time=_months.groups[season3m_dict[season]+1])],
                                   dim = 'time')
            return _years.groupby('time.year').mean('time', keep_attrs = True, skipna=False).rename({'year': 'time'}).assign_coords({'time': data.time.isel(time = _months.groups[season3m_dict[season]])})
        else:
            _seasons = data.rolling(time = 3, center = True).mean('time', skipna=False) #, keep_attrs = True)
            return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season3m_dict[season]))
    if season in season4m_dict.keys():
        ##if season in ['DJF', 'MAM', 'JJA', 'SON']: # !!!
        ##    return data.groupby('time.season').mean('time', keep_attrs = True)
        ##el
        if season in ['JFM','FMA','AMJ','MJJ','JAS','ASO',
                      'MAM', 'JJA', 'SON']:
            #if season in list(season3m_dict.keys()):
            #          'MAM', 'JJA', 'SON']:
            _months = data.groupby('time.month')
            _years = xarray.concat([data.isel(time=_months.groups[season3m_dict[season]-1]),
                                    data.isel(time=_months.groups[season3m_dict[season]]),
                                    data.isel(time=_months.groups[season3m_dict[season]+1])],
                                   dim = 'time')
            return _years.groupby('time.year').mean('time', keep_attrs = True, skipna=False).rename({'year': 'time'}).assign_coords({'time': data.time.isel(time = _months.groups[season3m_dict[season]])})
        else:
            _seasons = data.rolling(time = 4, center = True).mean('time', skipna=False) #, keep_attrs = True)
            return _seasons.sel(time = numpy.in1d(_seasons['time.month'], season4m_dict[season]))

def get_anomalies(data:xarray.DataArray, n_max=None, new_dim = 'member', calcul:str='internal variability',
                  start_year:int=None, ref_start_year:int=None, period_length:int=20, period_gap:int=5,
                  relative:bool=False, lag:int=0, out_print=False):
    if calcul == 'internal variability':
        out = get_iv_anomalies(data, n_max=n_max, new_dim = new_dim, start_year=start_year, period_length=period_length, period_gap=period_gap, relative=relative, lag=lag, out_print=out_print)
    elif calcul == 'super internal variability':
        out = get_iv_anomalies(data, n_max=n_max, new_dim = new_dim, start_year=start_year, period_length=period_length, period_gap=period_gap, relative=relative, lag=lag, apply_mean=False, out_print=out_print)
    elif calcul == 'forcing':
        out = get_fc_anomalies(data=data, n_max=n_max, new_dim = new_dim, start_year=start_year, ref_start_year=ref_start_year, period_length=period_length, relative=relative, lag=lag, out_print=out_print)
    return out

def get_fc_anomalies(data:xarray.DataArray, n_max=None, new_dim = 'member',
                     start_year:int=None, ref_start_year:int=None, period_length:int=20,
                     relative:bool=False, lag:int=0, out_print=False):

    _start_year = int(int(start_year) - data.time.dt.year[0])

    ref_start_year = ref_start_year or start_year
    _ref_start_year = int(int(ref_start_year) - data.time.dt.year[0])

    _i = 0
    _keep_computing = True

    _tmp_list = list()

    while _keep_computing:
        ref_str = _ref_start_year
        ref_end = ref_str + period_length
        int_str = _start_year
        int_end = int_str + period_length
        try:
            _units = data.attrs['units']
        except:
            _units = ''
        if _start_year < lag:
            _tmp = xarray.full_like(data.isel(time = 0), numpy.nan).drop('time')
        else:
            _tmp = data.isel(time = slice(int_str - lag, int_end - lag)).mean('time', skipna=False)
            if len(data.dims) > 1:
                _forcing = cmip6.ensemble_estimated_forcing(data.isel(time = slice(ref_str - lag, ref_end - lag))).mean('time', skipna=False)
                _tmp = _tmp - _forcing
                if relative and (
                    data.name[:2] == 'pr'
                    or data.name[:3] == 'snc'
                    or data.name[:3] == 'gnb'
                    or data.name in cmip6.relative_variables
                ):
                    _tmp = _tmp / _forcing * 100.
                    _units = '%'
        _i += 1
        _start_year += 1
        _new_dim = str(data.coords[new_dim].values)+'-length'+str(period_length)+'yr-'+str(_i)
        if n_max != 1:
            _tmp = _tmp.assign_coords({'period_id': _i, new_dim: _new_dim})
        if n_max is not None:
            if _i >= n_max:
                _keep_computing = False
        if _start_year + period_length > len(data.time):
            _keep_computing = False
        _tmp_list.append(_tmp)
        if out_print:
            print(data.time.dt.year[ref_str].values, data.time.dt.year[ref_end-1].values, '-', data.time.dt.year[int_str].values, data.time.dt.year[int_end-1].values)

    out = xarray.concat(_tmp_list, dim=new_dim)
    out.name = out.name # 'delta_'+
    out.attrs = data.attrs
    out.attrs['units'] = _units

    return out

def get_iv_anomalies(data:xarray.DataArray, n_max=None, new_dim = 'member',
                     start_year:int=None, period_length:int=20, period_gap:int=5, apply_mean=True,
                     relative:bool=False, lag:int=0, out_print=False):

    start_year = start_year or data.time.dt.year[0]
    _start_year = int(int(start_year) - data.time.dt.year[0])
    #print(start_year)
    #print(_start_year)
    #print(data.time.dt.year[_start_year])

    _i = 0
    _keep_computing = True

    _tmp_list = list()

    while _keep_computing:
        ref_str = _start_year
        ref_end = ref_str + period_length
        int_str = ref_end + period_gap
        int_end = int_str + period_length
        try:
            _units = data.attrs['units']
        except:
            _units = ''
        if _start_year < lag:
            _tmp = xarray.full_like(data.isel(time = slice(int_str - lag, int_end - lag)), numpy.nan)
        else:
            _tmp = data.isel(time = slice(int_str - lag, int_end - lag))\
                 - data.isel(time = slice(ref_str - lag, ref_end - lag)).mean('time', skipna=False)
            if relative and (
                data.name[:2] == 'pr'
                or data.name[:3] == 'snc'
                or data.name[:3] == 'gnb'
                or data.name in cmip6.relative_variables
            ):
                _tmp = _tmp / data.isel(time = slice(ref_str - lag, ref_end - lag)).mean('time', skipna=False) * 100.
                _units = '%'
        _tmp = _tmp.rename({'time': 'outcome'}).assign_coords({'outcome': numpy.arange(1, len(_tmp.time)+1)})
        if apply_mean:
            _tmp = _tmp.mean('outcome') #.drop('outcome')
        _i += 1
        _start_year += 1
        if n_max != 1:
            _new_dim = str(data.coords[new_dim].values)+'-length'+str(period_length)+'yr-gap'+str(period_gap)+'yr-'+str(_i)
            _tmp = _tmp.assign_coords({'period_id': _i, new_dim: _new_dim})
        if n_max is not None:
            if _i >= n_max:
                _keep_computing = False
        if _start_year + 2 * period_length + period_gap > len(data.time):
            _keep_computing = False
        _tmp_list.append(_tmp)
        if out_print:
            print(data.time.dt.year[ref_str].values, data.time.dt.year[ref_end-1].values, '-', data.time.dt.year[int_str].values, data.time.dt.year[int_end-1].values)

    out = xarray.concat(_tmp_list, dim=new_dim)
    out.name = out.name # 'delta_'+out.name
    out.attrs = data.attrs
    out.attrs['units'] = _units

    return out

def year_to_date(year):
    time_list = list()
    for _date in year:
        _year = int(_date)
        _month = int((_date - _year)*12)
        if _month == 1:
            _len_month = 28
        elif _month in [0, 2 , 4, 6, 7, 9, 11]:
            _len_month = 31
        else:
            _len_month = 30
        _day = int((_date - _year - _month/12)*12*_len_month)
        _hour = int((_date - _year - _month/12 - _day/_len_month/12)*12*_len_month*24)
        _hour = 12 if _hour == 11 else _hour
        if _hour == 23:
            _hour = 0
            _day += 1
        #print(_year, _month+1, _day, _hour)
        time_list.append(pandas.to_datetime(str(_year)+'-'+str(_month+1)+'-'+str(_day)+' '+str(_hour)+':00:00'))

    return pandas.to_datetime(time_list)





