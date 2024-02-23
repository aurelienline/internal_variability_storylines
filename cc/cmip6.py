from . import util

import gcsfs
import pandas
import xarray


experiment_dict = {'historical': {'name': 'Historical', 'color': 'gray'},
                   'piControl': {'name': 'Pre-industrical Control', 'color': 'black'},
                   'ssp119': {'name': 'SSP1-1.9', 'color': 'green'},
                   'ssp126': {'name': 'SSP1-2.6', 'color': 'blue'},
                   'ssp245': {'name': 'SSP2-4.5', 'color': 'darkgoldenrod'},
                   'ssp370': {'name': 'SSP3-7.0', 'color': 'red'},
                   'ssp434': {'name': 'SSP4.3-4', 'color': 'purple'},
                   'ssp585': {'name': 'SSP5-8.5', 'color': 'brown'}}


experiment_dict['hist-ssp119'] = experiment_dict['ssp119']
experiment_dict['hist-ssp126'] = experiment_dict['ssp126']
experiment_dict['hist-ssp245'] = experiment_dict['ssp245']
experiment_dict['hist-ssp370'] = experiment_dict['ssp370']
experiment_dict['hist-ssp434'] = experiment_dict['ssp434']
experiment_dict['hist-ssp585'] = experiment_dict['ssp585']


scenario_mip_dict = {
    'CNRM-CM6-1': {
        'experiment': ['hist-ssp245', 'hist-ssp126', 'hist-ssp370', 'hist-ssp585'],
        'ripf': ['r1i1p1f2',  'r2i1p1f2',  'r3i1p1f2',  'r4i1p1f2', 'r5i1p1f2',
                 'r6i1p1f2',  'r7i1p1f2',  'r8i1p1f2',  'r9i1p1f2', 'r10i1p1f2',
                 'r11i1p1f2', 'r12i1p1f2', 'r13i1p1f2', 'r14i1p1f2', 'r15i1p1f2',
                 'r16i1p1f2', 'r17i1p1f2', 'r18i1p1f2', 'r19i1p1f2', 'r20i1p1f2',
                 'r21i1p1f2', 'r22i1p1f2', 'r23i1p1f2', 'r24i1p1f2', 'r25i1p1f2',
                 'r26i1p1f2', 'r27i1p1f2', 'r28i1p1f2', 'r29i1p1f2', 'r30i1p1f2'],
        'color': '#EE6677',
        'hatch': '**',
        'marker': '*',
        'short_name': 'CNRM',
        #
    },
    'ACCESS-ESM1-5': {
        'experiment': ['hist-ssp245', 'hist-ssp126', 'hist-ssp370', 'hist-ssp585'],
        'ripf': ['r1i1p1f1',  'r2i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',
                 'r6i1p1f1',  'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                 'r11i1p1f1', 'r12i1p1f1', 'r13i1p1f1', 'r14i1p1f1', 'r15i1p1f1',
                 'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1',
                 'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1',
                 'r26i1p1f1', 'r27i1p1f1', 'r28i1p1f1', 'r29i1p1f1', 'r30i1p1f1',
                 'r31i1p1f1', 'r32i1p1f1', 'r33i1p1f1', 'r34i1p1f1', 'r35i1p1f1',
                 'r36i1p1f1', 'r37i1p1f1', 'r38i1p1f1', 'r39i1p1f1', 'r40i1p1f1'],
        'color': '#4477AA',
        'hatch': '++',
        'marker': 'P',
        'short_name': 'ACCESS',
    },
                     #'CanESM5': {
                     #    'experiment': ['hist-ssp245', 'hist-ssp119', 'hist-ssp126', 'hist-ssp370', 'hist-ssp585'],
                     #    'ripf': ['r1i1p1f1',  'r2i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',
                     #             'r6i1p1f1',  'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                     #             'r11i1p1f1', 'r12i1p1f1', 'r13i1p1f1', 'r14i1p1f1', 'r15i1p1f1',
                     #             'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1',
                     #             'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1',]},
                     #             #'r1i1p2f1',  'r2i1p2f1',  'r3i1p2f1',  'r4i1p2f1',  'r5i1p2f1',
                     #             #'r6i1p2f1',  'r7i1p2f1',  'r8i1p2f1',  'r9i1p2f1',  'r10i1p2f1',
                     #             #'r11i1p2f1', 'r12i1p2f1', 'r13i1p2f1', 'r14i1p2f1', 'r15i1p2f1',
                     #             #'r16i1p2f1', 'r17i1p2f1', 'r18i1p2f1', 'r19i1p2f1', 'r20i1p2f1',
                     #             #'r21i1p2f1', 'r22i1p2f1', 'r23i1p2f1', 'r24i1p2f1', 'r25i1p2f1']},
    'CanESM5': {
        'experiment': ['hist-ssp245', 'hist-ssp119', 'hist-ssp126', 'hist-ssp370', 'hist-ssp585'],
        'ripf': ['r1i1p2f1',  'r2i1p2f1',  'r3i1p2f1',  'r4i1p2f1',  'r5i1p2f1',
                 'r6i1p2f1',  'r7i1p2f1',  'r8i1p2f1',  'r9i1p2f1',  'r10i1p2f1',
                 'r11i1p2f1', 'r12i1p2f1', 'r13i1p2f1', 'r14i1p2f1', 'r15i1p2f1',
                 'r16i1p2f1', 'r17i1p2f1', 'r18i1p2f1', 'r19i1p2f1', 'r20i1p2f1',
                 'r21i1p2f1', 'r22i1p2f1', 'r23i1p2f1', 'r24i1p2f1', 'r25i1p2f1'],
        'color': '#228833',
        'hatch': 'XX',
        'marker': 'X',
        'short_name': 'Can',
    },
                     #'GISS-E2-1-G': {
                     #    'experiment': ['hist-ssp245', 'hist-ssp370'],
                     #    'ripf': ['r1i1p1f1',  'r2i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',
                     #             'r6i1p1f1',  'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                     #             'r11i1p1f1', 'r12i1p1f1', 'r13i1p1f1', 'r14i1p1f1', 'r15i1p1f1',
                     #             'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1',
                     #             'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1',
                     #             'r26i1p1f1', 'r27i1p1f1', 'r28i1p1f1']},
    'IPSL-CM6A-LR': {
        'experiment': ['hist-ssp245'],
        'ripf': ['r1i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',  'r6i1p1f1',
                 'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1', 'r11i1p1f1',
                 'r12i1p1f1', 'r13i1p1f1', 'r14i1p1f1', 'r15i1p1f1', 'r16i1p1f1',
                 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1', 'r21i1p1f1',
                 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1', 'r26i1p1f1',
                 'r27i1p1f1', 'r28i1p1f1', 'r29i1p1f1', 'r30i1p1f1', 'r31i1p1f1',
                 'r32i1p1f1'],
        'color': '#AA3377',
        'hatch': '..',
        'marker': 'o',
        'short_name': 'IPSL',
    },
    'MIROC6': {
        'experiment': ['hist-ssp245', 'hist-ssp126', 'hist-ssp585'], # , 'hist-ssp245'
        'ripf': ['r1i1p1f1',  'r2i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',
                 'r6i1p1f1',  'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                 'r11i1p1f1', 'r12i1p1f1', 'r13i1p1f1', 'r14i1p1f1', 'r15i1p1f1',
                 'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1',
                 'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1',
                 'r26i1p1f1', 'r27i1p1f1', 'r28i1p1f1', 'r29i1p1f1', 'r30i1p1f1',
                 'r31i1p1f1', 'r32i1p1f1', 'r33i1p1f1', 'r34i1p1f1', 'r35i1p1f1',
                 'r36i1p1f1', 'r37i1p1f1', 'r38i1p1f1', 'r39i1p1f1', 'r40i1p1f1',
                 'r41i1p1f1', 'r42i1p1f1', 'r43i1p1f1', 'r44i1p1f1', 'r45i1p1f1',
                 'r46i1p1f1', 'r47i1p1f1', 'r48i1p1f1', 'r49i1p1f1', 'r50i1p1f1'],
        'color': '#CCBB44',
        'hatch': '--',
        'marker': 'v',
        'short_name': 'MIROC',
    },
    'MPI-ESM1-2-LR': {
        'experiment': ['hist-ssp245', 'hist-ssp119', 'hist-ssp126', 'hist-ssp370', 'hist-ssp585'],
        'ripf': ['r1i1p1f1',  'r2i1p1f1',  'r3i1p1f1',  'r4i1p1f1',  'r5i1p1f1',
                 #'r6i1p1f1',  'r7i1p1f1',  'r8i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                 'r6i1p1f1',  'r7i1p1f1',  'r9i1p1f1',  'r10i1p1f1',
                 'r11i1p1f1', 'r12i1p1f1', 'r14i1p1f1', 'r15i1p1f1',
                 'r16i1p1f1', 'r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1', 'r20i1p1f1',
                 'r21i1p1f1', 'r22i1p1f1', 'r23i1p1f1', 'r24i1p1f1', 'r25i1p1f1',
                 'r26i1p1f1', 'r27i1p1f1', 'r28i1p1f1', 'r29i1p1f1', 'r30i1p1f1'],
        'color': '#66CCEE',
        'hatch': '//',
        'marker': '^',
        'short_name': 'MPI',
    },
    'UKESM1-0-LL': {
        'experiment': ['hist-ssp126', 'hist-ssp370'], # , 'hist-ssp245'
        'ripf': ['r1i1p1f2',  'r2i1p1f2',  'r3i1p1f2',  'r4i1p1f2', 'r5i1p1f2',
                 'r6i1p1f2',  'r7i1p1f2',  'r8i1p1f2',  'r9i1p1f2', 'r10i1p1f2',
                 'r11i1p1f2', 'r12i1p1f2', 'r16i1p1f2', 'r17i1p1f2', 'r18i1p1f2',
                 'r19i1p1f2'],
        'color': '#BBBBBB',
        'hatch': 'OO',
        'marker': 'D',
        'short_name': 'UK',
    }}


table_dict = {
    'tas':        'Amon',
    'tasmin':     'Amon',
    'tasmax':     'Amon',
    'psl':        'Amon',
    'ts':         'Amon',
    'pr':         'Amon',
    'prsn':       'Amon',
    'sfcWind':    'Amon',
    'rsds':       'Amon',
    'rlds':       'Amon',
    'huss':       'Amon',
    'clt':        'Amon',
    'snc':        'LImon',
    'siconc':     'SImon',
    'tos':        'Omon',
    'evspsblsoi': 'Lmon',
    'od550aer':   'AERmon',
    'msftyz':     'Omon',
}


grid_dict = {
    'tas':        'gr',
    'tasmin':     'gr',
    'tasmax':     'gr',
    'psl':        'gr',
    'ts':         'gr',
    'pr':         'gr',
    'prsn':       'gr',
    'sfcWind':    'gr',
    'rsds':       'gr',
    'rlds':       'gr',
    'huss':       'gr',
    'snc':        'gr',
    'siconc':     'gn',
    'tos':        'gn',
    'evspsblsoi': 'gr',
    'od550aer':   'gr',
    'clt':        'gr',
    'msftyz':     'gn',
}


name_dict = {
    'tas':        'temperature', #'near-surface air temperature',
    'tasmin':     'minimal near-surface air temperature',
    'tasmax':     'maximal near-surface air temperature',
    'psl':        'sea level pressure',
    'ts':         'surface temperature',
    'pr':         'precipitation',
    'prlq':       'rainfall',
    'prsn':       'snowfall',
    'sfcWind':    '10m wind speed', #'wind', #'near-surface wind speed',
    'rsds':       'surface downward solar radiation', #'surface solar irradiance',
    'snc':        'snow cover',
    'siconc':     'sea ice cover',
    'tos':        'sea surface temperature',
    'evspsblsoi': 'water evaporation from soil',
    'gnb':        'G-N-B ice',
    'od550aer':   'Ambient Aerosol Optical Thickness at 550nm',
    'clt':        'cloud cover',
    'rlds':       'surface downwelling longwave radiation',
    'huss':       'specific humidity',
}


time_dict = {
    'yr':    'annual',
    'NDJF':  'winter',
    'DJF':   'winter',
    'JFM':   'winter',
    'JJA':   'summer',
    'JAS':   'summer',
}

relative_variables = ['clt', 'od550aer', 'pr', 'prlq', 'prsn', 'sfcWind', 'siconc', 'snc', 'rsds']

def load_scenario_mip_member(model:str=None, variable:str=None,
                             start:str=None, stop:str=None, period:str=None,
                             experiment:str=None, ripf:str=None, table_id:str=None,
                             grid:str=None,
                             decode_times = True):
    _variable_id = variable or 'tas'
    _table_id = table_id or table_dict[_variable_id]
    _source_id = model or 'CNRM-CM6-1'
    _experiment_id = experiment or scenario_mip_dict[_source_id]['experiment'][0]
    _member_id = ripf or scenario_mip_dict[_source_id]['ripf'][0]

    if _source_id in ('CNRM-CM6-1', 'ACCESS-ESM1-5', 'CanESM5', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-LR', 'UKESM1-0-LL'):
        if _table_id in ['yr', 'JFM', 'JJA']:
            _path = '/data/home/globc/line/scratch/Data/Computed/'
        else:
            _path = '/data/home/globc/line/scratch/Data/Imported/'
        _grid_label = grid or grid_dict[_variable_id]
        if _grid_label == 'gr' and _source_id in ['ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR', 'UKESM1-0-LL']:
            _grid_label = 'gn'
        _period = period
        if _experiment_id == 'piControl':
            _period = '*' #_piControl_period_dict[_source_id]
            #_path = '/data/scratch/globc/dcom/CMIP6/CMIP/*/'+_source_id+'/'+_experiment_id+'/*/'\
            #+_table_id+'/'+_variable_id+'/'+_grid_label+'/latest/'
            decode_times = True #False
        if _experiment_id == 'historical':
            _period = '18500101-20141231' if _table_id == 'day' else '1850-2014' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-201412'
        if _period is None:
            if _source_id == 'CNRM-CM6-1':
                _period = '18500101-20391231' if _table_id == 'day' else '1850-2039' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-203912'
            elif _source_id == 'ACCESS-ESM1-5':
                _period = '18500101-21001231' if _table_id == 'day' else '1850-2100' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-210012'
            elif _source_id == 'CanESM5':
                _period = '18500101-21001231' if _table_id == 'day' else '1850-2100' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-210012'
            elif _source_id == 'IPSL-CM6A-LR':
                _period = '18500101-20591231' if _table_id == 'day' else '1850-2059' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-205912'
            elif _source_id == 'MIROC6':
                if _experiment_id == 'hist-ssp245':
                    _period = '18500101-20391231' if _table_id == 'day' else '1850-2039' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-203912'
                else:
                    _period = '18500101-21001231' if _table_id == 'day' else '1850-2100' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-210012'
            elif _source_id == 'MPI-ESM1-2-LR':
                _period = '18500101-21001231' if _table_id == 'day' else '1850-2100' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-210012'
            elif _source_id == 'UKESM1-0-LL':
                _period = '18500101-21001231' if _table_id == 'day' else '1850-2100' if _table_id in ['yr', 'JFM', 'JJA'] else '185001-210012'
        print(_path+_variable_id+'_'+_table_id+'_'+_source_id\
            +'_'+_experiment_id+'_'+_member_id+'_'+_grid_label+'_'+_period+'.nc')
        _out = xarray.open_mfdataset(
            _path+_variable_id+'_'+_table_id+'_'+_source_id\
            +'_'+_experiment_id+'_'+_member_id+'_'+_grid_label+'_'+_period+'.nc'
        ).load()
        # decode_times=decode_times # .get(_variable_id)
        if 'time' in _out.coords: #_out.data_vars:
            _out = _out.sel(time = slice(start, stop))
    else:
        _gcs = gcsfs.GCSFileSystem(token='anon')
        _df = pandas.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        if _experiment_id == 'piControl':
            _df_pi = _df.query("activity_id == 'CMIP' & source_id == '"+_source_id+"' & experiment_id == 'piControl' & table_id == '"+_table_id+"' & variable_id == '"+_variable_id+"' & member_id == '"+_member_id+"'")
            #_out['time'] = _out.indexes['time'].to_datetimeindex()

        else:
            _member_hist = _member_id[:-1]+'3' if _source_id == 'UKESM1-0-LL' and _member_id[1] in ['5', '6', '7'] else _member_id
            _df_hist = _df.query("activity_id == 'CMIP' & source_id == '"+_source_id+"' & experiment_id == 'historical' & table_id == '"+_table_id+"' & variable_id == '"+_variable_id+"' & member_id == '"+_member_hist+"'")
            _df_ssp = _df.query("activity_id == 'ScenarioMIP' & source_id == '"+_source_id+"' & experiment_id == '"+_experiment_id.split('-')[-1]+"' & table_id == '"+_table_id+"' & variable_id == '"+_variable_id+"' & member_id == '"+_member_id+"'")
            _hist = xarray.open_zarr(_gcs.get_mapper(_df_hist.zstore.values[0]), consolidated=True).get(_variable_id).sel(time = slice(start, '2014')).load()
            _ssp = xarray.open_zarr(_gcs.get_mapper(_df_ssp.zstore.values[0]), consolidated=True).get(_variable_id).sel(time = slice('2015', stop)).load()
            if _source_id in ['CanESM5', 'NorESM2-LM', 'UKESM1-0-LL']:
                try:
                    _hist['time'] = _hist.indexes['time'].to_datetimeindex()
                except:
                    pass
                try:
                    _ssp['time'] = _ssp.indexes['time'].to_datetimeindex()
                except:
                    pass
            _out = xarray.merge([_hist, _ssp])
    if model == 'IPSL-CM6A-LR':
        if variable in ['tas', 'ts']:
            _out.get(variable).attrs['units'] = 'K'
            _out.attrs['grid_label'] = _grid_label
    if variable[:2] == 'pr':
        print('WARNING: Check precipitation units!')
        #print(_out)
        #if _out.get(variable).attrs['units'] in ['kg m-2 s-1', 'kg/m**2/s']:
        #    #_out.get(variable) = _out.get(variable) * 86400.
        #    _out.get(variable).attrs['units'] = 'mm/day'
    return _out.assign_coords({'member':_source_id+'_'+_experiment_id+'_'+_member_id})
            
            
def load_scenario_mip_ensemble(model:str=None, variable:str=None,
                               start:str=None, stop:str=None, period:str=None,
                               experiment:list=None, ripf:list=None, table_id:str=None,
                               grid:str=None,
                               decode_times = True):
    _variable_id = variable or 'tas'
    _table_id = table_id or table_dict[_variable_id]
    _source_id = model or 'CNRM-CM6-1'
    _experiment_id = experiment or scenario_mip_dict[_source_id]['experiment']
    _member_id = ripf or scenario_mip_dict[_source_id]['ripf']

    if isinstance(_experiment_id, str): _experiment_id = [_experiment_id]
    if isinstance(_member_id, str): _member_id = [_member_id]
    
    _tmp = list()
    for e in _experiment_id:
        _loc_mem = [scenario_mip_dict[_source_id]['ripf'][0]] if e == 'piControl' else _member_id
        for r in _loc_mem:
            #print(_variable_id, _source_id, e, r)
            _tmp.append(load_scenario_mip_member(model=_source_id, variable=_variable_id,
                                                 start=start, stop=stop, period=period,
                                                 experiment=e, ripf=r, table_id=_table_id,
                                                 grid=grid, decode_times = decode_times))
    return xarray.concat(_tmp, dim = 'member')


def load_sftlf(model:str=None):
    _source_id = model or 'CNRM-CM6-1'
    if _source_id in ('CIESM', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1', 'E3SM-1-1', 'EC-Earth3', 'EC-Earth3-AerChem', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'IPSL-CM6A-LR', 'KACE-1-0-G'):
        _out = xarray.open_dataset('/data/scratch/globc/line/Data/Masks/sftlf_fx_'+_source_id+'_gr.nc').sftlf
    elif _source_id in ('GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'KIOST-ESM'):
        _out = xarray.open_dataset('/data/scratch/globc/line/Data/Masks/sftlf_fx_'+_source_id+'_gr1.nc').sftlf
    else: #if _source_id in ('ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR', 'UKESM1-0-LL'):
        _out = xarray.open_dataset('/data/scratch/globc/line/Data/Masks/sftlf_fx_'+_source_id+'_gn.nc').sftlf
    #else:
    #    _gcs = gcsfs.GCSFileSystem(token='anon')
    #    _df = pandas.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    #    _out = xarray.open_zarr(_gcs.get_mapper(_df.query("activity_id == 'CMIP' & source_id == '"+_source_id+"' & experiment_id in ('piControl', 'historical') & table_id == 'fx' & variable_id == 'sftlf'").zstore.values[0]), consolidated=True).load().sftlf
    return _out


def sort_by_IV_ensemble(members:list=None):
    _dict = {}
    for _member in members:
        _source, _experiment, _ripf = _member.split('_')
        _config = 'r*i'+_ripf.split('i')[1]
        if _source not in _dict:
            _dict[_source] = {_experiment: {_config: [_member]}}
        elif _experiment not in _dict[_source]:
            _dict[_source][_experiment] = {_config: [_member]}
        elif _config not in _dict[_source][_experiment]:
            _dict[_source][_experiment][_config] = [_member]
        else:
            _dict[_source][_experiment][_config].append(_member)
    return _dict


def get_weight(members, method='model', dim='member'):
    _weight = list()
    _members = list()
    if isinstance(members, xarray.DataArray) or isinstance(members, xarray.Dataset):
        members = members[dim].values
    _dict = sort_by_IV_ensemble(members)
    for _source in _dict:
        for _experiment in _dict[_source]:
            for _config in _dict[_source][_experiment]:
                _members += _dict[_source][_experiment][_config]
                [_weight.append(
                    1. / (
                          len(_dict[_source])
                        * len(_dict[_source][_experiment])
                        * len(_dict[_source][_experiment][_config])
                    )
                ) for _ in range(len(_dict[_source][_experiment][_config]))]
    out = xarray.DataArray(
        _weight,
        dims = [dim],
        coords = {dim: _members}
    )
    return out


def get_members_dict(members:list=None):
    '''
    Depreciated: misses the config dimension.
    '''
    _dict = {}
    for _member in members:
        _source, _experiment, _ = _member.split('_')
        if _source not in _dict:
            _dict[_source] = {_experiment: [_member]}
        elif _experiment not in _dict[_source]:
            _dict[_source][_experiment] = [_member]
        else:
            _dict[_source][_experiment].append(_member)
    return _dict


def normalisation(data:xarray, scaler='std', period:tuple=None):
    '''
    Parameters
    ----------
    data: array of either DataArray or Dataset type from the xarray package
    scaler: can either be a string, a float, or an int
     - 'std': divided by the standard deviation
     - float in (0., 1.] : devided by the interquantile range
     - int in (0., 100.] : devided by the interpercentile range
    period: None or tuple (or list)
    
    Returns
    -------
    out: array of same type as data, normalised according to the given period
    '''
    util.check_Xarray(data)
    if isinstance(data, xarray.Dataset):
        _das = [data.get(_var) for _var in data.data_vars]; _to_ds = True;
    elif isinstance(data, xarray.DataArray):
        _das = [data]; _to_ds = False;
    else:
        raise TypeError('Input data type should either be a Dataset or a DataArray from the xarray module.')
    _start = _stop = None
    if isinstance(period, tuple) or isinstance(period, list):
        if len(period) == 2:
            _start, _stop = period
    _dict = sort_by_IV_ensemble(list(data.member.values))
    _out = list()
    for _da in _das:
        if 'member' in _da.dims:
            _dims = ['member']
            if 'time' in _da.dims:
                _dims.append('time')
            _store = list()
            for s in _dict:
                for e in _dict[s]:
                    for c in _dict[s][e]:
                        _tmp = _da.sel(member=_dict[s][e][c])
                        _tmp_ref = _tmp.sel(time=slice(_start, _stop)) if 'time' in _tmp.dims and period is not None else _tmp
                        if isinstance(scaler, int):
                            if scaler > 0 and scaler <= 100:
                                scaler /= 100.
                            else:
                                scaler = 'std'; print('ERROR: if scaler is of type int, it must be in (0, 100].\nStandard deviation used instead.')
                        if isinstance(scaler, float):
                            if scaler > 0. and scaler <= 1.:
                                _scl = _tmp_ref.quantile(1.-(1.-scaler)/2., dim=_dims) - _tmp_ref.quantile((1.-scaler)/2., dim=_dims)
                            else:
                                scaler = 'std'; print('ERROR: if scaler is of type float, it must be in (0., 1.].\nStandard deviation used instead.')
                        elif scaler is None:
                            _scl = 1.
                        else:
                            scaler = 'std'; print('ERROR: scaler not understood.\nStandard deviation used instead.')
                        if scaler == 'std':
                            _scl = _tmp_ref.std(dim=_dims, keep_attrs=True)
                        _tmp = (_tmp - _tmp_ref.mean(dim=_dims, keep_attrs=True)) / _scl
                        try:
                            if _scl.size != 1:
                                _tmp = xarray.where(_scl < 1e-10, 0., _tmp)
                        except:
                            pass
                        _store.append(_tmp)
            _out.append(xarray.concat(_store, dim='member'))
    if _to_ds:
        out = xarray.merge(_out)
        out.attrs = data.attrs
    else:
        out = _out[0]
    return out


def standardisation(data):
    '''
    Old code.
    '''
    _tmp = list(data.member.values); _tmp.sort(); _dict = get_members_dict(_tmp)
    _store = list()
    for _source in _dict:
        for _experiment in _dict[_source]:
            _tmp = data.sel(member=_dict[_source][_experiment])
            _mean = _tmp.mean('member'); _std = _tmp.std('member')
            _tmp = (_tmp - _mean) / _std
            if _std.size != 1:
                _tmp = xarray.where(_std < 1e-10, 0., _tmp)
            _store.append(_tmp)
    out = xarray.concat(_store, dim='member')
    out.attrs = data.attrs
    out.attrs['units'] = ''
    return out


def ensemble_estimated_forcing(data:xarray.DataArray=None):
    '''
    Make it xarray.Dataset fiendly!
    '''
    util.check_DataArray(data)
    _dict = sort_by_IV_ensemble(list(data.member.values))
    #_coords = data.coords
    _tmp = list()
    #_label = list()
    for s in _dict:
        for e in _dict[s]:
            for c in _dict[s][e]:
                _mean = data.sel(member=_dict[s][e][c]).mean(dim='member', keep_attrs=True)
                for l in _dict[s][e][c]:
                    #_label.append(l)
                    _tmp.append(_mean.assign_coords({'member':l}))
                    _tmp[-1].attrs=data.attrs
    #_coords['member'] = _label
    #return xarray.DataArray(data=_tmp, dims=data.dims, coords=_coords, name=data.name, attrs=data.attrs)
    return xarray.concat(_tmp, dim='member')


def remove_forced_response(data:xarray.DataArray=None):
    '''
    Make it xarray.Dataset fiendly!
    '''
    util.check_DataArray(data)
    return data - ensemble_estimated_forcing(data)



def ensemble_std(data:xarray.DataArray=None):
    util.check_DataArray(data)
    _dict = sort_by_IV_ensemble(list(data.member.values))
    _coords = data.coords
    _tmp = list()
    _label = list()
    for s in _dict:
        for e in _dict[s]:
            for c in _dict[s][e]:
                _std = data.sel(member=_dict[s][e][c]).std(dim='member', keep_attrs=True)
                for l in _dict[s][e][c]:
                    _label.append(l)
                    _tmp.append(_std.values)
    _coords['member'] = _label
    return xarray.DataArray(data=_tmp, coords=_coords, dims=data.dims, name=data.name, attrs=data.attrs)


def ensemble_estimated_forcing_OLD(data:xarray.DataArray=None):
    '''
    Depreciated: misses the config dimension.
    '''
    util.check_DataArray(data)
    _dict = get_members_dict(list(data.member.values))
    _coords = data.coords
    _tmp = list()
    _label = list()
    for s in _dict:
        for e in _dict[s]:
            _mean = data.sel(member=_dict[s][e]).mean(dim='member', keep_attrs=True)
            for l in _dict[s][e]:
                _label.append(l)
                _tmp.append(_mean.values)
    _coords['member'] = _label
    return xarray.DataArray(data=_tmp, coords=_coords, dims=data.dims, name=data.name, attrs=data.attrs)


def ensemble_estimated_forcing_OLD2(data:xarray.DataArray=None):
    '''
    SLOW
    '''
    util.check_DataArray(data)
    _dict = get_members_dict(list(data.member.values))
    _tmp = list()
    for s in _dict:
        for e in _dict[s]:
            _tmp.append(xarray.concat([data.sel(member=_dict[s][e]).mean(dim='member', keep_attrs=True) for i in range(len(_dict[s][e]))], dim = pandas.Index(_dict[s][e], name='member')))
    return xarray.merge(_tmp).get(data.name)


def flatten_iv(data):
    if 'outcome' in data.dims:
        _out = list()
        for _member in data.member.values:
            for _outcome in data.outcome.values:
                _out.append(data.sel(member=_member, outcome=_outcome).assign_coords({'member':_member+'y'+str(_outcome)}))
        _out = xarray.concat(_out, dim='member')
        _out.attrs = data.attrs
        return _out
    else:
        return data

region_dict = {
    'FRA': 'France',
    'MED': 'Mediterranean Basin',
    'NEU': 'Northern Europe',
    'WCE': 'Western and Central Europe',
}

piControl_period_dict = {
    'ACCESS-ESM1-5': '10101-110012',
    'CanESM5': '555001-660012', #'520101-620012',
    'CNRM-CM6-1': '185001-284912',
    'IPSL-CM6A-LR': '185001-384912',
    'MIROC6': '320001-399912',
    'MPI-ESM1-2-LR': '185001-284912',
}





