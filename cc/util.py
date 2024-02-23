import matplotlib.colors
import matplotlib.cm
import numpy
import xarray


def check_Xarray(data):
    if not isinstance(data, xarray.Dataset) and not isinstance(data, xarray.DataArray):
        raise TypeError('Data type must either be a Dataset or a DataArray from the xarray module.')


def check_DataArray(data):
    if not isinstance(data, xarray.DataArray):
        raise TypeError('Data type must be a DataArray from the xarray module.')


def check_Dataset(data):
    if not isinstance(data, xarray.Dataset):
        raise TypeError('Data type must be a Dataset from the xarray module.')


def check_time_bounds(data:xarray.Dataset):
    return data.drop('time_bounds') if 'time_bounds' in data.data_vars else data


def check_bool_attribute(xarray, attr):
    try:
        out = bool(xarray.attrs[attr])
    except KeyError:
        out = False
    return out


def drop(data, list_to_drop:list):
    check_Xarray(data)
    if not isinstance(list_to_drop, list):
        if isinstance(list_to_drop, str):
            list_to_drop = [list_to_drop]
        else:
            raise TypeError('Second input must be a list.')
    for _drop in list_to_drop:
        try:
            data = data.drop(_drop)
        except:
            pass
    return data


def space_to_new_line(string):
    return string.replace(' ', '\n')


def to_celcius(data): # if no unit?
    _out = data.copy(deep=True)
    if _out.attrs['units'] == 'K':
        _out -= 273.15
        _out.attrs['units'] = 'C'
    return _out


def get_cmap(vmin, vmax, neg = 'Blues', pos = 'YlOrRd'):
    viridis = matplotlib.cm.get_cmap('viridis', 256)
    newcolors = viridis(numpy.linspace(0, 1, 256))
    if vmin < 0. and vmax > 0.:
        vrange = vmax - vmin
        zero = int(-vmin / vrange * 256)
        negcolors = matplotlib.cm.get_cmap(neg, zero)
        poscolors = matplotlib.cm.get_cmap(pos, 256 - zero)
        newcolors[:zero, :] = negcolors(numpy.linspace(1, 0, zero))
        newcolors[zero:, :] = poscolors(numpy.linspace(0, 1, 256 - zero))
        return matplotlib.colors.ListedColormap(newcolors)
    else:
        print('ERROR in get_cmap().')