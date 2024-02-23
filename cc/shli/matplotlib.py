import cycler
import matplotlib
import matplotlib.colors
import matplotlib.cm
import numpy

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# colors from 'https://personal.sron.nl/~pault/'

colors_dict = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'],
    'high-contrast': ['#004488', '#DDAA33', '#BB5566'],
    'vibrant': ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB'],
    'muted': ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD'],
    'medium-contrast': ['#6699CC', '#004488', '#EECC66', '#994455', '#997700', '#EE99AA'],
    'pale': ['#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC', '#DDDDDD'],
    'dark': ['#222255', '#225555', '#225522', '#666633', '#663333', '#555555'],
    'light': ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD'],
}


cmap_dict = {
    'sunset': ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF', '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D', '#A50026'],
    'BuRd': ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'],
    'PRGn': ['#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7', '#D9F0D3', '#ACD39E', '#5AAE61', '#1B7837'],
    'YlOrBr': ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29', '#EC7014', '#CC4C02', '#993404', '#662506'],
    'iridescent': ['#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF', '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1', '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD', '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388', '#805770', '#684957', '#46353A'],
    'incandescent': ['#CEFFFF', '#C6F7D6', '#A2F49B', '#BBE453', '#D5CE04', '#E7B503', '#F19903', '#F6790B', '#F94902', '#E40515', '#A80003'],
    'discrete rainbow': ['#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4', '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141', '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518', '#DC050C', '#A5170E', '#72190E', '#42150A'],
    'smooth rainbow': ['#E8ECFB', '#DDD8EF', '#D1C1E1', '#C3A8D1', '#B58FC2', '#A778B4', '#9B62A7', '#8C4E99', '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6', '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190', '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541', '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D', '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17', '#521A13'],
}

def set_colors(name):
    try:
        colors = colors_dict[name]
    except:
        colors = colors_dict['default']
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', colors)


def set_cmap(name, segmented=True, reverse=False):
    try:
        _cmap = cmap_dict[name]
        if reverse:
            _cmap.reverse()
        cmap = LinearSegmentedColormap.from_list(name, _cmap) if segmented else ListedColormap(_cmap)
    except:
        cmap = 'viridis'
    matplotlib.rcParams['image.cmap'] = cmap

def cmap(name, segmented=True, reverse=False):
    try:
        _cmap = cmap_dict[name]
        if reverse:
            _cmap.reverse()
        cmap = LinearSegmentedColormap.from_list(name, _cmap) if segmented else ListedColormap(_cmap)
    except:
        cmap = 'viridis'
    return cmap

def get_gradient_color_map(name, c1, c2, N):
    return LinearSegmentedColormap.from_list(name, [c1, c2], N)

def get_diverging_cmap(vmin = -1., vmax = 1., origin = 0., neg = 'Blues', pos = 'YlOrBr'):
    viridis = matplotlib.cm.get_cmap('viridis', 256)
    newcolors = viridis(numpy.linspace(0, 1, 256))
    if vmin < origin and vmax > origin:
        vrange = vmax - vmin
        zero = int((origin-vmin) / vrange * 256)
        negcolors = matplotlib.cm.get_cmap(neg, zero)
        poscolors = matplotlib.cm.get_cmap(pos, 256 - zero)
        newcolors[:zero, :] = negcolors(numpy.linspace(1, 0, zero))
        newcolors[zero:, :] = poscolors(numpy.linspace(0, 1, 256 - zero))
        return matplotlib.colors.ListedColormap(newcolors)
    elif vmin >= origin and vmax > vmin:
        zero = int((vmin-origin) / (vmax-origin) * 256)
        poscolors = matplotlib.cm.get_cmap(pos, 256-zero)
        newcolors = poscolors(numpy.linspace((vmin-origin)/(vmax-origin), 1, 256-zero))
        return matplotlib.colors.ListedColormap(newcolors)
        #return matplotlib.cm.get_cmap(pos)
    elif vmax <= origin and vmax > vmin:
        zero = int((vmax-vmin) / (origin-vmin) * 256)
        negcolors = matplotlib.cm.get_cmap(neg, zero)
        newcolors = negcolors(numpy.linspace(1, (vmax-vmin) / (origin-vmin), zero))
        return matplotlib.colors.ListedColormap(newcolors)
        #return matplotlib.cm.get_cmap(neg+'_r')
    else:
        print('ERROR in get_cmap().')





