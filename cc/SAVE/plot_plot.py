import matplotlib.colors
import matplotlib.cm
import numpy

def get_cmap(min = -1., max = 1., neg = 'Blues', pos = 'YlOrRd'):
    viridis = matplotlib.cm.get_cmap('viridis', 256)
    newcolors = viridis(numpy.linspace(0, 1, 256))
    if min < 0. and max > 0.:
        vrange = max - min
        zero = int(-min / vrange * 256)
        negcolors = matplotlib.cm.get_cmap(neg, zero)
        poscolors = matplotlib.cm.get_cmap(pos, 256 - zero)
        newcolors[:zero, :] = negcolors(numpy.linspace(1, 0, zero))
        newcolors[zero:, :] = poscolors(numpy.linspace(0, 1, 256 - zero))
        return matplotlib.colors.ListedColormap(newcolors)
    elif min >= 0. and max > min:
        return matplotlib.cm.get_cmap(pos)
    elif max <= 0. and max > min:
        return matplotlib.cm.get_cmap(neg)
    else:
        print('ERROR in get_cmap().')