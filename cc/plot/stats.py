from ..shli.numpy import *

import matplotlib.pyplot
import matplotlib.patches
import numpy

def boxplot(data, ax = None,
            weights=None,
            label = None,
            yTitle = None,
            bar = 'median', box = .5, ext = .1,
            alpha = 1,
            color = None,
            edgecolor = None,
            hatch = None,
            outliers = True,
            extrema = False,
            markersize = 3,
            dx=0, width=0.5,
            rotation=0,
            orientation='vertical'):
    """
    Create a boxplot of the dispersion of *data*.

    Parameters
    ----------

    data : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    bar : 'mean' or 'median' or 'both'
    ...
    """

    if not isinstance(data, list):
        data = [data]

    _ax = ax or matplotlib.pyplot.gca()
    _size = numpy.shape(data)

    _color = color or ['black' for i in range (0, _size[0])]
    _edgecolor = edgecolor or ['black' for i in range (0, _size[0])]
    _hatch = hatch or [None for i in range (0, _size[0])]
    _label = label or ['' for i in range (0, _size[0])]
    for i in range(0, _size[0]):
        _tmp = numpy.array(data[i])
        _weights = weights[i] if weights is not None else weights
        if len(numpy.shape(_tmp)) > 1:
            _tmp = numpy.concatenate(data[i])
        if len(numpy.shape(_weights)) > 1:
            _weights = numpy.concatenate(_weights[i]) if weights is not None else None
        if len(_tmp) != 1:
            # compute statistics
            _median = median(_tmp, weights=_weights)
            _mean = mean(_tmp, weights=_weights)
            _boxlow = quantile(_tmp, box/2, weights=_weights)
            _boxhigh = quantile(_tmp, 1 - box/2, weights=_weights)
            _low = quantile(_tmp, ext/2, weights=_weights)
            _high = quantile(_tmp, 1 - ext/2, weights=_weights)
            _minima = numpy.min(_tmp)
            _maxima = numpy.max(_tmp)
            # plot
            _fill = True if _hatch[i] is None else None
            _x_data = [
                (dx + i, dx + i),
                (dx + i, dx + i),
                (dx + i - width / 4., dx + i + width / 4.),
                (dx + i - width / 4., dx + i + width / 4.),
                (dx + i - width / 2., dx + i + width / 2.),
                (dx + i - width / 2., dx + i + width / 2.),
                i,
                i,
            ]
            _y_data = [
                (_low, _boxlow),
                (_boxhigh, _high),
                (_low, _low),
                (_high, _high),
                (_mean, _mean),
                (_median, _median),
                _minima,
                _maxima,
            ]
            if orientation == 'vertical':
                _X = _x_data; _Y = _y_data;
                _ax.add_patch(matplotlib.patches.Rectangle(
                    (dx + i - width / 2., _boxlow), width, _boxhigh - _boxlow,
                    color = _color[i], edgecolor = _edgecolor[i], hatch = _hatch[i], fill = _fill, alpha = alpha)) # , ec = 'None'
            elif orientation == 'horizontal':
                _X = _y_data; _Y = _x_data;
                _ax.add_patch(matplotlib.patches.Rectangle(
                    (_boxlow, dx + i - width / 2.), _boxhigh - _boxlow, width,
                    color = _color[i], edgecolor = _edgecolor[i], hatch = _hatch[i], fill = _fill, alpha = alpha)) # , ec = 'None'

            _ax.plot(_X[0], _Y[0], 'k', ls = '--', lw = 1.5)
            _ax.plot(_X[1], _Y[1], 'k',ls = '--', lw = 1.5)
            _ax.plot(_X[2], _Y[2], 'k', lw = 1.5)
            _ax.plot(_X[3], _Y[3], 'k', lw = 1.5)
            
            if bar == 'mean':
                _ax.plot(_X[4], _Y[4], color = _color[i], lw = 7)
                _ax.plot(_X[4], _Y[4], color = 'w', lw = 2)
            elif bar == 'median':
                _ax.plot(_X[5], _Y[5], color = _color[i], lw = 7)
                _ax.plot((dx + i - width / 2., dx + i + width / 2.), _Y[5], color = 'w', lw = 2)
            elif bar == 'both':
                _ax.plot(_X[4], _Y[4], color = 'k', lw = 7)
                _ax.plot(_X[5], _Y[5], color = _color[i], lw = 7)
                _ax.plot(_X[5], _Y[5], color = 'w', lw = 2)

            if outliers:
                for j in range(0, len(_tmp)):
                    if (_tmp[j] < _low or _tmp[j] > _high):
                        if orientation == 'vertical':
                            _ax.plot(dx + i, _tmp[j], 'o', c = 'k', markersize = markersize) # size set by input (dict?)
                        elif orientation == 'horizontal':
                            _ax.plot(_tmp[j], dx + i, 'o', c = 'k', markersize = markersize) # size set by input (dict?)
            if extrema:
                _ax.plot(_X[6], _Y[6], 'o', c=_color[i], markersize=markersize)
                _ax.plot(_X[7], _Y[7], 'o', c=_color[i], markersize=markersize)
                #matplotlib.pyplot.text(i, maxima, str(int(tmp.argmax())+1), c = _color[i])

    if orientation == 'vertical':
        _ax.set_xticks(numpy.arange(_size[0]))
        _ax.set_xticklabels(_label, rotation=rotation) # ,weight="bold"
        _ax.set_ylabel(yTitle) # ,weight="bold"
    elif orientation == 'horizontal':
        _ax.set_yticks(numpy.arange(_size[0]))
        _ax.set_yticklabels(_label, rotation=rotation) # ,weight="bold"
        _ax.set_xlabel(yTitle) # ,weight="bold"

    return _ax


def confidence_ellipse(x, y, ax=None, n_std:int=3.0, scale:float=None, edgecolor = 'k', facecolor='none', linewidth=1., linestyle='-', **kwargs):
    """
    FROM: https://matplotlib.org/3.3.3/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    import matplotlib.pyplot
    import numpy

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    ax = ax or matplotlib.pyplot.gca()

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        edgecolor = edgecolor, facecolor=facecolor, linewidth=linewidth, linestyle=linestyle, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = scale if scale is not None else n_std
    mean_x = numpy.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = scale if scale is not None else n_std
    mean_y = numpy.mean(y)

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x * numpy.sqrt(cov[0, 0]), scale_y * numpy.sqrt(cov[1, 1])) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
