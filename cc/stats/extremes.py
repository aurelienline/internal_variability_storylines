import scipy

def compare_extreme_events(X, Y, p, weights_x = None, weights_y = None, distrib_fit = None, relative = False):
    _X_w = X.weighted(weights_x) if weights_x is not None else X
    _Y_w = Y.weighted(weights_y) if weights_y is not None else Y
    if distrib_fit in ['norm', 'normal']:
        _reference_intensity = scipy.stats.norm.ppf(p, loc=_X_w.mean(), scale=_X_w.std())
        _new_p = scipy.stats.norm.cdf(_reference_intensity, loc=_Y_w.mean(), scale=_Y_w.std())
        _delta_intensity = scipy.stats.norm.ppf(p, loc=_Y_w.mean(), scale=_Y_w.std()) - _reference_intensity
        if relative:
            _delta_intensity = _delta_intensity / _reference_intensity
    elif distrib_fit in ['skewnorm', 'skew', 'skew-norm', 'skew-normal']:
        _reference_intensity = scipy.stats.skewnorm.ppf(
            p, ((X ** 3.).weighted(weights_x).mean() - 3. * _X_w.mean() * _X_w.std() ** 2. - _X_w.mean() ** 3. ) / (_X_w.std() ** 3.),
            loc=_X_w.mean(), scale=_X_w.std())
        _new_p = scipy.stats.skewnorm.cdf(
                _reference_intensity,
                ((Y ** 3.).weighted(weights_y).mean() - 3. * _Y_w.mean() * _Y_w.std() ** 2. - _Y_w.mean() ** 3. ) / (_Y_w.std() ** 3.),
                loc=_Y_w.mean(), scale=_Y_w.std())
        _delta_intensity = scipy.stats.skewnorm.ppf(p,
                ((Y ** 3.).weighted(weights_y).mean() - 3. * _Y_w.mean() * _Y_w.std() ** 2. - _Y_w.mean() ** 3. ) / (_Y_w.std() ** 3.),
                loc=_Y_w.mean(), scale=_Y_w.std()) - _reference_intensity
        if relative:
            _delta_intensity = _delta_intensity / _reference_intensity
    elif distrib_fit in ('gev', 'GEV'):
        _cX, _locX, _scaleX = scipy.stats.genextreme.fit(_X_w)
        _cY, _locY, _scaleY = scipy.stats.genextreme.fit(_Y_w)
        _reference_intensity = scipy.stats.genextreme.ppf(p, _cX, loc=_locX, scale=_scaleX)
        _new_p = scipy.stats.genextreme.cdf(_reference_intensity, _cY, loc=_locY, scale=_scaleY)
        _delta_intensity = scipy.stats.genextreme.ppf(p, _cY, loc=_locY, scale=_scaleY) - _reference_intensity
        if relative:
            _delta_intensity = _delta_intensity / _reference_intensity
    else:
        _reference_intensity = float(_X_w.quantile(p))
        _index = int(abs(Y.sortby(Y)-_reference_intensity).argmin().values)
        if _index < 0.001*len(Y):
            _new_p = 0. if _reference_intensity < Y.min().values else 1. / len(Y)
        elif _index > 0.999*len(Y):
            _new_p = 1. if _reference_intensity > Y.max().values else 1. - 1. / len(Y)
        else:
            if weights_y is not None:
                weight_y = weight_y.sortby(Y)
                _new_p = weight_y[0:_index].sum() / weight_y.sum()
            else:
                _new_p = _index / len(Y)
        _delta_intensity = float(_Y_w.quantile(p)) - _reference_intensity
        if relative:
            _delta_intensity = _delta_intensity / _reference_intensity
    return _new_p, _delta_intensity


def probability_ratio_percentile(X, Y, p, weights_x = None, weights_y = None, distrib_fit = None):
    _X_w = X.weighted(weights_x) if weights_x is not None else X
    _Y_w = Y.weighted(weights_y) if weights_y is not None else Y
    _p = list()
    if distrib_fit in ['norm', 'normal']:
        _reference_intensity = scipy.stats.norm.ppf(p, loc=_X_w.mean(), scale=_X_w.std())
        for _Y in (Y, Y - _Y_w.mean() + _X_w.mean()):
            _p.append(scipy.stats.norm.cdf(_reference_intensity, loc=_Y.mean(), scale=_Y.std()))
    elif distrib_fit in ['skewnorm', 'skew', 'skew-norm', 'skew-normal']:
        _reference_intensity = scipy.stats.skewnorm.ppf(
            p, ((X ** 3.).weighted(weights_x).mean() - 3. * _X_w.mean() * _X_w.std() ** 2. - _X_w.mean() ** 3. ) / (_X_w.std() ** 3.),
            loc=_X_w.mean(), scale=_X_w.std())
        for _Y in (Y, Y - _Y_w.mean() + _X_w.mean()):
            _Y = _Y.weighted(weights_y) if weights_y is not None else _Y
            _p.append(scipy.stats.skewnorm.cdf(
                _reference_intensity,
                ((_Y ** 3.).mean() - 3. * _Y.mean() * _Y.std() ** 2. - _Y.mean() ** 3. ) / (_Y.std() ** 3.),
                loc=_Y.mean(), scale=_Y.std()))
    elif distrib_fit in ('gev', 'GEV'):
        _cX, _locX, _scaleX = scipy.stats.genextreme.fit(_X_w)
        _reference_intensity = scipy.stats.genextreme.ppf(p, _cX, loc=_locX, scale=_scaleX)
        for _Y in (Y, Y - _Y_w.mean() + _X_w.mean()):
            _Y = _Y.weighted(weights_y) if weights_y is not None else _Y
            _cY, _locY, _scaleY = scipy.stats.genextreme.fit(_Y)
            _p.append(scipy.stats.genextreme.cdf(_reference_intensity, _cY, loc=_locY, scale=_scaleY))
    else:
        _reference_intensity = float(_X_w.quantile(p))
        for _Y in (Y.sortby(Y), (Y - _Y_w.mean() + _X_w.mean()).sortby(Y)):
            _Y = _Y.weighted(weights_y) if weights_y is not None else _Y
            _index = int(abs(_Y-_reference_intensity).argmin().values)
            if _index < 0.001*len(_Y):
                if _reference_intensity < _Y.min().values:
                    _p.append(0.)
                else:
                    _p.append(1. / len(_Y))
            elif _index > 0.999*len(_Y):
                if _reference_intensity > _Y.max().values:
                    _p.append(1.)
                else:
                    _p.append(1. - 1. / len(_Y))
            else:
                if weights_y is not None:
                    weight_y = weight_y.sortby(_Y)
                    _p.append(weight_y[0:_index].sum() / weight_y.sum())
                else:
                    _p.append(_index / len(_Y))
    _pr = (_p[0]/p, (_p[0]-_p[1]+p)/p, _p[1]/p)
    _r = ((_pr[1]-1)/(_pr[0]-1), (_pr[2]-1)/(_pr[0]-1))
    
    return _pr, _r