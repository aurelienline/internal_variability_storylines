'''
Storyline analysis method

Data are supposed to have been pre-precessed as disired: scaled, normalized, standardized, etc.
'''

from .shli.numpy import *
from .shli.scipy import ftest_ind
from .plot.stats import boxplot, confidence_ellipse

import matplotlib.pyplot
import numpy
import numpy.linalg
import scipy.stats
import sklearn.linear_model
import xarray

class storylines:
    '''
    Description ###

    Parameters
    ----------

    Examples
    --------

    '''

    def __init__(
        self,
        delta = r'$\Delta$',
        confidence=None, exclusion=0.,
        colors=None,
        significance_limit=.05, permutations=0,
        prints=False
    ):
        #self.data = xarray.Dataset()
        self.drivers = dict(); self.drivers_names = list()
        self.ndrivers = 0
        self.targets = dict()

        self.delta = delta
        self.confidence = confidence
        self.exclusion = exclusion
        self.colors = colors or ['#EE7733', '#EE3377', '#009988', '#33BBEE']
        # from 'https://personal.sron.nl/~pault/';  previously: ['orange', 'deepskyblue', 'purple', 'green']
        self.prints = prints
        self.significance_limit = significance_limit
        self.permutations = permutations
        self.score = xarray.Dataset()
        self.coef = xarray.Dataset()
        self.outputs = xarray.Dataset()
        self.weights = None
        self.reg = dict()

    def add_driver(
        self, data, name = None
    ):
        try: _name = name or data.name
        except KeyError: raise InputError("The new driver needs a name. Specify it with 'name=...' as input or with 'xarray.DataArray.name=...'.")
        if _name in self.drivers:
            raise Error('The new driver can not have the same name as one of the previoulsy added data.')

        self.drivers[_name] = data; self.drivers_names.append(_name)
        self.ndrivers += 1
        self.members = data.member.values
        for _driver in self.drivers:
            self.members = list(set(self.members).intersection(set(self.drivers[_driver].member.values)))
        for _driver in self.drivers:
            self.drivers[_driver] = self.drivers[_driver].sel(member=self.members)
        if self.weights is not None:
            self.weights = self.weights.sel(member=self.members)
        if self.ndrivers == 3 and len(self.colors) < 8:
            #self.colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
            self.colors = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00']
        if self.ndrivers == 4 and len(self.colors) < 16:
            self.colors = ['#125A56', '#00767B', '#238F9D', '#42A7C6', '#60BCE9', '#9DCCEF', '#C6DBED', '#DEE6E7', '#F0E6B2', '#F9D576', '#FFB954', '#FD9A44', '#F57634', '#E94C1F', '#D11807', '#A01813']

    def add_target(
        self, data, name = None
    ):
        try: _name = name or data.name
        except KeyError: raise InputError("The new target needs a name. Specify it with 'name=...' as input or with 'xarray.DataArray.name=...'.")
        if _name in self.targets:
            raise Error('The new target can not have the same name as one of the previoulsy added data.')

        #self.data = self.data.assign({_name: data})
        self.targets[_name] = data
        if self.drivers != dict():
            _reg = sklearn.linear_model.LinearRegression()
            _Y = self.targets[_name]
            self.std_driver = list()
            _weights = None if self.weights is None else self.weights.sel(member=_Y.member)
            
            for _d in self.drivers:
                _tmp = self.drivers[_d].sel(member=_Y.member)
                self.std_driver.append((_tmp - mean(_tmp, weights=_weights)) / std(_tmp, weights=_weights))
            _X = numpy.array(self.std_driver).T
            
            self.reg[_name] = _reg.fit(_X, _Y, _weights)

    def add_weights(
        self, data
    ):

        #if numpy.size(data) == numpy.size(self.data[0]):
            self.weights = data
            ##self.weights = xarray.DataArray(
            ##    numpy.array(data),
            ##    dims=self.data.sel(
            ##)
        #else:
        #    raise Error('The size of the weights array must be the same as the one of the previously loaded data.')

    def build_storylines(
        self,
        exclusion = None,
        significance_limit=None, permutations=None,
        significance_print=False
    ):
        '''
        Ideas and work:
         - Remove x_j driver (or suggest it) if abs(cor(x_i, x_j)) >= .9?
        '''
        exclusion = exclusion or self.exclusion
        _stories = ['']
        _story_labels = ['']
        for _d in self.drivers:
            _tmp = _stories.copy()
            _stories = list()
            for _l in _tmp:
                _stories.append(_l+_d[0].upper())
                _stories.append(_l+_d[0].lower())
            _tmp = _story_labels.copy()
            _story_labels = list()
            for _l in _tmp:
                _story_labels.append(_l+' '+self.delta+'$_+$'+_d)
                _story_labels.append(_l+' '+self.delta+'$_-$'+_d)
        self.stories = _stories
        self.labels = [i[1:] for i in _story_labels]

        if False: #self.ndrivers == 1:
            _driver_avg = mean(self.drivers[self.drivers_names[0]].values, weights=self.weights)
            _families_ind = [list(set(numpy.where(self.drivers[self.drivers_names[0]].values > _driver_avg)[0].tolist())),
                             list(set(numpy.where(self.drivers[self.drivers_names[0]].values < _driver_avg)[0].tolist()))]
        else:
            _drivers = xarray.concat([self.drivers[_driver] for _driver in self.drivers], dim = 'driver').assign_coords({'driver': self.drivers_names})
            #_drivers_std = (_drivers - _drivers.mean('member')) / _drivers.std('member')
            self.mean_dri = xarray.DataArray(
                [mean(_drivers.sel(driver = _dri), weights=self.weights) for _dri in _drivers.driver],
                dims = ['driver'],
                coords = {'driver': _drivers.driver},
            )
            self.std_dri = xarray.DataArray(
                [std(_drivers.sel(driver = _dri), weights=self.weights) for _dri in _drivers.driver],
                dims = ['driver'],
                coords = {'driver': _drivers.driver},
            )
            _drivers_std = (_drivers - self.mean_dri) / self.std_dri
            _story_indeces = numpy.ones([2**self.ndrivers, self.ndrivers])
            for iSto, _sto in enumerate(self.stories):
                for iCha, _char in enumerate(_sto):
                    if _char == _char.lower():
                        _story_indeces[iSto, iCha] = -1
            _da = xarray.DataArray(
                _story_indeces,
                dims = ['story', 'driver'],
                coords = {'story': self.stories,
                          'driver': self.drivers_names})
            if self.ndrivers != 1:
                _var_cov = numpy.corrcoef(_drivers_std.values)
                _inv_var_cov = numpy.linalg.inv(_var_cov)
                (_var_cov)
            _store_indeces = list()
            for iS, _sto in enumerate(_da.story):
                _sys = _da.sel(story=_sto).values @ _inv_var_cov @ _da.sel(story=_sto).values if self.ndrivers != 1 else 1.
                _index = _da.sel(story=_sto) * (1 / _sys) ** (1 / self.ndrivers)
                _store_indeces.append(_index)
            self.outputs = self.outputs.assign({
                'index': xarray.DataArray(
                _store_indeces,
                dims = ['story', 'driver'],
                coords = {'story': self.stories,
                          'driver': self.drivers_names})})
            _confidence_factor = scipy.stats.chi2.ppf(self.confidence, df = self.ndrivers) if self.confidence is not None else 1.
            self.outputs = self.outputs.assign({
                'storindex': self.outputs.index * _confidence_factor ** (1 / self.ndrivers) * self.std_dri + self.mean_dri
            })
            self.families = {}
            self.stats = {}
            for _sto in self.stories:
                self.families[_sto] = {'label': list(), 'index': list()}
            for i, _member in enumerate(self.members):
                _sto = self.which_storyline(_drivers.sel(member = _member), exclusion = exclusion)
                if _sto is not None:
                    self.families[_sto]['label'].append(_member)
                    self.families[_sto]['index'].append(i)

        ### make a panda dataframe with informations instead would be great
        _significance_limit = significance_limit or self.significance_limit
        _permutations = permutations or self.permutations
        if self.prints:
            print()
        
        self.significance = xarray.Dataset()
        self.significance.attrs['T-test'] = 'Averages are significantly different.'
        self.significance.attrs['F-test'] = 'Sample variances are significantly different.'
        for _target in self.targets:
            _members = self.targets[_target].member.values
            _signi_store1 = list()
            for i in range(len(self.families)):
                _story_members = list(set(self.families[self.stories[i]]['label']).intersection(set(_members)))
                _a = self.targets[_target].sel(member=_story_members)
                _b = self.targets[_target].sel(member=_members)
                _signi_store2 = list()
                _signi_store2.append([scipy.stats.ttest_ind(_a, _b, permutations = permutations).pvalue,
                                      ftest_ind(_a, _b).pvalue])
                for j in range(len(self.families)):
                    _story_members = list(set(self.families[self.stories[j]]['label']).intersection(set(_members)))
                    _b = self.targets[_target].sel(member=_story_members)
                    _signi_store2.append([scipy.stats.ttest_ind(_a, _b, permutations = permutations).pvalue,
                                          ftest_ind(_a, _b).pvalue])
                _signi_store1.append(_signi_store2)
            self.significance = self.significance.assign(
                {_target:xarray.DataArray(numpy.array(_signi_store1).transpose(),
                                          dims=['test', 'c', 'r'],
                                          coords={'test': ['T-test', 'F-test'],
                                                  'c':['Mega-ensemble']+self.labels,
                                                  'r':self.labels})})

            '''
            for i in range(len(self.families)):
                self.stats[self.stories[i]][_target] = numpy.mean(self.data.get(_target).sel(member=self.families[self.stories[i]]['label'])).values
                _signi = scipy.stats.ttest_ind(self.data.get(_target).sel(member=self.families[self.stories[i]]['label']),
                                               self.data.get(_target).sel(member=self.data.member.values),
                                               permutations = _permutations).pvalue
                _yes_or_no = [' ', '<'] if _signi < _significance_limit else [' NOT ', '>']
                if self.prints:
                    print('The '+self.labels[i]+' storyline distribution average of "'+_target+'" is'+_yes_or_no[0]+'significantly different from the mega-ensemble distribution average (pvalue = {0:1.3f}'.format(_signi)+' '+_yes_or_no[1]+' {0:1.3f}'.format(_significance_limit)+').')
                for j in range(i+1, len(self.families)):
                    _signi = scipy.stats.ttest_ind(self.data.get(_target).sel(member=self.families[self.stories[i]]['label']),
                                                   self.data.get(_target).sel(member=self.families[self.stories[j]]['label']),
                                                   permutations = _permutations).pvalue
                    _yes_or_no = [' ', '<'] if _signi < _significance_limit else [' NOT ', '>']
                    if self.prints:
                        print('The '+self.labels[i]+' storyline distribution average of "'+_target+'" is'+_yes_or_no[0]+'significantly different from the '+self.labels[j]+' storyline distribution average (pvalue = {0:1.3f}'.format(_signi)+' '+_yes_or_no[1]+' {0:1.3f}'.format(_significance_limit)+').')
                if self.prints:
                    print()
            '''

            self.explained_variance(_target)
            pass
        #self.significance = self.significance.assign(
        #    {'null_hypothesis':xarray.DataArray(numpy.array(['Averages are significantly different.',
        #                                                     'Sample variances are significantly different.']),
        #                                        dims=['test'], coords={'test': ['T-test', 'F-test']})})


    def which_storyline(
        self, data, exclusion = None
    ):
        exclusion = exclusion or self.exclusion
        _refs = self.outputs.index
        #_data = (data - self.outputs.storindex.mean(dim='story')) / self.outputs.storindex.std(dim='story')
        _data = (data - self.mean_dri) / self.std_dri
        _new_basis = numpy.array([(_refs[0] + _refs[i+1])/2. for i in range(self.ndrivers)])
        _refs_nb = xarray.DataArray(
            [numpy.linalg.inv(_new_basis.T) @ _refs[i].values for i in range(len(_refs))],
            dims=_refs.dims, coords = _refs.coords)
        print(numpy.linalg.inv(_new_basis.T))
        print(_data.values)
        _data_nb = xarray.DataArray(
            (numpy.linalg.inv(_new_basis.T) @ _data.values),
            dims=_data.dims, coords = _data.coords)
        if _data_nb.dot(_data_nb) ** .5 < exclusion:
            out = None
        else:
            _distance = ((_refs_nb - _data_nb) ** 2.).sum(dim='driver') ** .5
            _arg = _distance.argmin()
            out = str(self.outputs.storindex.story[_arg].values)
        return out


    def story_predict(
        self,
        target = None,
    ):
        if target is None:
            target = list(self.targets.keys())
        if isinstance(target, str):
            target = [target]
        out = dict()
        for _target in target:
            tmp = list()
            for _sto in self.stories:
                _X = numpy.array([self.outputs.index.sel(story=_sto).values])
                if self.confidence is not None:
                    _X = _X * scipy.stats.chi2.ppf(self.confidence, df = self.ndrivers) ** (1 / self.ndrivers)
                tmp.append(self.reg[_target].predict(_X)[0])
            out[_target] = xarray.DataArray(tmp, dims = ['story'], coords = {'story': self.stories})
        if len(out) == 1:
            out = out[target[0]]
        return out


    #def story_predict(
    #    self,
    #    target = None,
    #):
    #    if target is None:
    #        target = list(self.targets.keys())
    #    if isinstance(target, str):
    #        target = [target]
    #    out = dict()
    #    for _target in target:
    #
    #        tmp = list()
    #        for _sto in self.stories:
    #            _X = self.outputs.storindex.sel(story=_sto)
    #            tmp.append(self.reg[_target].predict(_X))
    #        out[_target] = xarray.DataArray(numpy.array(_tmp.T), dims = ['story'],coords = {'story': self.stories})
    #    if len(out) == 1:
    #        out = out[target[0]]
    #    return out



    def predict(
        self,
        target:str=None,
        drivers=None,
        weights=None,
    ):
        if weights is None:
            _weights = self.weights
        target = target or self.targets[0]
        _reg = sklearn.linear_model.LinearRegression()
        _Y = self.targets[target]
        self.std_driver = list()
        for _d in self.drivers:
            _tmp = self.drivers[_d].sel(member=_Y.member)
            self.std_driver.append((_tmp - mean(_tmp, weights=self.weights)) / std(_tmp, weights=self.weights))
        _X = numpy.array(self.std_driver).T
        _self_weights = self.weights.sel(member=_Y.member) if self.weights is not None else None
        _reg.fit(_X, _Y, _self_weights)
        _std_indriver = list()
        if drivers is None:
            _x = _X
        else:
            for _d in drivers.data_vars:
                _tmp = drivers.get(_d)
                _std_indriver.append((_tmp - mean(_tmp, weights=weights)) / std(_tmp, weights=weights))
            _x = numpy.array(_std_indriver).T
        out = _reg.predict(_x)
        return out


    def get_score(
        self,
        target=None,
        drivers=None,
        weights=None,
    ):
        out = list()
        _reg = sklearn.linear_model.LinearRegression()
        self.std_driver = list()
        _std_indriver = list()
        if drivers is None:
            drivers = self.drivers
        for _d in drivers.data_vars:
            _tmp = self.drivers[_d]; _tmp = _tmp.sortby(_tmp.member)
            self.std_driver.append((_tmp - mean(_tmp, weights=self.weights)) / std(_tmp, weights=self.weights))
            _tmp = drivers.get(_d); _tmp = _tmp.sortby(_tmp.member)
            _std_indriver.append((_tmp - mean(_tmp, weights=weights)) / std(_tmp, weights=weights))
        _X = numpy.array(self.std_driver).T
        _x = numpy.array(_std_indriver).T
        if target is None:
            target = self.targets
        elif isinstance(target, str):
            target = [target]
            #target = self.data.get(target)
        for _target in target:
            _Y = self.targets[_target]; _Y = _Y.sortby(_Y.member)
            _self_weights = self.weights.sortby(_Y.member) if self.weights is not None else None
            _reg.fit(_X, _Y, _self_weights)
            _y = target.get(_target); _y = _y.sortby(_y.member)
            _weights = weights.sortby(_y.member) if weights is not None else None
            out.append(_reg.score(_x, _y, _weights))
        if len(out) == 1:
            out = out[0]
        return out


    def explained_variance(
        self,
        target = None,
        weights = None,
        store = True,
        prints = None,
    ):

        if isinstance(target, str):
            targets = self.targets[target].to_dataset()
        elif isinstance(target, xarray.DataArray):
            targets = target.to_dataset()
        elif isinstance(target, xarray.Dataset):
            targets = target
        elif target is None:
            targets = self.targets
        if weights is None:
            weights = self.weights
        if prints is None or not isinstance(prints, bool):
            prints = self.prints
        _reg = sklearn.linear_model.LinearRegression()
        out = list()
        for _target in targets:
            _Y = targets.get(_target)
            weights = weights.sel(member=_Y.member)
            self.std_driver = list()
            for _d in self.drivers:
                _tmp = self.drivers[_d].sel(member=_Y.member)
                self.std_driver.append((_tmp - mean(_tmp, weights=weights)) / std(_tmp, weights=weights))

            _X = numpy.array(self.std_driver).T
            _reg.fit(_X, _Y, weights)
            if store:
                self.score = self.score.assign({_target: _reg.score(_X, _Y, weights)})
                self.coef = self.coef.assign({_target: _reg.coef_})
            if prints:
                _text = [self.delta + _target +' =', '{0:.1f}'.format(_reg.intercept_)]
                for _id, _d in enumerate(self.drivers):
                    _text.append('+ {0:.1f} * '.format(_reg.coef_[_id])+r'$\tilde{\Delta}$'+_d)
                _text.append('({0:.0f}'.format(_reg.score(_X, _Y, weights)*100)+'% of variance explained)')
                print(' '.join(_text))
            out.append(_reg.score(_X, _Y, weights))
        if len(out) == 1:
            out = out[0]
        return out


    def spatial_response(
        self
    ):
        pass


    def boxplot(
        self, target=None, bar = 'mean', box = 25, ext=5, outliers = True, extrema = False, figsize=(10,10), ylabels=None,
    ):
        target = target or self.targets
        if isinstance(target, str):
            target = [target]
        for _target in target:
            fig, ax = matplotlib.pyplot.subplots(figsize = figsize)
            boxplot(
                [self.targets[_target].sel(member=self.families[self.stories[i]]['label']) for i in range(len(self.stories))],
                weights = [self.weights.sel(member=self.families[self.stories[i]]['label']) for i in range(len(self.stories))],
                color = self.colors,
                label = [_sto.replace(' ', '\n') for _sto in self.labels], #.replace('\Delta', '\\tilde{\Delta}')
                bar = bar, box = box, ext = ext, outliers = outliers, extrema = extrema)
            ax.axhline(y=0,color='k',lw=.5)
            if ylabels in ('std', 'standardised', 'standardized'):
                ax.set_yticks([-2, -1, 0, 1, 2])
                ax.set_yticklabels(['$\mu-2\sigma$', '$\mu-\sigma$', '$\mu$', '$\mu+\sigma$', '$\mu+2\sigma$'])
            #matplotlib.pyplot.show()
        pass


    def plot_quadrants(self, ax = None, marker = '.', markersize=12, colors = None, legend = None, alpha=.25, show_stories=True, show_members=True, members_value=None):
        _ax = ax or matplotlib.pyplot.gca()
        _colors = colors or self.colors
        if self.ndrivers == 1:
            pass
        elif self.ndrivers == 2:
            #_centers = numpy.empty((2,2,2), dtype = float)
            #_centers[0,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[2,0]) / 2.,
            #                 (self.outputs.storindex[3,0] + self.outputs.storindex[1,0]) / 2.]
            #_centers[0,1] = [(self.outputs.storindex[0,1] + self.outputs.storindex[2,1]) / 2.,
            #                 (self.outputs.storindex[3,1] + self.outputs.storindex[1,1]) / 2.]
            #_centers[1,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[1,0]) / 2.,
            #                 (self.outputs.storindex[3,0] + self.outputs.storindex[2,0]) / 2.]
            #_centers[1,1] = [(self.outputs.storindex[0,1] + self.outputs.storindex[1,1]) / 2.,
            #                 (self.outputs.storindex[3,1] + self.outputs.storindex[2,1]) / 2.]
            #self.Xreg, self.Xinter, _, _, _ = scipy.stats.linregress(_centers[0,1], _centers[0,0])
            #self.Yreg, self.Yinter, _, _, _ = scipy.stats.linregress(_centers[1,0], _centers[1,1])
            for i in range(2**self.ndrivers):
                if show_stories:
                    _ax.plot(self.outputs.storindex[i,0], self.outputs.storindex[i,1], 'o', c = _colors[i], markersize = 20)
                if show_members:
                    _ax.plot(self.drivers[self.drivers_names[0]].sel(member=self.families[self.stories[i]]['label']),
                             self.drivers[self.drivers_names[1]].sel(member=self.families[self.stories[i]]['label']),
                             marker = marker, c = 'k', ls='', markersize=markersize)
                    # option to show value instead (from target variable)

            _xMin, _xMax = _ax.get_xlim(); _ax.set_xlim(_xMin, _xMax);
            _yMin, _yMax = _ax.get_ylim(); _ax.set_ylim(_yMin, _yMax);

            _centers = numpy.empty((2,2,2), dtype = float)
            _centers[0,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[2,0]) / 2.,
                             (self.outputs.storindex[3,0] + self.outputs.storindex[1,0]) / 2.]
            _centers[0,1] = [(self.outputs.storindex[0,1] + self.outputs.storindex[2,1]) / 2.,
                             (self.outputs.storindex[3,1] + self.outputs.storindex[1,1]) / 2.]
            _centers[1,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[1,0]) / 2.,
                             (self.outputs.storindex[3,0] + self.outputs.storindex[2,0]) / 2.]
            _centers[1,1] = [(self.outputs.storindex[0,1] + self.outputs.storindex[1,1]) / 2.,
                             (self.outputs.storindex[3,1] + self.outputs.storindex[2,1]) / 2.]
            if numpy.amax(_centers[0,0:1]) != numpy.amin(_centers[0,0:1]):
                self.Xreg, self.Xinter, _, _, _ = scipy.stats.linregress(_centers[0,1], _centers[0,0])
            else:
                self.Xreg, self.Xinter = 0, _centers[0,1,0]
            _ax.plot([self.Xinter + self.Xreg * _yMin, self.Xinter + self.Xreg * _yMax], [_yMin, _yMax],
                     c = 'k', lw = .5, zorder = 2)
            if numpy.amax(_centers[1,0:1]) != numpy.amin(_centers[1,0:1]):
                self.Yreg, self.Yinter, _, _, _ = scipy.stats.linregress(_centers[1,0], _centers[1,1])
            else:
                self.Yreg, self.Yinter = 0, _centers[1,0,1]
            _ax.plot([_xMin, _xMax], [self.Yinter + self.Yreg * _xMin, self.Yinter + self.Yreg * _xMax],
                     c = 'k', lw = .5, zorder = 2)
            if self.confidence is not None:
                confidence_ellipse(
                    self.outputs.storindex.sel(driver = self.drivers_names[0]),
                    self.outputs.storindex.sel(driver = self.drivers_names[1]),
                    scale = .975*(scipy.stats.chi2.ppf(self.confidence, df = self.ndrivers) / 2.) ** (1 / self.ndrivers),
                    ax = _ax, edgecolor = 'k', linewidth = .5, linestyle = '--')
            if self.exclusion is not None:
                confidence_ellipse(
                    self.outputs.storindex.sel(driver = self.drivers_names[0]),
                    self.outputs.storindex.sel(driver = self.drivers_names[1]),
                    scale = .975*self.exclusion, #/ numpy.sqrt(self.outputs.index[:self.ndrivers].dot(self.outputs.index[:self.ndrivers])),
                    ax = _ax, facecolor = 'w', edgecolor = 'k', linewidth = .5, linestyle = '-', zorder=3)
            _legend = []

            _top_lft = [_xMin, _yMax]
            _top_mdl = [self.Xinter + self.Xreg * _yMax, _yMax]
            _top_rgt = [_xMax, _yMax]
            _mdl_rgt = [_xMax, self.Yinter + self.Yreg * _xMax]
            _btm_rgt = [_xMax, _yMin]
            _btm_mdl = [self.Xinter + self.Xreg * _yMin, _yMin]
            _btm_lft = [_xMin, _yMin]
            _mdl_lft = [_xMin, self.Yinter + self.Yreg * _xMin]
            _mdl_mdl = [mean(self.drivers[self.drivers_names[0]], weights=self.weights),
                        mean(self.drivers[self.drivers_names[1]], weights=self.weights)]

            _ax.add_patch(matplotlib.pyplot.Polygon([_mdl_mdl, _mdl_rgt, _top_rgt, _top_mdl],
                                                    color = _colors[0], fill = True, alpha = alpha, ec = 'None'))
            #_legend.append(mpatches.Patch(color = storyColors[0], label = story[0])) #, alpha = polyAlpha

            _ax.add_patch(matplotlib.pyplot.Polygon([_mdl_mdl, _mdl_lft, _btm_lft, _btm_mdl],
                                                    color = _colors[3], fill = True, alpha = alpha, ec = 'None'))
            _ax.add_patch(matplotlib.pyplot.Polygon([_mdl_mdl, _btm_mdl, _btm_rgt, _mdl_rgt],
                                                    color = _colors[1], fill = True, alpha = alpha, ec = 'None'))
            _ax.add_patch(matplotlib.pyplot.Polygon([_mdl_mdl, _top_mdl, _top_lft, _mdl_lft],
                                                    color = _colors[2], fill = True, alpha = alpha, ec = 'None'))


"""
    def build_families( # build storylines?
        self,
        significance_limit=None, permutations=None,
        significance_print=False
    ):

        self._get_storylines_indices()

        if self.ndrivers == 1:
            self.stories = [self.drivers[0].upper(), self.drivers[0].lower()]
            self.labels = [self.delta+'$_+$'+self.drivers[0], self.delta+'$_-$'+self.drivers[0]]
            #self.story_members = numpy.where((numpy.sqrt(X[0]**2) > numpy.sqrt(self.exclusion)) and (anoDvrPrdSclGE > anoDvrPrdScGE.mean()))
            _driver_avg = mean(self.data.get(self.drivers[0]).values, weights=self.weights)
            _families_ind = [list(set(numpy.where(self.data.get(self.drivers[0]).values > _driver_avg)[0].tolist())),
                             list(set(numpy.where(self.data.get(self.drivers[0]).values < _driver_avg)[0].tolist()))]
        elif self.ndrivers == 2:
            self.stories = [self.drivers[0][0].upper()+self.drivers[1][0].upper(),
                            self.drivers[0][0].lower()+self.drivers[1][0].lower(),
                            self.drivers[0][0].upper()+self.drivers[1][0].lower(),
                            self.drivers[0][0].lower()+self.drivers[1][0].upper()]
            self.labels = [self.delta+'$_+$'+self.drivers[0]+' '+self.delta+'$_+$'+self.drivers[1],
                           self.delta+'$_-$'+self.drivers[0]+' '+self.delta+'$_-$'+self.drivers[1],
                           self.delta+'$_+$'+self.drivers[0]+' '+self.delta+'$_-$'+self.drivers[1],
                           self.delta+'$_-$'+self.drivers[0]+' '+self.delta+'$_+$'+self.drivers[1]]
            _centers = numpy.empty((2,2,2), dtype = float)
            _centers[0,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[0,3]) / 2.,
                             (self.outputs.storindex[0,1] + self.outputs.storindex[0,2]) / 2.]
            _centers[0,1] = [(self.outputs.storindex[1,0] + self.outputs.storindex[1,3]) / 2.,
                             (self.outputs.storindex[1,1] + self.outputs.storindex[1,2]) / 2.]
            _centers[1,0] = [(self.outputs.storindex[0,0] + self.outputs.storindex[0,2]) / 2.,
                             (self.outputs.storindex[0,1] + self.outputs.storindex[0,3]) / 2.]
            _centers[1,1] = [(self.outputs.storindex[1,0] + self.outputs.storindex[1,2]) / 2.,
                             (self.outputs.storindex[1,1] + self.outputs.storindex[1,3]) / 2.]
            self.Xreg, self.Xinter, _, _, _ = scipy.stats.linregress(_centers[0,1], _centers[0,0])
            self.Yreg, self.Yinter, _, _, _ = scipy.stats.linregress(_centers[1,0], _centers[1,1])
            if self.exclusion > 0.:
                self.inner = numpy.where(numpy.sqrt(  self.data.get(self.drivers[0]).values ** 2.
                                                    - self.drivers_cor * 2.
                                                      * self.data.get(self.d[0]).values
                                                      * self.data.get(self.drivers[1]).values
                                                    + self.data.get(self.drivers[1]).values ** 2.)
                                         > numpy.sqrt((1. - self.drivers_cor ** 2) * numpy.sqrt(self.exclusion)))
            else:
                self.inner = (numpy.arange(len(self.data.get(self.drivers[0]).values)), )
            _cond_X_g = numpy.where(self.data.get(self.drivers[0]).values
                                    > self.Xreg * self.data.get(self.drivers[1]).values + self.Xinter)
            _cond_X_l = numpy.where(self.data.get(self.drivers[0]).values
                                    < self.Xreg * self.data.get(self.drivers[1]).values + self.Xinter)
            _cond_Y_g = numpy.where(self.data.get(self.drivers[1]).values
                                    > self.Yreg * self.data.get(self.drivers[0]).values + self.Yinter)
            _cond_Y_l = numpy.where(self.data.get(self.drivers[1]).values
                                    < self.Yreg * self.data.get(self.drivers[0]).values + self.Yinter)
            _families_ind = [list(set(self.inner[0].tolist()) & set(_cond_X_g[0].tolist()) & set(_cond_Y_g[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_l[0].tolist()) & set(_cond_Y_l[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_g[0].tolist()) & set(_cond_Y_l[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_l[0].tolist()) & set(_cond_Y_g[0].tolist()))]

        else:
            raise Error('The storylines method only has been coded for 1 or 2 drivers yet. Feel free to add the calculation for 3 drivers or more. :)')

        self.families = {}
        self.stats = {}
        for i in range(len(_families_ind)):
            self.families[self.stories[i]] = {'label': list(self.data.member[_families_ind[i]].values),
                                              'index': _families_ind[i]}
            self.stats[self.stories[i]] = {}

        ### make a panda dataframe with informations instead would be great
        _significance_limit = significance_limit or self.significance_limit
        _permutations = permutations or self.permutations
        if self.prints:
            print()
        for _target in self.targets:
            for i in range(len(_families_ind)):
                self.stats[self.stories[i]][_target] = mean(
                    self.data.get(_target).sel(member=self.families[self.stories[i]]['label']),
                    weights = self.weights.sel(member=self.families[self.stories[i]]['label'])
                ).values
                _signi = scipy.stats.ttest_ind(self.data.get(_target).sel(member=self.families[self.stories[i]]['label']),
                                               self.data.get(_target).sel(member=self.data.member.values),
                                               permutations = _permutations).pvalue
                _yes_or_no = [' ', '<'] if _signi < _significance_limit else [' NOT ', '>']
                if self.prints:
                    print('The '+self.labels[i]+' storyline distribution is'+_yes_or_no[0]+'significantly different from the mega-ensemble distribution (pvalue = {0:1.3f}'.format(_signi)+' '+_yes_or_no[1]+' {0:1.3f}'.format(_significance_limit)+').')
                for j in range(i+1, len(_families_ind)):
                    _signi = scipy.stats.ttest_ind(self.data.get(_target).sel(member=self.families[self.stories[i]]['label']),
                                                   self.data.get(_target).sel(member=self.families[self.stories[j]]['label']),
                                                   permutations = _permutations).pvalue
                    _yes_or_no = [' ', '<'] if _signi < _significance_limit else [' NOT ', '>']
                    if self.prints:
                        print('The '+self.labels[i]+' storyline distribution is'+_yes_or_no[0]+'significantly different from the '+self.labels[j]+' storyline distribution (pvalue = {0:1.3f}'.format(_signi)+' '+_yes_or_no[1]+' {0:1.3f}'.format(_significance_limit)+').')
                if self.prints:
                    print()

            self.explained_variance(_target)

        pass


    def _get_storylines_indices(
        self
    ):
        _iniindex = numpy.sqrt(scipy.stats.chi2.ppf(self.confidence, df = self.ndrivers) / 2.) if self.confidence is not None else 1.
        if self.ndrivers == 1:
            _tmp = self.data.get(self.drivers[0]).values
            _index = _iniindex * numpy.array([1., -1.])
            self.outputs = self.outputs.assign({
                'storindex': xarray.DataArray(
                    [mean(_tmp, weights=self.weights) + _index * _tmp.std(), ],
                    dims=('driver', 'index'),
                    coords={'driver': self.drivers,
                            'index': ['index+', 'index-']})})
        elif self.ndrivers == 2:
            _tmp1 = self.data.get(self.drivers[0]).values
            _tmp2 = self.data.get(self.drivers[1]).values
            #self.drivers_cor, _ = scipy.stats.pearsonr(_tmp1, _tmp2)
            self.drivers_cor = _pearson(_tmp1, _tmp2, self.weight)
            _index1 = _iniindex * numpy.sqrt((1. - self.drivers_cor ** 2.) / (1. - self.drivers_cor))
            _index2 = _iniindex * numpy.sqrt((1. - self.drivers_cor ** 2.) / (1. + self.drivers_cor))
            _indexes1 = numpy.array([_index1, -_index1, _index2, -_index2])
            _indexes2 = numpy.array([_index1, -_index1, -_index2, _index2])
            self.outputs = self.outputs.assign({
                'storindex': xarray.DataArray(
                    [float(mean(_tmp1, weights=self.weights)) + _indexes1 * float(std(_tmp1, weights=self.weights)),
                     float(mean(_tmp2, weights=self.weights)) + _indexes2 * float(std(_tmp2, weights=self.weights))],
                    dims=('driver', 'index'),
                    coords={'driver': self.drivers,
                            'index': ['index++', 'index--', 'index+-', 'index-+']})})
        elif self.ndrivers == 3:
            _tmp1 = self.data.get(self.drivers[0]).values
            _tmp2 = self.data.get(self.drivers[1]).values
            _tmp3 = self.data.get(self.drivers[2]).values
            _cor12 = _pearson(_tmp1, _tmp2, self.weights) # self.drivers
            _cor13 = _pearson(_tmp1, _tmp3, self.weights)
            _cor23 = _pearson(_tmp2, _tmp3, self.weights)
            _index1 = _iniindex * numpy.sqrt((_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*_cor13*_cor23 - 1) / (_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*(_cor13+_cor23-1) - 2*_cor13(_cor23-1) + 2*_cor23 - 3))
            _index2 = _iniindex * numpy.sqrt((_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*_cor13*_cor23 - 1) / (_cor12**2+_cor13**2+_cor23**2 + 2*_cor12*(_cor13-_cor23-1) + 2*_cor13(_cor23+1) - 2*_cor23 - 3))
            _index3 = _iniindex * numpy.sqrt((_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*_cor13*_cor23 - 1) / (_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*(_cor13-_cor23+1) + 2*_cor13(_cor23-1) + 2*_cor23 - 3))
            _index4 = _iniindex * numpy.sqrt((_cor12**2+_cor13**2+_cor23**2 - 2*_cor12*_cor13*_cor23 - 1) / (_cor12**2+_cor13**2+_cor23**2 + 2*_cor12*(_cor13+_cor23+1) - 2*_cor13(_cor23+1) - 2*_cor23 - 3))
            _indexes1 = numpy.array([ _index1, -_index1,  _index2, -_index2,  _index3, -_index3,  _index4, -_index4])
            _indexes2 = numpy.array([ _index1, -_index1, -_index2,  _index2, -_index3,  _index3,  _index4, -_index4])
            _indexes3 = numpy.array([ _index1, -_index1,  _index2, -_index2, -_index3,  _index3, -_index4,  _index4])
            self.outputs = self.outputs.assign({
                'storindex': xarray.DataArray(
                    [float(mean(_tmp1, weights=self.weights)) + _indexes1 * float(std(_tmp1, weights=self.weights)),
                     float(mean(_tmp2, weights=self.weights)) + _indexes2 * float(std(_tmp2, weights=self.weights)),
                     float(mean(_tmp3, weights=self.weights)) + _indexes3 * float(std(_tmp3, weights=self.weights))],
                    dims=('driver', 'index'),
                    coords={'driver': self.drivers,
                            'index': ['index+++', 'index---',
                                      'index+-+', 'index-+-',
                                      'index+--', 'index-++',
                                      'index++-', 'index--+']})})
        else:
            raise Error('The storylines method only has been coded for 1 to 3 drivers yet. Feel free to add the extended calculation for 3 drivers or more. :)')


    def which_storyline_OLD_VERY(
        self, drivers, predict=False
    ):
        if self.ndrivers == 2:
            _cond_X_g = numpy.where(drivers[0].values > self.Xreg * drivers[1].values + self.Xinter)
            _cond_X_l = numpy.where(drivers[0].values < self.Xreg * drivers[1].values + self.Xinter)
            _cond_Y_g = numpy.where(drivers[1].values > self.Yreg * drivers[0].values + self.Yinter)
            _cond_Y_l = numpy.where(drivers[1].values < self.Yreg * drivers[0].values + self.Yinter)
            _families_ind = [list(set(self.inner[0].tolist()) & set(_cond_X_g[0].tolist()) & set(_cond_Y_g[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_l[0].tolist()) & set(_cond_Y_l[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_g[0].tolist()) & set(_cond_Y_l[0].tolist())),
                             list(set(self.inner[0].tolist()) & set(_cond_X_l[0].tolist()) & set(_cond_Y_g[0].tolist()))]
        out = list(_families_ind)
        if predict:
            out.append()
        return out


def _pearson(x, y, weights=None):
    '''
    Weighted correlation
    '''
    if weights is not None:
        #def _mean_w(x, w):
        #    '''Weighted Mean'''
        #    return numpy.sum(x * w) / numpy.sum(w)
        #def _cov_w(x, y, w):
        #    '''Weighted Covariance'''
        #    return numpy.sum(w * (x - _mean_w(x, w)) * (y - _mean_w(y, w))) / numpy.sum(w)
        #out = _cov_w(x, y, w) / numpy.sqrt(_cov_w(x, x, w) * _cov_w(y, y, w))
        out = cov(numpy.array([x, y]), weights=weights) / numpy.sqrt(std(x, weights=weights) * std(y, weights=weights))
    else:
        out, _ = scipy.stats.pearsonr(x, y)
    return out

"""





