from statsmodels.stats.weightstats import DescrStatsW

def mean(data, weights=None):
    '''
    Parameters
    ----------
    data: array_like, 1-D or 2-D
        dataset

    weights: None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    '''
    return DescrStatsW(data, weights=weights).mean


def var(data, weights=None):
    return DescrStatsW(data, weights=weights).var


def std(data, weights=None):
    return DescrStatsW(data, weights=weights).std


def median(data, weights=None):
    return DescrStatsW(data, weights=weights).quantile(0.5)


def quantile(data, probs, weights=None, return_pandas=False):
    return DescrStatsW(data, weights=weights).quantile(probs, return_pandas=return_pandas)


def cov(data, weights=None):
    return DescrStatsW(data, weights=weights).cov


def demeaned(data, weights=None):
    return DescrStatsW(data, weights=weights).demeaned


def ttest_mean(data, value=0, alternative='two-sided', weights=None):
    '''
    Parameters
    ----------

    value: float or array
        The hypothesized value for the mean (0 by default).

    alternative: str
        The alternative hypothesis, H1, has to be one of the following:
         - ‘two-sided’: H1: mean not equal to value (default)
         - ‘larger’ : H1: mean larger than value
         - ‘smaller’ : H1: mean smaller than value

    Returns
    -------

    tstat: float
        Test statistic

    pvalue: float
        pvalue of the t-test

    df: int or float
    '''
    return DescrStatsW(data, weights=weights).ttest_mean(value=value, alternative=alternative)