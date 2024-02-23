import numpy

def mean_w(x, w):
    """Weighted Mean"""
    return numpy.sum(x * w) / numpy.sum(w)

def cov_w(x, y, w):
    """Weighted Covariance"""
    return numpy.sum(w * (x - mean_w(x, w)) * (y - mean_w(y, w))) / numpy.sum(w)

def pearson_w(x, y, w):
    """Weighted Correlation"""
    return cov_w(x, y, w) / numpy.sqrt(cov_w(x, x, w) * cov_w(y, y, w))
