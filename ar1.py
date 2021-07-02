import numpy as np
from scipy.optimize import curve_fit
import scipy.signal
import scipy

# OASIS deconvolution 
# https://github.com/j-friedrich/OASIS.git

def deconvolve(y, g=(None,), sn=None, b=None, b_nonneg=True,
               optimize_g=0, penalty=0, **kwargs):
    """Infer the most likely discretized spike train underlying an fluorescence trace
    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_q subject to |c-y|^2 = sn^2 T and s = Gc >= 0
    where q is either 1 or 0, rendering the problem convex or non-convex.
    Parameters:
    -----------
    y : array, shape (T,)
        Fluorescence trace.
    g : tuple of float, optional, default (None,)
        Parameters of the autoregressive model, cardinality equivalent to p.
        Estimated from the autocovariance of the data if no value is given.
    sn : float, optional, default None
        Standard deviation of the noise distribution.  If no value is given,
        then sn is estimated from the data based on power spectral density if not provided.
    b : float, optional, default None
        Fluorescence baseline value. If no value is given, then b is optimized.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        If optimize_g=0 the provided or estimated g is not further optimized.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0
    kwargs : dict
        Further keywords passed on to constrained_oasisAR1 or constrained_onnlsAR2.
    Returns:
    --------
    c : array, shape (T,)
        The inferred denoised fluorescence signal at each time-bin.
    s : array, shape (T,)
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : tuple of float
        Parameters of the AR(2) process that models the fluorescence impulse response.
    lam: float
        Optimal Lagrange multiplier for noise constraint under L1 penalty
    """

    if g[0] is None or sn is None:
        fudge_factor = .97 if (optimize_g and len(g) == 1) else .98
        est = estimate_parameters(y, p=len(g), fudge_factor=fudge_factor)
        if g[0] is None:
            g = est[0]
        if sn is None:
            sn = est[1]
    if len(g) == 1:
        return constrained_oasisAR1(y, g[0], sn, optimize_b=True if b is None else False,
                                    b_nonneg=b_nonneg, optimize_g=optimize_g,
                                    penalty=penalty, **kwargs)
    elif len(g) == 2:
        if optimize_g > 0:
            warn("Optimization of AR parameters is already fairly stable for AR(1), "
                 "but slower and more experimental for AR(2)")
        return constrained_onnlsAR2(y, g, sn, optimize_b=True if b is None else False,
                                    b_nonneg=b_nonneg, optimize_g=optimize_g,
                                    penalty=penalty, **kwargs)
    else:
        print('g must have length 1 or 2, cause only AR(1) and AR(2) are currently implemented')

# functions to estimate AR coefficients and sn from
# https://github.com/agiovann/Constrained_NMF.git
# https://github.com/j-friedrich/OASIS.git
def estimate_parameters(y, p=2, range_ff=[0.25, 0.5], method='mean', lags=10, fudge_factor=1., nonlinear_fit=False):
    """
    Estimate noise standard deviation and AR coefficients
    Parameters
    ----------
    p : positive integer
        order of AR system
    lags : positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    sn = GetSn(y, range_ff, method)
    g = estimate_time_constant(y, p, sn, lags, fudge_factor, nonlinear_fit)

    return g, sn


def estimate_time_constant(y, p=2, sn=None, lags=10, fudge_factor=1., nonlinear_fit=False):
    """
    Estimate AR model parameters through the autocovariance function
    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p : positive integer
        order of AR system
    sn : float
        sn standard deviation, estimated if not provided.
    lags : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    Returns
    -------
    g : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(y)

    lags += p
    # xc = axcov(y, lags)[lags:]
    y = y - y.mean()
    xc = np.array([y[i:].dot(y[:-i if i else None]) for i in range(1 + lags)]) / len(y)

    if nonlinear_fit and p <= 2:
        xc[0] -= sn**2
        g1 = xc[:-1].dot(xc[1:]) / xc[:-1].dot(xc[:-1])
        if p == 1:
            def func(x, a, g):
                return a * g**x
            popt, pcov = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1)) #, bounds=(0, [3 * xc[0], 1]))
            return popt[1:2] * fudge_factor
        elif p == 2:
            def func(x, a, d, r):
                return a * (d**(x + 1) - r**(x + 1) / (1 - r**2) * (1 - d**2))
            popt, pcov = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1, .1))
            d, r = popt[1:]
            d *= fudge_factor
            return np.array([d + r, -d * r])

    xc = xc[:, np.newaxis]
    A = scipy.linalg.toeplitz(xc[np.arange(lags)],
                              xc[np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(y, range_ff=[0.25, 0.5], method='mean'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies
    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues
    Returns
    -------
    sn : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sn