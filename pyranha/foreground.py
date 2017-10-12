import numpy as np
import matplotlib.pyplot as plt


def BB_scaling(nu, nu_0, T):
    """Blackbody scaling factor from frequency nu, to frequency nu_0.

    Parameters
    ----------
    nu : `float`
        Frequency to be scaled to.
    nu_0 : `float`
        Reference frequency to be scaled from.
    T : `float`
        Temperature of blackbody.

    Returns
    -------
    `float`
        Ratio of BB radiance between frequencies `nu` and `nu_0`.
    """
    h = 6.63e-34
    kb = 1.38e-23
    a = np.exp(h * nu_0 * 1.e9 / (kb * T)) - 1.
    b = np.exp(h * nu * 1.e9 / (kb * T)) - 1.
    return (a / b) ** 2


def synch_cl(nu, ell, A_S, alpha_S, beta_S, nu_S_0, ell_S_0):
    """Model for the synchrotron power spectrum.

    Parameters
    ----------
    nu : float
        Frequency at which to evaluate the spectrum.
    ell : array_like(`int`, ndim=1)
        Multipole range over which to evaluate spectrum.
    A_S : `float`
        Amplitdue of spectrum at reference multipole `ell_S_0`.
    alpha_S : `float`
        Index of the frequency dependence.
    beta_S : `float`
        Index of the multipole dependence.
    nu_S_0 :`float`
        Reference frequency.
    ell_S_0 : `int`
        Reference multipole.

    Returns
    -------
    array_like(`float`, ndim=1)
        The synchrotron spectrum at frequency `nu`.
    """
    s = (nu / nu_S_0) ** (2. * alpha_S) * (ell / ell_S_0) ** beta_S
    return A_S * s


def dust_cl(nu, ell, p, T, A_D, alpha_D, beta_D, nu_D_0, ell_D_0):
    """Model for the dust power spectrum.

    Parameters
    ----------
    nu : `float`
        Frequency at which to evaluate the spectrum.
    ell : array_like(int, ndim=1)
        Multipole range over which to evaluate spectrum.
    p : `float`
        Polarization fraction of the dust.
    T : `float`
        Temperature of the dust.
    A_D : `float`
        Amplitude of dust spectrum at reference multipole `ell_D_0`.
    alpha_D : `float`
        Index of the frequency dependence of spectrum.
    beta_D : `float`
        Index of multipole dependence of spectrum.
    nu_D_0 : `float`
        Reference frequency.
    ell_D_0 : `int`
        Reference multipole.

    Returns
    -------
    array_like(`float`, ndim=1)
        Dust spectrum at frequency `nu`.
    """
    s = (nu / nu_D_0) ** (2. * alpha_D) * (ell / ell_D_0) ** beta_D
    bb = BB_scaling(nu, nu_D_0, T)
    return p ** 2 * A_D * s * bb


def fg_res_sys(nu, nu_S_ref, alpha_S, nu_D_ref, alpha_D, N_chan, n_l):
    """Systematics introduced in CMB channels by foreground removal.

    Parameters
    ----------
    nu : `float`
        Frequency at which to evaluate.
    nu_S_ref : `float`
        Reference frequency of synchrotron spectrum.
    alpha_S : `float`
        Index of frequency dependence of synchrotron spectrum.
    nu_D_ref : `float`
        Reference frequency of dust spectrum.
    alpha_D : `float`
        Index of frequency dependence of dust spectrum.
    N_chan : `int`
        Number of foreground removal.
    n_l: list(array_like(float, ndim=1))
        List of the instrumental noise in foreground channels.

    Returns
    -------
    array_like(float, ndim=1)
        Total noise spectrum at frequency nu due to foreground channel
        systematics.
    """
    f = (nu / nu_S_ref) ** (2 * alpha_S) + (nu / nu_D_ref) ** (2 * alpha_D)
    summation = 1. / sum([1. / n for n in n_l])
    a = 4. / (N_chan * (N_chan - 1.))
    return a * summation * f
