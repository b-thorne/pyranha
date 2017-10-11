import numpy as np
import matplotlib.pyplot as plt


def BB_scaling(nu, nu_0, T):
    """Blackbody scaling factor from frequency nu, to frequency nu_0.

    :param float nu: Frequency scaled to
    :param float nu_0: Reference frequency
    :param float T: Temperature of blackbody
    :return: Ratio of BB radiance between frequencies
    :rtype: float
    """
    h = 6.63e-34
    kb = 1.38e-23
    a = np.exp(h * nu_0 * 1.e9 / (kb * T)) - 1.
    b = np.exp(h * nu * 1.e9 / (kb * T)) - 1.
    return (a / b) ** 2


def synch_cl(nu, ell, A_S, alpha_S, beta_S, nu_S_0, ell_S_0):
    """Model for the synchrotron power spectrum.

    :param float nu: Frequency at which to evaluate the spectrum
    :param numpy.array nu: Multipole range over which to evaluate spectrum
    :param float A_S: amplitdue of spectrum at reference Multipole
    :param float alpha_S: index of the frequency dependence
    :param float beta_S: index of the multipole dependence
    :param float nu_S_0: reference frequency
    :param float ell_S_0: reference multipole
    :return: the synchrotron spectrum at frequency nu.
    :rtype: numpy.array
    """
    s = (nu / nu_S_0) ** (2. * alpha_S) * (ell / ell_S_0) ** beta_S
    return A_S * s


def dust_cl(nu, ell, p, T, A_D, alpha_D, beta_D, nu_D_0, ell_D_0):
    """Model for the dust power spectrum.

    :param float nu: Frequency at which to evaluate the spectrum
    :param numpy.ndarray ell: Multipole range over which to evaluate spectrum
    :param float p: Polarization fraction of the dust_cl
    :param float T: Temperature of the dust
    :param float A_D: Amplitude of dust spectrum at reference Multipole
    :param float alpha_D: index of the frequency dependence of spectrum
    :param float beta_D: index of multipole dependence of spectrum
    :param float nu_D_0: reference frequency
    :param float ell_D_0: reference multipole
    :return: dust spectrum at frequency nu
    :rtype: numpy.array
    """
    s = (nu / nu_D_0) ** (2. * alpha_D) * (ell / ell_D_0) ** beta_D
    bb = BB_scaling(nu, nu_D_0, T)
    return p ** 2 * A_D * s * bb


def fg_res_sys(nu, nu_S_ref, alpha_S, nu_D_ref, alpha_D, N_chan, n_l):
    """Systematics introduced in CMB channels by foreground removal.

    :param float nu: frequency at which to evaluate
    :param float nu_S_ref: reference frequency of synchrotron spectrum
    :param float alpha_S: index of frequency dependence of synchrotron spectrum
    :param float nu_D_ref: reference frequency of dust spectrum
    :param float alpha_D: index of frequency dependence of dust spectrum
    :param int N_chan: number of foreground removal channels
    :param n_l: list of the instrumental noise in foreground channels
    :type n_l: list of numpy.array
    :return: total noise spectrum at frequency nu due to foreground channel systematics
    :rtype: numpy.array
    """
    f = (nu / nu_S_ref) ** (2 * alpha_S) + (nu / nu_D_ref) ** (2 * alpha_D)
    summation = 1. / sum([1. / n for n in n_l])
    a = 4. / (N_chan * (N_chan - 1.))
    return a * summation * f
