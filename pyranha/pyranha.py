import numpy as np
from instrument import instrument
from classy import Class
import ConfigParser


def read_array_from_csv(csv_str):
    """Function to return an array from a string of float separated by commas.
    """
    return np.array([float(f) for f in csv_str.split(',')])


def calculate_lensing_template(bb_lensed, bb_unlensed):
    """Function to calculate the BB spectrum of lensing from output CLASS
    BB spectra. These spectra are the lened and unlensed outputs for a given
    set of cosmological parameters, therefore we simply take the difference
    of the lensed and unlensed BB spectrum. Note that this requires CLASS
    to have been run for l_max >= 2500 in order to achieve reliable accuracy.
    """
    return bb_lensed - bb_unlensed


class Pyranha(object):
    """Function to calculate the fisher information for a given input set of
    cosmological spectra and instrument spectra.
    """
    def __init__(self, fpath):
        """Read in a configuration file specified by fpath. This file containts
        all the information about the experimental setup, the various
        foreground models, fitting parameters, and number of runs.
        """
        cnfg = ConfigParser.RawConfigParser()
        cnfg.read(fpath)

        # Read in experiment parameters
        self.nus = read_array_from_csv(cnfg.get('experiment', 'nus'))
        self.sens = read_array_from_csv(cnfg.get('experiment', 'sens'))
        # Convert sensitivity from uK to unitless in order to combine with
        # spectra from CLASS.
        self.sens /= 2.7255e6
        self.beams = read_array_from_csv(cnfg.get('experiment', 'beams'))
        self.fsky = cnfg.getfloat('experiment', 'fsky')
        self.lmin = cnfg.getint('experiment', 'lmin')
        self.lmax = cnfg.getint('experiment', 'lmax')

        # Read in foreground parameters
        self.include_foregrounds = cnfg.getboolean('foreground', 'include')
        self.map_res = cnfg.getfloat('foreground', 'map_res')

        # Read in cosmology parameters
        self.delensing = cnfg.get('cosmology', 'delensing')
        if self.delensing:
            self.delensing_factor = cnfg.getfloat('cosmology', 'delensing_factor')
        else:
            self.delensing_factor = 1.
        if cnfg.getboolean('cosmology', 'lensing'):
            self.lensing = 'yes'
        self.r = cnfg.getfloat('cosmology', 'r')
        return


    def run_class(self, **kwargs):
        """Function to run class and return the lensed and unlensed spectra as
        dictionaries.
        :param kwargs: CLASS parameters different from the default.
        :return: lensed and unlensed spectra
        :rtype: (dict, dict)
        """
        cosmo = Class()

        # Set some parameters we want that are not in the default CLASS setting.
        class_pars = {
            'output': 'tCl pCl lCl',
            'modes': 's, t',
            'lensing': self.lensing,
            'r': self.r,
            }

        # Update CLASS run with any kwargs that were passed. This is useful in
        # Pyranha.compute_cosmology in order to compute the r=1 case.
        class_pars.update(kwargs)
        cosmo.set(class_pars)
        cosmo.compute()

        # Get the lensed and unlensed spectra as dictionaries.
        cls_l = cosmo.lensed_cl(self.lmax)
        cls_u = cosmo.raw_cl(self.lmax)

        # Do the memory cleanup.
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_l, cls_u


    def compute_cosmology(self):
        """Function to run CLASS with the given parameters. And return
        the relevant spectra for the Fisher calculation.

        :return: the unlensed and lensed spectra, and the primordial BB template and lensing template.
        :rtype: list of numpy.ndarray

        """
        # Calculate the lensed and unlensed spectra for the requested cosmology.
        cls_l, cls_u = self.run_class()

        # Calculate the lensing template for this cosmology
        self.BB_lens_template = calculate_lensing_template(cls_l['bb'], cls_u['bb'])[self.lmin : self.lmax + 1]
        self.BB_lensed = cls_l['bb'][self.lmin : self.lmax + 1]

        # Change the tensor-to-scalar ratio to one for the calculation of the
        # primordial B-mode signal template.
        _, cls_u = self.run_class(r=1)
        self.BB_prim_template = cls_u['bb'][self.lmin : self.lmax + 1]
        return


    def compute_instrument(self):
        """Method to compute the instrumental and foreground contribution to BB
        spectrum given the current state of the object and a given value of
        the residual foregrounds, sigma_rf.

        :return: instrument + foreground BB noise spectrum
        :rtype: numpy.array
        """
        self.BB_inst = instrument(self.nus, self.sens, self.beams, self.map_res,
                                    lmin=self.lmin, lmax=self.lmax)


    def fisher(self):
        """Method to compute the Fisher matrix corresponding to the current
        state of the object (can change during a parameter iteration).

        :return: Fisher matrix corresponding to current properties
        :type: numpy.ndarray
        """
        # Shape of summand is then multipolesxNparamsxNparams.
        fish = np.zeros((2, 2), dtype=np.float32)
        self.ell = np.arange(self.lmin, self.lmax + 1)
        dCdi = np.zeros((2, 2, self.lmax - self.lmin + 1), dtype=np.float32)
        # Assemble noise spectrum from lensing spectrum and instrument spectrum
        # including foreground residuals.
        N_ell_BB = self.delensing_factor * self.BB_lensed + self.BB_inst
        #fill up the summand arrays.
        dCdi[0, 0] = self.BB_prim_template ** 2 / N_ell_BB ** 2
        dCdi[1, 1] = self.BB_lens_template ** 2 / N_ell_BB ** 2
        dCdi[0, 1] = self.BB_lens_template * self.BB_prim_template / N_ell_BB ** 2
        dCdi[1, 0] = dCdi[0, 1]

        summand =  (2. * self.ell + 1.) / 2. * self.fsky * dCdi
        fish = np.sum(summand[..., self.lmin : self.lmax + 1], axis=2)
        return fish
