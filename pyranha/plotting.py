import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pylab import cm
import numpy as np


def plot_fisher_corner(fisher_mats, labels, xcen=0, ycen=1, xmin=-2.1, xmax=2.1,
                    ymin=0.88, ymax=1.12, title="", opath=None):
    """Function to plot 2x2 Fisher matrices.
    """
    # Set up the figure environment.
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    plt.subplots_adjust(hspace=0, wspace=0)
    # Set limits:
    xarr = np.linspace(xmin, xmax, 1000)
    yarr = np.linspace(ymin, ymax, 1000)
    # Loope over the matrices to add them to the plot.
    for fisher_mat, label in zip(fisher_mats, labels):
        # Rescale the elements of the matrix x -> 10^3 x
        fisher_mat[0, 0] *= 10**-6
        fisher_mat[0, 1] *= 10**-3
        fisher_mat[1, 0] *= 10**-3
        # Compute the inverse of the matrix.
        covar = np.linalg.inv(fisher_mat)
        # Get the marginalized 1-sigma values.
        sigma_00 = np.sqrt(covar[0, 0])
        sigma_11 = np.sqrt(covar[1, 1])
        # Compute Gaussians over this range centered on the parameters xcen
        # and ycen.
        oned_gauss_x = np.exp(- (xarr - xcen) ** 2 / (2. * sigma_00 ** 2))
        oned_gauss_y = np.exp(- (yarr - ycen) ** 2 / (2. * sigma_11 ** 2))
        # Get eigenvalues and eigenvectors of covariance matrix
        w, v = np.linalg.eigh(covar)
        # Get the angle (in degrees) from the vector with largest eigenvalue
        angle_deg = np.arctan2(v[1, 1], v[0, 1]) * 180. / np.pi
        width1 = 2 * np.sqrt(2.3 * w[1])
        height1 = 2 * np.sqrt(2.3 * w[0])
        width2 = 2 * np.sqrt(5.99 * w[1])
        height2 = 2 * np.sqrt(5.99 * w[0])

        l1, = ax1.plot(xarr, oned_gauss_x, linewidth=2, label=label)

        ax3.plot(yarr, oned_gauss_y, linewidth=2, color=l1.get_color())
        eas1 = Ellipse(xy=(xcen, ycen), width=width1, height=height1,
                    linewidth=2, angle=angle_deg, alpha=0.8, linestyle="-",
                    color=l1.get_color())
        eas2 = Ellipse(xy=(xcen, ycen), width=width2, height=height2,
                    linewidth=2, angle=angle_deg, alpha=.4, linestyle="--",
                    color=l1.get_color())

        ax2.add_artist(eas1)
        ax2.add_artist(eas2)
    ax2.set_xlim(xarr[0], xarr[-1])
    ax2.set_ylim(yarr[0], yarr[-1])
    ax1.set_xlim(xarr[0], xarr[-1])
    ax3.set_xlim(yarr[0], yarr[-1])
    ax2.set_ylabel(r"$A_L$")
    ax2.set_xlabel(r"$10^{-3}r$")
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax1.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    ax3.set_xticks([0.9, 0.95, 1., 1.05, 1.1])
    ax2.set_yticks([0.9, 0.95, 1., 1.05, 1.1])
    ax2.set_xticks([-2, -1, 0., 1, 2])
    if opath is not None:
        fig.savefig(opath, bbox_inches='tight')
    return


def plot_fisher_1d(arr_x, arr_fisher_mats, labels, xlabel=None, opath=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    sigma = []
    for fisher_mats, label in zip(arr_fisher_mats, labels):
        sigmas = map(calculate_sigma_00, fisher_mats)
        ax.loglog(arr_x, sigmas, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$10^{-3]} \sigma_r$")
    ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
    if opath is not None:
        fig.savefig(opath, bbox_inches='tight')
    return


def plot_fisher_2d(arr_x, arr_y, fisher_mats_2d, xlabel=None, ylabel=None,
                    opath=None):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    X, Y = np.meshgrid(arr_x, arr_y)
    Z = calculate_sigma_2d(fisher_mats_2d)
    im = plt.imshow(Z, cmap=cm.RdBu, interpolation='bilinear',
                    extent=[arr_x[0], arr_x[-1], arr_y[0], arr_y[-1]],
                    origin='lower', aspect='auto')
    cset = plt.contour(X, Y, Z, colors='k')
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=12)
    plt.colorbar(im, label=r'$10^{-3} \sigma_r$')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if opath is not None:
        fig.savefig(opath, bbox_inches='tight')
    return


def calculate_sigma_2d(fisher_mat_2d):
    shape = fisher_mat_2d.shape
    sigma = np.zeros((shape[1], shape[0]))
    for i, fisher_1d in enumerate(fisher_mat_2d):
        for j, fisher in enumerate(fisher_1d):
            sigma[j, i] = calculate_sigma_00(fisher)
    return sigma


def calculate_sigma_00(fisher_mat):
    # Rescale the elements of the matrix x -> 10^3 x
    fisher_mat[0, 0] *= 10**-6
    fisher_mat[0, 1] *= 10**-3
    fisher_mat[1, 0] *= 10**-3
    # Compute the inverse of the matrix.
    # Get the marginalized 1-sigma values.
    return np.sqrt(np.linalg.inv(fisher_mat)[0, 0])
