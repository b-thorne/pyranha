import pyranha
import numpy as np
import matplotlib.pyplot as plt


def main():
    fpath = 'configurations/test.py'
    pyr = pyranha.Pyranha(fpath)
    #fishers = pyr.iterate_instrument_parameter_1d(fsky=np.linspace(0.2, 0.8, 20))
    #print fishers
    fishers2d = pyr.iterate_instrument_parameter_2d(lmin=np.arange(2, 20), lmax=np.arange(200, 220))
    print fishers2d
    """
    pyr.compute_instrument()
    pyr.compute_cosmology()
    fisher = pyr.fisher()

    pyr.delensing = True
    pyr.delensing_factor = 0.1
    fisher_lens = pyr.fisher()

    pyr.map_res = 0.02
    pyr.compute_instrument()
    fisher_fgnd = pyr.fisher()

    fig = pyranha.plot_fisher_1d([fisher, fisher_fgnd, fisher_lens], [r'LiteBIRD', r'LiteBIRD + 2%, foreground', r'LiteBIRD + 90% delensing'])
    fig.savefig("plots/demonstration.pdf", bbox_inches='tight')
    fig.savefig("plots/demonstration.png", bbox_inches='tight')
    plt.show()
    """
    return


if __name__ == "__main__":
    main()
