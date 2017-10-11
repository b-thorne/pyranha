import pyranha
import numpy as np
import matplotlib.pyplot as plt


def main():
    fpath = 'configurations/test.py'
    pyr = pyranha.Pyranha(fpath)
    pyr.compute_instrument()
    pyr.compute_cosmology()
    fisher = pyr.fisher()
    pyranha.plot_fisher(fisher, xcen=0, ycen=1)
    plt.show()
    return

if __name__ == "__main__":
    main()
