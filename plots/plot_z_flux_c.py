#command: python3 plot/plot_z_flux_c.py --con /njord2/paimans/std_adv/bead_pack/bead_pack_bigger/concentration/ -D 0.015 0.0015 0.00015 --L 0.15 --it 6

import dolfin as df
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Flux of c")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("-D", nargs='+', type=float, help='<Required> Diffusivities', required=True)
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    #parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

def mpi_print(*args):
    if rank == 0:
        print(*args) 

if __name__ == "__main__":

    args = parse_args()

    D_ = list(reversed(sorted(args.D)))
    pore_size = args.L
    it = args.it
    Pe_values = [pore_size / D for D in D_]

    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff"]

    markers = ['o', "v", "s"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

    markers = ['o', "v", "s"]

    cmap = plt.get_cmap("viridis")
    colab = cmap(np.linspace(0, 0.8, len(Pe_values)))
    ax.set_prop_cycle(color=colab)
    
    for Pe_idx, Pe__ in enumerate(Pe_values):
        data = dict()
        with h5py.File(args.con+"data/abc_data_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f:
            #data["ux"] = np.array(h5f["ux"])
            #data["uy"] = np.array(h5f["uy"])
            data["uz"] = np.array(h5f["uz"])
            xgrid = np.array(h5f["x"])
            ygrid = np.array(h5f["y"])
            zgrid = np.array(h5f["z"])
            for species in specii:
                data[species] = dict()
                for field in fields:
                    data[species][field] = np.array(h5f["{}/{}".format(species, field)])

        Jz_mean = dict()
        Jz = dict()
        conc = dict()
        for species in specii:
            conc[species] = data[species]["conc"]
            Jz[species] = -data["uz"] * conc[species] - data[species]["J_diff"]


        for species in ['c']:
            Jz_mean[species] = Jz[species].mean(axis=(1, 2))
    #############################################
        z = 1 - zgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))
        #ax.set_title("Total mass per cross sectional area")
        marker = markers[Pe_idx]
        ax.plot(z[1:-1], (Jz_mean['c'][1:-1]), label="Pe = {}".format(int(Pe__)))

        #plt.yticks(fontsize=19)
        #plt.xticks(fontsize=19)
        #plt.style.use('classic')

        ax.set_xlabel(r"$z$")# , fontsize=25)
        ax.set_ylabel('Avg. flux of c')
        plt.tick_params(axis='both', which='both')
        #ax.set_xlim(-0.05, 1.05)
        plt.legend(fancybox=True, shadow=False)
        #for text in legend.get_texts():
            #   text.set_fontsize(19)
        plt.tight_layout()
        plt.savefig(args.con + 'plots/flux/flux_c_it_{}.png'.format(it),dpi=300)
