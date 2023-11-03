'''
#command: python3 plot/plot_z_reaction_rate.py --con /njord2/paimans/std_adv/2D/concentration/ -D 0.01 0.0025 0.001 --it_max 8
'''
import dolfin as df
from itertools import product
import h5py
import os
from numpy.core.fromnumeric import size
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Reaction rate of c")
    parser.add_argument("--it_max", type=int, default=0, help="Maximum iteration")
    parser.add_argument("-D", nargs='+', type=float, help='<Required> Diffusivities', required=True)
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    #parser.add_argument("--Lx", type=float, default=1, help="Lx")
    #parser.add_argument("--Ly", type=float, default=1, help="Ly")
    parser.add_argument("--Lz", type=float, default=1, help="Lz")
    #parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    def mpi_print(*args):
        if rank == 0:
            print(*args)  

    D_ = list(reversed(sorted(args.D)))
    pore_size = args.L
    Pe = [pore_size / D for D in D_]
    it_max = args.it_max
    it_values = list(range(it_max+1))

    #Lx = args.Lx
    #Ly = args.Ly
    Lz = args.Lz
  
    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff"]

    markers = ["*","H", "<","x",'o',"+", "v", "s","D"]
    colors = ['b', 'g', 'r', 'c']

    for Pe_idx, Pe__ in enumerate(Pe):

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        cmap = plt.get_cmap("plasma")
        colab = cmap(np.linspace(0, 0.8, len(it_values)))

        for it_idx, it in enumerate(it_values):
            print(it_idx)
            color = colab[it_idx]
            marker = markers[it_idx]
            data = dict()
            with h5py.File(args.con+"data/abc_data_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f:
                data["ux"] = np.array(h5f["ux"])
                data["uy"] = np.array(h5f["uy"])
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
            uz_mean = -data["uz"].mean(axis=(1, 2))
            for species in specii:
               conc[species] = data[species]["conc"]
               Jz[species] = -data["uz"] * conc[species] - data[species]["J_diff"]

            #por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"])), dtype=float)
            #ma = np.array(np.logical_not(por), dtype=bool)
            
            #mask = np.logical_not(np.logical_or(conc["a"], conc["b"]))
            #for species in specii:
            #    conc[species] = np.ma.masked_where(mask, conc[species])

            for species in ['c']:
                conc[species] = data[species]["conc"]
                Jz[species] = -data["uz"] * conc[species] - data[species]["J_diff"]
            Jz_mean['c'] = Jz[species].mean(axis=(1, 2))

            



        #############################################
            z = Lz - zgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))

            dJz_dz = np.gradient(Jz_mean['c'], z)
            ax.plot(z, dJz_dz, label='Iteration {}'.format(it),color=color, marker=marker, alpha=0.85)
            #ax.set_title(r'$\alpha = {:.4f}$'.format(D_[Pe_idx]))
            ax.set_title('Pe = {}'.format(int(Pe__))) 
            
            ax.set_xlabel("Z")
            ax.set_ylabel(r'$\mathrm{R_{c}}$')
            legend = plt.legend(fancybox=True, shadow=True)
            plt.tight_layout()
        fig.savefig(args.con + 'plots/reaction_rate/R_{}_Pe{}_itmax{}_different_it.pdf'.format(species,Pe__, it), dpi=300)

