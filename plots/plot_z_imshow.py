import dolfin as df
from fenicstools import interpolate_nonmatching_mesh_any, StructuredGrid
from itertools import product
import h5py
import os
from numpy.core.fromnumeric import size
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady ADR")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--Lx", type=float, default=1, help="Lx")
    parser.add_argument("--Ly", type=float, default=1, help="Ly")
    parser.add_argument("--Lz", type=float, default=1, help="Lz")
    parser.add_argument("--arrows", action="store_true", help="Plot arrows")
    #parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()


df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True  

if __name__ == "__main__":

    args = parse_args()

    def mpi_print(*args):
        if rank == 0:
            print(*args)  

    D=args.D
    pore_size = args.L
    it = args.it
    Lx = args.Lx
    Ly = args.Ly
    Lz = args.Lz
    eps = 1e-8

    Pe__ = pore_size / D
    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff", "Iz"]

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

    conc = dict()
    uz_mean = -data["uz"].mean(axis=(1, 2))
    for species in specii:
        conc[species] = data[species]["conc"]

    por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"]), np.copy(conc["c"])), dtype=float)
    ma = np.array(np.logical_not(por), dtype=bool)
    
    X, Y = np.meshgrid(xgrid, ygrid)

    for iz, z in enumerate(zgrid):
        for species in ['delta', 'c']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if species=='delta':
                cmap = plt.get_cmap('coolwarm')
            if species=='c':
                cmap = plt.get_cmap('viridis')
            pcm = ax.pcolormesh(X, Y, np.ma.masked_where(ma[iz, :, :], conc[species][iz, :, :]), cmap=cmap, shading="nearest")
            if args.arrows:
                ax.streamplot(X, Y, np.ma.masked_where(ma[iz, :, :], data["ux"][iz, :, :]), data["uy"][iz, :, :], color='r', density=1.0)
            ax.set_aspect("equal")
            ax.set_xlabel("X".format(species), fontsize=19)
            ax.set_ylabel("Y".format(species), fontsize=19)
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            ax.tick_params(labelsize=18)
            cbar = plt.colorbar(pcm, ax=ax, label='Concentration of {}'.format(species), shrink=0.8)
            cbar.ax.tick_params(labelsize=18) 
            cbar.ax.set_ylabel('Concentration of {}'.format(species), rotation=90, labelpad=20, fontsize=18)  

            ax.set_title("Distance from the inlet: Z = {:.2f}".format(1-zgrid[iz]), fontsize=20)
            plt.savefig(args.con + 'plots/imshow/Pe{}/video_{}_Pe={}_it{}_z_{}.png'.format(Pe__, species, Pe__, it, iz))
            plt.close()
