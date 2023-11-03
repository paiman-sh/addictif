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
    #mpi_print(conc['delta'][10,:,400])

    por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"])), dtype=float)
    ma = np.array(np.logical_not(por), dtype=bool)
    
    mask = np.logical_not(np.logical_or(conc["a"], conc["b"]))
    for species in specii:
        conc[species] = np.ma.masked_where(mask, conc[species])
    

#############################################
    num_curves = 100
    skip = 10
    nbins = 100
    for species in specii:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.viridis(np.linspace(0, 1, skip))))
        cmap = plt.get_cmap('viridis')
        #plt.imshow(conc[species][0, :, :])
        #if species == "a":
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.rcParams.update({'font.size': 18})
        labels = []
        for iz in range(conc[species].shape[0])[::skip]:
            #print("iz =", iz)
            h, xb = np.histogram(conc[species][iz, :, :].compressed(), bins=nbins, density=True)
            x = 0.5*(xb[1:]+xb[:-1])
            color = cmap(iz / conc[species].shape[0])
            ax.plot(x, h, label="z={:.2f}".format(zgrid[iz]), color=color, linewidth=2.0)
            
            labels.append("z={}".format("{:.2f}".format(1 - zgrid[iz])))
        #ax.set_yscale("log")
        #ax.set_xscale("log")
        ax.set_xlabel("{}".format(species), fontsize=18)
        ax.set_ylabel("PDF p({})".format(species), fontsize=18)
        plt.legend(labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.138),fancybox=True, shadow=True, prop={'size': 16})
        plt.yticks(fontsize=17)
        plt.xticks(fontsize=17)
        #ax.set_title("Pe = {}".format(Pe__),fontsize=19)
        #ax.set_yscale("log")
        ylim = False
        plt.savefig(args.con + 'plots/pdf/Pe_{}/pdf_{}_Pe={}_it{}_ylim{}.png'.format(Pe__, species, Pe__, it, ylim))

        ylim = True
        ax.set_xlabel("{}".format(species), fontsize=23)
        ax.set_ylabel("PDF p({})".format(species), fontsize=23)
        plt.legend(labels, ncol=3, loc='upper center',fancybox=True, shadow=True, prop={'size': 21})
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        if ylim==True:
            if species == 'c':
                plt.ylim(0,15)
                plt.savefig(args.con + 'plots/pdf/Pe_{}/pdf_{}_Pe={}_it{}_ylim{}.png'.format(Pe__, species, Pe__, it, ylim))
            else:
                plt.ylim(0,5)
                plt.savefig(args.con + 'plots/pdf/Pe_{}/pdf_{}_Pe={}_it{}_ylim{}.png'.format(Pe__, species, Pe__, it, ylim))
        #plt.show()
        
