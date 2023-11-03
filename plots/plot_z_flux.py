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
    parser = argparse.ArgumentParser(description="Solve steady ADR")
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
    eps = 1e-8

    name2tex = dict(
        a=r"$a$",
        b=r"$b$",
        c=r"$c$",
        delta=r"$\delta$"
    )

    Pe_values = [pore_size / D for D in D_]

    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff"]
    
    markers = ['o', "v", "s"]


   
    fig, ax = plt.subplots(1, 1, figsize=((6, 4)))
    fig_adv, ax_adv = plt.subplots(1, 1, figsize=((6, 4)))
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=((6, 4)))
    fig_por, ax_por = plt.subplots(1, 1, figsize=((6, 4)))

    markers = ['o', 's', '^', 'D']
    colors = ['b', 'g', 'r', 'c']
    
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



        #Jz_mean = dict()
        Jz_tot = dict()
        Jz_diff_tot = dict()
        por_mean = dict()
        Jz = dict()
        Jz_diff = dict()
        conc = dict()
        uz_mean = -data["uz"].mean(axis=(1, 2))
        for species in specii:
            conc[species] = data[species]["conc"]
            Jz[species] = -data["uz"] * conc[species] - data[species]["J_diff"]
            Jz_diff[species] = - data[species]["J_diff"]

        #por = np.array(np.logical_not(np.logical_and(conc["a"] == 0, conc["b"] == 0)), dtype=float)
        por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"])), dtype=float)
        #ma = np.array(np.logical_not(por), dtype=bool)
        
        #mask = np.logical_not(np.logical_or(conc["a"], conc["b"]))
        #for species in specii:
        #    conc[species] = np.ma.masked_where(mask, conc[species])
            

        #for species in specii:           
        #    Jz[species] = np.ma.masked_where(mask, Jz[speci


        for species in specii:
            #Jz_mean[species] = Jz[species].mean(axis=(1, 2))
            Jz_tot[species] = Jz[species].sum(axis=(1, 2))
            Jz_diff_tot[species] = Jz_diff[species].sum(axis=(1, 2))
            por_mean = por.mean(axis=(1, 2))
    
    #############################################
        z = 1 - zgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))
        #ax.set_title("Total mass per cross sectional area")

        for species_idx, species in enumerate(specii):
            marker = markers[species_idx]
            color = colors[species_idx]
            
            ax.plot(z[1:-1], (Jz_tot[species][1:-1]), label="{} (Pe = {})".format(name2tex[species], int(Pe__)), linewidth=2.0, marker=marker,color =color)
            ax_adv.plot(z[1:-1], (Jz_tot[species][1:-1]-Jz_diff_tot[species][1:-1]), label="{} (Pe = {})".format(name2tex[species], int(Pe__)), linewidth=2.0, linestyle='-', marker=marker,color =color)
            ax_diff.plot(z[1:-1], (Jz_diff_tot[species][1:-1]), label="{} (Pe = {})".format(name2tex[species], int(Pe__)), linewidth=2.0, marker=marker,color =color)

        #plt.yticks(fontsize=19)
        #plt.xticks(fontsize=19)
        #plt.style.use('classic')

        for _fig, _ax, filename in [(fig, ax, "total"), (fig_adv, ax_adv, "adv"), (fig_diff, ax_diff, "diff")]:
            _ax.set_xlabel(r"$z$")# , fontsize=25)
            _ax.set_ylabel(r'$\mathrm{J_{z}}$')# , fontsize=25)
            _ax.tick_params(axis='both')
            _ax.set_xlim(-0.05, 1.05)
            legend = _ax.legend(ncol=3 , loc='upper center', fancybox=True, shadow=False)
            #for text in legend.get_texts():
             #   text.set_fontsize(19)
            #_fig.tight_layout()
            _fig.savefig(args.con + 'plots/flux/{}_it_{}.png'.format(filename, it))
