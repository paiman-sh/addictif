import dolfin as df
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
    parser = argparse.ArgumentParser(description="flux of delta")
    parser.add_argument("--it1", type=int, default=0, help="Iteration")
    parser.add_argument("--it2", type=int, default=8, help="Iteration")
    parser.add_argument("--D1", type=float, default=1e-2, help="Diffusion1")
    parser.add_argument("--D2", type=float, default=1e-2, help="Diffusion2")
    parser.add_argument("--D3", type=float, default=1e-2, help="Diffusion3")
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

    D1=args.D1
    D2=args.D2
    D3=args.D3
    pore_size = args.L
    it1 = args.it1
    it2 = args.it2
    Lx = args.Lx
    Ly = args.Ly
    Lz = args.Lz
    eps = 1e-8

    name2tex = dict(
        a=r"$a$",
        b=r"$b$",
        c=r"$c$",
        delta=r"$\delta$"
    )

    Pe__1 = pore_size / D1
    Pe__2 = pore_size / D2
    Pe__3 = pore_size / D3
    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff"]

   
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig_adv, ax_adv = plt.subplots(1, 1, figsize=(6, 4))
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=(6, 4))
    fig_por, ax_por = plt.subplots(1, 1, figsize=(6, 4))

    markers = ['o', "*"]
    #colors = ['b', 'g', 'r', 'c']
    Pe_values = [Pe__1,Pe__2,Pe__3]
    it_values = [it1,it2]

    cmap = plt.get_cmap("viridis")
    colab = cmap(np.linspace(0, 0.8, len(Pe_values)))
    for _ax in [ax]:
        _ax.set_prop_cycle(color=colab)
    for it_idx, it__ in enumerate(it_values):
        for Pe_idx, Pe__ in enumerate(Pe_values):
            data = dict()
            with h5py.File(args.con+"data/abc_data_Pe{}_it{}.h5".format(Pe__, it__), "r") as h5f:
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
                #Jz_diff[species] = - data[species]["J_diff"]

            #por = np.array(np.logical_not(np.logical_and(conc["a"] == 0, conc["b"] == 0)), dtype=float)
            por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"])), dtype=float)
            ma = np.array(np.logical_not(por), dtype=bool)
            
            #mask = np.logical_not(np.logical_or(conc["a"], conc["b"]))
            #for species in specii:
            #    conc[species] = np.ma.masked_where(mask, conc[species])
                

            #for species in specii:           
            #    Jz[species] = np.ma.masked_where(mask, Jz[speci


            for species in ['delta']:
                #Jz_mean[species] = Jz[species].mean(axis=(1, 2))
                Jz_tot[species] = Jz[species].mean(axis=(1, 2))
                #Jz_diff_tot[species] = Jz_diff[species].sum(axis=(1, 2))
                #por_mean = por.mean(axis=(1, 2))
        
        #############################################
            z = Lz - zgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))
            #ax.set_title("Total mass per cross sectional area")


            marker = markers[it_idx]
            #color = colors[Pe_idx]
            ax.plot(z[1:-1], (Jz_tot['delta'][1:-1]), label="Pe = {} (iteration = {})".format(int(Pe__), it__), marker=marker)#, color=color)
            #ax_adv.plot(z[1:-1], (Jz_tot[species][1:-1]-Jz_diff_tot[species][1:-1]), label="{} (Pe = {})".format(name2tex[species], int(Pe__)), linewidth=2.0, linestyle='-', marker=marker, color=color)
            #ax_diff.plot(z[1:-1], (Jz_diff_tot[species][1:-1]), label="{} (Pe = {})".format(name2tex[species], int(Pe__)), linewidth=2.0, linestyle='-', marker=marker, color=color)

            #plt.yticks(fontsize=19)
            #plt.xticks(fontsize=19)
            #plt.style.use('classic')

            for _fig, _ax, filename in [(fig, ax, "total")]:
                _ax.set_xlabel(r"$z$")# , fontsize=25)
                _ax.set_ylabel(r'$\mathrm{J_{z}}$')# , fontsize=25)
                #_ax.tick_params(axis='both', labelsize=19)
                _ax.set_xlim(-0.02, 1.02)
                legend = _ax.legend(ncol=2, handlelength=1 , loc='upper center', bbox_to_anchor=(0.5, 1.35),fancybox=True, shadow=False)# , prop={'size': 16})
                #for text in legend.get_texts():
                #    text.set_fontsize(19)
                _fig.tight_layout()
                _fig.savefig(args.con + 'plots/flux/flux_delta.png', dpi=300)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.plot(z[1:-1], uz_mean[1:-1], label="Avg velocity", linewidth=2.0,marker ='s', color='k')
    ax2.set_ylabel(r'$\mathrm{u_{z}}$')# , fontsize=25)
    #plt.style.use('classic')
    #ax2.ticklabel_format(axis='both', style='sci')
    ax2.set_xlabel(r"$z$")# , fontsize=25)
    #ax2.tick_params(axis='both', labelsize=19)
    #ax2.set_xlim(-0.02, 1.02)
    #ax2.set_ylim(0.43, 0.441)
    fig2.tight_layout()
    fig2.savefig(args.con + 'plots/uz/average_uz.png', dpi=300)