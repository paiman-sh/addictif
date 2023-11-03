'''

#command: python plot_z_iterations.py -D 0.015 0.0015 0.00015 --con /njord2/paimans/std_adv/bead_pack/bead_pack_bigger/concentration/ --it_max 7 --mesh /njord2/paimans/std_adv/bead_pack/bead_pack_bigger/mesh/ -L 0.15

#command2: python plot_z_iterations.py -D 0.0015 --con /njord2/paimans/std_adv/berea/concentration/ --it_max 5 --mesh /njord2/paimans/std_adv/berea/mesh/ -L 0.15
'''

#import dolfin as df
from itertools import product
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def mpi_print(*args):
    if rank == 0:
        print(*args)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot mesh convergence with iterations")
    parser.add_argument("--it_max", type=int, default=0, help="Maximum iteration")
    parser.add_argument("-D", nargs='+', type=float, help='<Required> Diffusivities', required=True)
    #parser.add_argument("--D1", type=float, default=1e-2, help="Diffusion1")
    #parser.add_argument("--D2", type=float, default=1e-2, help="Diffusion2")
    #parser.add_argument("--D3", type=float, default=1e-2, help="Diffusion3")
    parser.add_argument("-L", type=float, default=1, help="Pore size")
    #parser.add_argument("-Lx", type=float, default=1, help="Lx")
    #parser.add_argument("-Ly", type=float, default=1, help="Ly")
    #parser.add_argument("-Lz", type=float, default=1, help="Lz")
    parser.add_argument("--meshfolder", type=str, default='mesh/', help="path to the mesh")
    #parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if size > 1:
        mpi_print("Please run in serial")
        exit()

    it_max = args.it_max
    it_values = list(range(it_max+1))
    
    D_ = list(reversed(sorted(args.D)))

    pore_size = args.L

    Pe_ = [pore_size / D for D in D_]

    fig_cel, ax_cel = plt.subplots(1, 1, figsize=(4, 4))
    fig_nod, ax_nod = plt.subplots(1, 1, figsize=(4, 4))
    

    markers = ['o', "v", "s"]

    cmap = plt.get_cmap("viridis")
    colab = cmap(np.linspace(0, 0.8, len(Pe_)))
    for _ax in [ax_nod, ax_cel]:
        _ax.set_prop_cycle(color=colab)

    for Pe_idx, Pe in enumerate(Pe_):
        num_nodes = []
        num_cells = []
    
        for it_idx, it in enumerate(it_values):
            h5filename = os.path.join(args.meshfolder, "mesh.h5") if it == 0 else os.path.join(args.meshfolder, "refined_mesh/mesh_Pe{}_it{}.h5".format(Pe, it))
            with h5py.File(h5filename, "r") as h5f:
                cells = h5f["mesh/topology"]
                verts = h5f["mesh/coordinates"]

                num_nodes.append(len(verts))
                num_cells.append(len(cells))

        marker = markers[Pe_idx]
        plt.xticks(it_values)
        # for 2D plot
        ax_cel.plot(it_values, num_cells, linewidth=2.0, label=r'$\alpha = {:.4f}$'.format(D_[Pe_idx]), marker=marker)
        ax_nod.plot(it_values, num_nodes, linewidth=2.0, label=r'$\alpha = {:.4f}$'.format(D_[Pe_idx]), marker=marker)
        

        #ax_nod.plot(it_values, num_nodes, linewidth=2.0, label="Pe = {}".format(int(Pe)), marker=marker)
        #ax_cel.plot(it_values, num_cells, linewidth=2.0, label="Pe = {}".format(int(Pe)), marker=marker)

    for _fig, _ax, filename in [(fig_nod, ax_nod, "nodes"), (fig_cel, ax_cel, "cells")]:
        _ax.set_xlabel("Iterations" )#, fontsize=28)
        _ax.set_ylabel('Number of ' + filename )#, fontsize=28)
        _ax.tick_params(axis='both', which='both')#, labelsize=25, width=1.5, length=7)
        _ax.set_yscale("log")
        #plt.yticks(fontsize=26)
        #plt.xticks(fontsize=26)
        legend = _ax.legend(fancybox=True, shadow=False)
        #for text in legend.get_texts():
        #    text.set_fontsize(28)

        _fig.tight_layout()
        _fig.savefig(args.con + 'plots/iterations/{}.pdf'.format(filename),dpi=300)