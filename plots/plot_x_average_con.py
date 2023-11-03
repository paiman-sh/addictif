'''
#command: python plot_x_average_con.py -D 0.0015 --con /njord2/paimans/std_adv/berea/concentration/ --it 4 --L 0.15
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Plot average concentration")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("-D", nargs='+', type=float, help='<Required> Diffusivities', required=True)
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--Lx", type=float, default=1, help="Lx")
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
    it = args.it
    Lx = args.Lx

    Pe_values = [pore_size / D for D in D_]
    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff", "Iz"]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    markers = ["*",'o', "v", "s"]
    colors = ['b', 'r', 'g', 'c']

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


        conc_mean = dict()
        conc = dict()
        uz_mean = -data["uz"].mean(axis=(0,1))
        for species in specii:
            conc[species] = data[species]["conc"]

        por = np.array(np.logical_or(np.copy(conc["a"]), np.copy(conc["b"])), dtype=float)
        ma = np.array(np.logical_not(por), dtype=bool)
        
        mask = np.logical_not(np.logical_or(conc["a"], conc["b"]))
        for species in specii:
            conc[species] = np.ma.masked_where(mask, conc[species])

        for species in specii:
            conc_mean[species] = conc[species].mean(axis=(0,1))
        #por_mean = por.mean(axis=(0,1))
    #############################################
        x = xgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))
        #ax.set_title("Total concentration per cross sectional area")
        specii = ["a", "b", "c"]
        for species_idx, species in enumerate(specii):
            marker = markers[species_idx]
            color = colors[species_idx]

            ax.plot(x, conc_mean[species], label="{}".format(species), marker=marker, color=color)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Avg. concentration")
        legend = plt.legend(loc='upper right')
    #plt.show()
    plt.tight_layout()
    plt.savefig(args.con + 'plots/average_conc/conc_Pe{}_it{}_.png'.format(Pe__, it), dpi=300)

