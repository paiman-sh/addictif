'''
#command: python3 plot/plot_z_2D_Iz.py --con /njord2/paimans/std_adv/2D/concentration/ -D 0.01 0.0025 0.001 --it 8 (eps=0.001)
'''
import dolfin as df
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dissipation rate for 2D")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("-D", nargs='+', type=float, help='<Required> Diffusivities', required=True)
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    D_ = list(reversed(sorted(args.D)))
    pore_size = args.L
    Pe = [pore_size / D for D in D_]
    it = args.it

    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff", "Iz"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cmap = plt.get_cmap("viridis")
    colab = cmap(np.linspace(0, 0.8, len(Pe)))
    ax.set_prop_cycle(color=colab)


    for Pe_idx, Pe__ in enumerate(Pe):
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

        Iz_mean = dict()
        Iz = dict()
        for species in specii:
            Iz[species] = data[species]["Iz"]    
            Iz_mean[species] = Iz[species].mean(axis=(1,2))


############################################

        # for 2D verification 
        Lz=1
        z = Lz - zgrid #Lz - np.linspace(0., Lz, len(conc_mean[specii[0]]))
        z_analytical = np.linspace(0.01, 1, 400) 
        U=1
        alpha = D_[Pe_idx]/U
        Iz_analytical = np.sqrt(2/np.pi/alpha/z_analytical)
        #plt.rcParams.update({'font.size': 30})
        for species in ["delta"]:
            plt.scatter(z, Iz_mean[species], label=r'$\alpha = {:.4f}$'.format(D_[Pe_idx]))
            plt.plot(z_analytical, Iz_analytical, linewidth=2)
            ax.set_ylabel(r'$\mathrm{I_{x}} = \langle |\nabla \delta|^2 \rangle_x$', fontsize=12)
            ax.set_xlabel("Z")#, fontsize=32)
            ax.set_yscale("log")
            ax.set_xscale("log")
            #plt.yticks(fontsize=30)
            #plt.xticks(fontsize=30)
            #ax.tick_params(axis='both', which='both', width=1.5, length=7)
            plt.gca().set_xlim([0.008,1.2])
            plt.gca().set_ylim(top=500)
            plt.tight_layout()
            plt.legend(fancybox=True, shadow=True)
            plt.savefig(args.con + 'plots/Iz/Iz_{}_vs_x_it{}_different_Pe.pdf'.format(species, it), dpi=300)
            #plt.show()
