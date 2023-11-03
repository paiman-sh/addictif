'''
#command: python3 plot/plot_z_2D_delta.py --con /njord2/paimans/std_adv/2D/concentration/ --D 0.01 --it 8 (eps=0.001)
'''
import dolfin as df
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady ADR")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    D=args.D
    pore_size = args.L
    it = args.it

    eps = 1e-8

    Pe__ = pore_size / D
    specii = ["a", "b", "c", 'delta']
    fields = ["conc", "J_diff", "Iz"]

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
    conc_mean_x_axis = dict()  #for 2D case
    conc = dict()
    Iz_mean = dict()
    Iz = dict()
    uz_mean = -data["uz"].mean(axis=(1, 2))
    for species in specii:
        conc[species] = data[species]["conc"]
        
        Iz[species] = data[species]["Iz"]
        conc_mean[species] = conc[species].mean(axis=(1, 2))
       
        conc_mean_x_axis[species] = conc[species].mean(axis=(2))  #for 2D case
        
        Iz_mean[species] = Iz[species].mean(axis=(1,2))


###########################################

################################################ for paper1
    # for 2D verification
    x = 1 - np.linspace(0., 1, len(conc_mean_x_axis[specii[0]][1]))
    x_analytical = np.linspace(0, 1, 400) 
    def erffunc(x,x0,alpha,z,c1,c2):
        return 0.5*(c1-c2)*erf((x-x0)/np.sqrt(4 * alpha * z))
    U=1
    alpha = D/U

    skip = 5
    skip2 = 7
    for species in ['delta']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        cmap = plt.get_cmap('viridis')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plt.rcParams.update({'font.size': 30})
        plt.grid(True)
        for iz in range(conc_mean_x_axis[species].shape[0])[::skip]:
            color = cmap(iz /  conc_mean_x_axis[species].shape[0])
            for ii in range(len(conc_mean_x_axis[specii[0]][1])):
                if ii%skip2 == 0:
                    ax.scatter(x[ii], conc_mean_x_axis[species][iz, :].flatten()[ii], color=color)

            delta_analytical = erffunc(x_analytical,0.5,alpha,1-zgrid[iz],-1,1)
            plt.plot(x_analytical, delta_analytical, label="z={:.2f}".format(1-zgrid[iz]), color=color)
            
        #ax.set_yscale("log")
        #ax.set_xscale("log")
        ax.set_xlabel("X", fontsize=30)
        ax.set_ylabel("{}".format(r'$\delta$'), fontsize=32)
        #plt.legend(labels,bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3,fancybox=True, shadow=True, prop={'size': 18})
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1],loc='upper right', ncol=1,fancybox=True, shadow=True, prop={'size': 30})
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.show()
        plt.ylim(-1.02,1.02)
        plt.xlim(0,1)
        plt.tight_layout()
        plt.savefig(args.con + 'plots/delta/{}_vs_x_Pe={}_it{}.png'.format(species, Pe__, it), dpi=300)



