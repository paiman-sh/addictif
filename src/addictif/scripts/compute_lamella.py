import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from addictif.common.utils import index2axis, axis2index, create_folder_safely, fetch_intp_data, mpi_rank, mpi_size
from scipy.interpolate import InterpolatedUnivariateSpline

def parse_args():
    parser = argparse.ArgumentParser(description="Plot imshow")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to interpolated data (e.g. intpdata.h5)")
    parser.add_argument("--direction", type=str, default=None, help="Override direction (x, y or z)")
    parser.add_argument("--show", action="store_true", help="Show plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    x, phi, u, conc, grad, prm = fetch_intp_data(args.input)

    direction = prm["direction"]
    if args.direction is not None:
        direction = axis2index[args.direction]
    x_min = np.array(prm["x_min"])
    x_max = np.array(prm["x_max"])

    if len(conc) > 0:
        D = prm["D"]

    analysisfolder = os.path.join(os.path.dirname(args.input), "Analysis")
    create_folder_safely(analysisfolder)

    imgfolder = os.path.join(os.path.dirname(args.input), "Images")
    create_folder_safely(imgfolder)

    tdims = [0, 1, 2]
    tdims.remove(direction)
    
    X, Y = np.meshgrid(x[tdims[0]], x[tdims[1]], indexing='ij')

    ds = np.linalg.norm([x[dim][1] - x[dim][0] for dim in tdims])/np.sqrt(2)

    ma = np.array(np.logical_not(phi), dtype=bool)

    vmin = dict()
    vmax = dict()
    level = dict()
    for species in conc:
        vmin[species] = conc[species].min()
        vmax[species] = conc[species].max()
        level[species] = 0.5*(vmin[species]+vmax[species])

    zvals = list(enumerate(x[direction]))

    for iz, z in zvals[mpi_rank::mpi_size]:
        ind = [slice(len(x[0])), slice(len(x[1])), slice(len(x[2]))]
        ind[direction] = iz
        ind = tuple(ind)
        for species in conc:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            pcm = ax.pcolormesh(X, Y, np.ma.masked_where(ma[ind], conc[species][ind]), shading="nearest", vmin=vmin[species], vmax=vmax[species])
            
            cs = ax.contour(X, Y, np.ma.masked_where(ma[ind], conc[species][ind]), levels=[level[species]])
            paths = cs.collections[0].get_paths()
            for path in paths:
                x_ = path.vertices
                dx_ = np.diff(x_, axis=0)
                ds_ = np.linalg.norm(dx_, axis=1)
                s_ = np.array([0, *np.cumsum(ds_)])
                
                print(s_.shape, x_[:, 0].shape)

                x_intp = [InterpolatedUnivariateSpline(s_, x_[:, i]) for i in range(2)]

                Ns = round(s_[-1]/ds)
                s = np.linspace(0, s_[-1], Ns)

                xx = [x_intp[i](s) for i in range(2)]

                # dxds_intp = [x_intp[i].derivative() for i in range(2)]

            ax.plot(xx[0], xx[1], 'r-*')
            
            ax.set_aspect("equal")
            ax.set_xlabel(f"${index2axis[tdims[0]]}$", fontsize=16)
            ax.set_ylabel(f"${index2axis[tdims[1]]}$", fontsize=16)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax.tick_params(labelsize=16)
            cbar = plt.colorbar(pcm, ax=ax, shrink=0.8) #, label=f"Concentration of ${species}$") # 
            cbar.ax.tick_params(labelsize=16) 
            cbar.ax.set_ylabel(f"${species}$", rotation=90, labelpad=20, fontsize=16)

            ax.set_title(f"${index2axis[direction]}$ = {x[direction][iz]:.2f}", fontsize=16)
            fig.tight_layout()
            #plt.savefig(os.path.join(imgfolder, f"lamella_{species}_{index2axis[direction]}_step{iz}.png"))
            #plt.close()
            plt.show()
