
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot time_mesh")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    fig_, ax_ = plt.subplots(1, 1, figsize=(4, 4))

    cmap = plt.get_cmap("plasma")
    colab = cmap(np.linspace(0, 0.8, 5))
    ax_.set_prop_cycle(color=colab)


    Cores = np.array([1, 2, 4, 8, 16, 32, 64])

    #preconditioner = "hypre_euclid"
    Time0 = [86.306, 62.861, 22.271, 19.606, 7.1702, 5.2179, 3.0291]
    Time1 = [225.79, 134.73, 46.507, 18.849, 8.6826, 5.8454, 3.8234]
    Time2 = [487.67, 231.23, 106.75, 31.869, 19.363, 10.578, 6.6035]
    Time3 = [959.36, 486.9, 209.47, 111.46, 66.087, 49.355, 40.048]
    Time4 = [1313.5, 747.38, 320.25, 170.78, 99.558, 70.248, 56.481]
    '''

    #preconditioner = "petsc_amg"
    Time0 = [389.07, 221.26, 40.864, 19.736, 13.73, 3.3976, 6.1752]
    Time1 = [8491.8, 258.2, 32.313, 21.475, 7.6971, 14.946, 8.4395]
    Time2 = [1, 1704.9, 326.59, 52.474, 16.398, 11.231, 20.748]
    Time3 = [40719, 1645.4, 389.52, 81.139, 63.3, 40.116, 20.466]
    Time4 = [1, 2766.1, 220.11, 78.355, 58.41, 51.237, 87.71]
    '''
    ax_.plot(Cores, Time0, marker=">",label = 'Iteration 0')
    ax_.plot(Cores, Time1, marker='*',label = 'Iteration 1')
    ax_.plot(Cores, Time2, marker='s',label = 'Iteration 2')
    ax_.plot(Cores, Time3, marker="v",label = 'Iteration 3')
    ax_.plot(Cores, Time4, marker='o',label = 'Iteration 4')
    ax_.plot(Cores, 40./Cores, "k", label = 'Ideal')

    ax_.set_xlabel("Number of CPU cores")#, fontsize=28)
    ax_.set_ylabel('Solution time (sec)')#, fontsize=28)
    ax_.set_yscale("log")
    ax_.set_xscale("log")
    ax_.tick_params(axis='both', which='both')#, labelsize=25, width=1.5, length=7)
    ax_.legend(fancybox=True, shadow=False)
    #_ax.set_yscale("log")
    fig_.tight_layout()
    fig_.savefig(args.con + 'plots/time_cores_hypre_euclid.pdf',dpi=300)
    #fig_.savefig(args.con + 'plots/time_cores_petsc_amg.pdf',dpi=300)
