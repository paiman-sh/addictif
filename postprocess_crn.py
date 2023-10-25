import argparse
import dolfin as df
import numpy as np
from chemistry.react_1 import equilibrium_constants, compute_secondary_spec, compute_primary_spec, compute_conserved_spec, nspec

import matplotlib.pyplot as plt
import scipy.interpolate as intp
import os

from mpi4py.MPI import COMM_WORLD as comm
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Complex reaction network")
    parser.add_argument("--mesh", required=True, type=str, help="Mesh file (required)")
    parser.add_argument("--conc", required=True, type=str, help="Concentration file (required)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    c_ref = 10**-7

    # End members from the second example of de Simoni et al. WRR 2007
    c_a = np.zeros(nspec)
    c_b = np.zeros(nspec)

    c_a[0] = 3.4*10**-4 / c_ref
    c_a[1] = 10**-7.3 / c_ref

    c_b[0] = 3.4*10**-5 / c_ref
    c_b[1] = 10**-7.3  / c_ref

    K_ = equilibrium_constants(c_ref)

    compute_secondary_spec(c_a, K_)
    compute_secondary_spec(c_b, K_)
    u_a = compute_conserved_spec(c_a)
    u_b = compute_conserved_spec(c_b)

    # Solving 5th degree polynomials is expensive.
    # First we make an interpolation scheme to speed things up!

    N_intp = 1000

    alpha = np.linspace(0., 1., N_intp)
    u_ = np.outer(1-alpha, u_a) + np.outer(alpha, u_b)

    c_ = np.zeros((len(alpha), nspec))
    for i in range(len(alpha)):
        compute_primary_spec(c_[i, :], u_[i, :], K_)
        compute_secondary_spec(c_[i, :], K_)

    c_intp = [None for _ in range(nspec)]
    for ispec in range(nspec):
        c_intp[ispec] = intp.InterpolatedUnivariateSpline(alpha, c_[:, ispec])

    if False and rank == 0:
        fig, ax = plt.subplots(1, 6, figsize=(15,3))

        # 1: CO2, 2: H^+, 3: HCO3^-, 4: CO3^2-, 5: Ca^2+, 6: OH^-

        ax[0].plot(alpha, c_ref * c_[:, 0])
        ax[0].plot(alpha, c_ref * c_intp[0](alpha))
        ax[0].plot(alpha, c_ref * c_a[0]*np.ones_like(alpha))
        ax[0].plot(alpha, c_ref * c_b[0]*np.ones_like(alpha))
        ax[0].set_title("CO2")

        ax[1].plot(alpha, -np.log10(c_ref * c_[:, 1]))
        ax[1].plot(alpha, -np.log10(c_ref * c_intp[1](alpha)))
        ax[1].set_title("pH")
        
        ax[2].plot(alpha, c_ref * c_[:, 2])
        ax[2].plot(alpha, c_ref * c_intp[2](alpha))
        ax[2].set_title("HCO3^-")
        
        ax[3].plot(alpha, c_ref * c_[:, 3])
        ax[3].plot(alpha, c_ref * c_intp[3](alpha))
        ax[3].set_title("CO3^2-")
        
        ax[4].plot(alpha, c_ref * c_[:, 4])
        ax[4].plot(alpha, c_ref * c_intp[4](alpha))
        ax[4].set_title("Ca^2+")
        
        ax[5].plot(alpha, c_ref * c_[:, 5])
        ax[5].plot(alpha, c_ref * c_intp[5](alpha))
        ax[5].set_title("OH^-")

        plt.show()

    mesh = df.Mesh()

    with df.HDF5File(mesh.mpi_comm(), args.mesh, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    alpha_ = df.Function(S, name="alpha")

    with df.HDF5File(mesh.mpi_comm(), args.conc, "r") as h5f:
        h5f.read(alpha_, "delta")

    # Translate from delta (-1, 1) to alpha (0, 1)
    alpha_.vector()[:] = 0.5*(alpha_.vector()[:]+1)
    # Clip for physical reasons
    #alph = alpha_.vector()[:]
    #alpha_.vector()[alph < 0] = 0.0
    #alpha_.vector()[alph > 1] = 1.0
    # Leads to unphysical gradients!

    logalpha_ = df.Function(S, name="logalpha")
    logalpha_.vector()[:] = alpha_.vector()[:]  # np.log(alpha_.vector()[:])

    output_dir = os.path.join(os.path.dirname(args.conc), "crn")
    if rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xdmff = df.XDMFFile(mesh.mpi_comm(), os.path.join(output_dir, "c_spec.xdmf"))
    xdmff.parameters.update(dict(
        functions_share_mesh=True,
        rewrite_function_mesh=False,
        flush_output=True
    ))

    c_spec_ = [df.Function(S, name=f"c_{ispec}") for ispec in range(nspec)]

    for ispec in range(nspec):
        c_spec_[ispec].vector()[:] = c_intp[ispec](alpha_.vector()[:])
        xdmff.write(c_spec_[ispec], 0.)

    from ade_steady import helper_code
    helpers = df.compile_cpp_code(helper_code)

    params = dict(
        relative_tolerance=1e-9
    )

    dlogalphadx_ = df.project(logalpha_.dx(0), S, solver_type="cg", form_compiler_parameters=params)
    dlogalphady_ = df.project(logalpha_.dx(1), S, solver_type="cg", form_compiler_parameters=params)
    dlogalphadz_ = df.project(logalpha_.dx(2), S, solver_type="cg", form_compiler_parameters=params)

    #absgrad_alpha = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=alpha_, degree=1), S)

    gradalpha2_ = df.Function(S, name="sqGradAlpha")
    gradalpha2_.vector()[:] = dlogalphadx_.vector()[:]**2 + dlogalphady_.vector()[:]**2 + dlogalphadz_.vector()[:]**2
    #gradalpha2_.vector()[:] = absgrad_alpha.vector()[:]**2
    xdmff.write(gradalpha2_, 0.)

    R_spec_ = [df.Function(S, name=f"R_{ispec}") for ispec in range(nspec)]
    for ispec in range(nspec):
        d2c_intp = c_intp[ispec].derivative(2)
        R_spec_[ispec].vector()[:] = d2c_intp(alpha_.vector()[:]) * gradalpha2_.vector()[:]
        xdmff.write(R_spec_[ispec], 0.)

    xdmff.close()