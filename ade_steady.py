import dolfin as df
import numpy as np
from utils import helper_code, mpi_print, mpi_sum, mpi_rank, mpi_max, mpi_min, Top, Btm, Boundary, SideWallsY, SideWallsZ, SideWallsX
import os
import h5py
import argparse
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Solve steady ADR")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--refine", type=bool, default=True, help="Do you want refinement")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    parser.add_argument("--direction", type=str, default='z', help="x or z direction of flow")
    parser.add_argument("--tol", type=float, default=df.DOLFIN_EPS_LARGE, help="tol for subdomains")
    return parser.parse_args()


# Test for PETSc or Tpetra
if not df.has_linear_algebra_backend("PETSc") and not df.has_linear_algebra_backend("Tpetra"):
    df.info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not df.has_krylov_solver_preconditioner("amg"):
    df.info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
        "preconditioner, Hypre or ML.")
    exit()

if df.has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif df.has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    df.info("Default linear algebra backend was not compiled with MINRES or TFQMR "
        "Krylov subspace method. Terminating.")
    exit()


if __name__ == "__main__":
    
    args = parse_args()
    direction = args.direction

    helpers = df.compile_cpp_code(helper_code)

    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["optimize"] = True

    D=args.D  # diffusion coefficient
    pore_size = args.L
    it = args.it 
    eps=0.01
    refine_tol=0.2

    linear_solver = "bicgstab"
    preconditioner = "hypre_euclid"

    # Create the mesh
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), args.mesh+'mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh_u, "mesh", False)

    Pe__ = pore_size / D
    mpi_print("Pe = {}".format(Pe__))

    V_u = df.VectorFunctionSpace(mesh_u, "Lagrange", 2)
    u_ = df.Function(V_u)

    with df.HDF5File(mesh_u.mpi_comm(), args.vel+"v_hdffive.h5", "r") as h5f:
        h5f.read(u_, "u")
   
    if it == 0: 
      mesh = mesh_u
    else: 
      mesh = df.Mesh()
      with df.HDF5File(mesh.mpi_comm(), args.mesh+"refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f_up:
          h5f_up.read(mesh, "mesh", False)
   
    tol = args.tol

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print(x_max, x_min)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    grains = Boundary()
    if direction == 'x':
        sidewalls_z = SideWallsZ(x_min, x_max, tol)
    if direction == 'z':
        sidewalls_x = SideWallsX(x_min, x_max, tol)
    sidewalls_y = SideWallsY(x_min, x_max, tol)
    top = Top(x_min, x_max, tol, direction)
    btm = Btm(x_min, x_max, tol, direction)

    grains.mark(subd, 3)
    if direction == 'x':
        sidewalls_z.mark(subd, 4)
    if direction == 'z':
        sidewalls_x.mark(subd, 4)
    sidewalls_y.mark(subd, 5)
    top.mark(subd, 1)
    btm.mark(subd, 2)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S_DG0 = df.FunctionSpace(mesh, "DG", 0)

    xi = df.TrialFunction(S)
    psi = df.TestFunction(S)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)
    n = df.FacetNormal(mesh)

    expr_str = "tanh((x[1]-0.5)/(sqrt(2)*eps))"
    delta_top_expr = df.Expression(expr_str, eps=eps, degree=2)
    rho_top_expr = df.Expression("1.0", degree=2)

    delta_top = df.interpolate(delta_top_expr, S)
    rho_top = df.interpolate(rho_top_expr, S)

    mpi_print("interpolating velocity field...")
    u_proj_ = df.Function(V, name="u")
    df.LagrangeInterpolator.interpolate(u_proj_, u_)
    mpi_print("done.")

    with df.XDMFFile(mesh.mpi_comm(), os.path.join(args.vel + 'velocity_proj', "u_proj.xdmf")) as xdmff:
        xdmff.write(u_proj_)

    mpi_print("interpolating norm")
    u_norm_ = df.interpolate(df.CompiledExpression(helpers.AbsVecCell(), u=u_proj_, degree=0), S_DG0)
    u_norm_.rename("u_norm", "u_norm")

    mpi_print("interpolating cell size")
    h_ = df.interpolate(df.CompiledExpression(helpers.CellSize(), mesh=mesh, degree=0), S_DG0)
    h_.rename("h", "h")


    mpi_print("computing grid Peclet")
    Pe_el_ = df.Function(S_DG0, name="Pe_el")
    Pe_el_.vector()[:] = u_norm_.vector()[:] * h_.vector()[:] / (2 * D)

    mpi_print("computing tau")
    tau_ = df.Function(S_DG0, name="tau")
    tau_.vector()[:] = h_.vector()[:] / (2 * u_norm_.vector()[:] + 1e-16)
    arr = 1. - 1. / (Pe_el_.vector()[:] + 1e-16)
    arr[arr < 0] = 0.
    #arr = 1./np.tanh(Pe_el_.vector()[:]) - 1./Pe_el_.vector()[:]
    tau_.vector()[:] *= arr
    mpi_print("done")

    indicator_ = df.Function(S_DG0, name="indicator")


    r_xi = df.dot(u_proj_, df.grad(xi)) - D * df.div(df.grad(xi))
    a_xi = df.dot(u_proj_, df.grad(xi)) * psi * df.dx \
        + D * df.dot(df.grad(psi), df.grad(xi)) * df.dx
    a_xi += tau_ * r_xi * df.dot(u_proj_, df.grad(psi)) * df.dx
    #a_xi += -D * df.dot(n, df.grad(xi)) * psi * ds(3)

    q_delta = df.Constant(0.)
    L_delta = q_delta * psi * df.dx

    delta_ = df.Function(S, name="delta")

    bc_delta_top = df.DirichletBC(S, delta_top, subd, 1)
    bcs_delta = [bc_delta_top]

    problem_delta = df.LinearVariationalProblem(a_xi,L_delta, delta_, bcs=bcs_delta)
    solver_delta = df.LinearVariationalSolver(problem_delta)


    solver_delta.parameters["linear_solver"] = linear_solver
    solver_delta.parameters["preconditioner"] = preconditioner
    solver_delta.parameters["krylov_solver"]["monitor_convergence"] = True
    solver_delta.parameters["krylov_solver"]["relative_tolerance"] = 1e-9
    solver_delta.solve()

    mpi_print('solving done')
    
    delta = df.Function(S, name="delta")
    delta.vector()[:] = delta_.vector()[:]
    mpi_print('start saving')

    with df.XDMFFile(mesh.mpi_comm(), args.con + "delta/con_show_Pe{}_it{}.xdmf".format(Pe__, it)) as xdmff:
        xdmff.parameters.update({"functions_share_mesh": True,"rewrite_function_mesh": False})
        xdmff.write(delta, 0.)
        xdmff.write(Pe_el_, 0.)
        xdmff.write(tau_, 0.)
        xdmff.write(u_norm_, 0.)
        xdmff.write(h_, 0.)

    with df.HDF5File(mesh.mpi_comm(), args.con + "delta/con_Pe{}_it{}.h5".format(Pe__, it), "w") as h5f:
        h5f.write(delta, "delta")

    mpi_print('saving done')
   
    if args.refine == True:
        mpi_print("computing absgrad")
        absgrad_delta = df.interpolate(df.CompiledExpression(helpers.AbsGrad(), a=delta, degree=0), S_DG0)
        indicator_.vector()[:] = h_.vector()[:] * absgrad_delta.vector()[:]

        # refinement
        mpi_print("marking for refinement")
        cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
        num_marked = helpers.mark_for_refinement(cell_markers, indicator_.cpp_object(), refine_tol)
        mpi_print("refining")
        new_mesh = df.refine(mesh, cell_markers)
        with df.HDF5File(new_mesh.mpi_comm(), args.mesh + "refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it+1), "w") as h5f:
            h5f.write(new_mesh, "mesh")
        mpi_print("done")

        prev_size = mpi_sum(mesh.num_cells())
        new_size = mpi_sum(new_mesh.num_cells())
        num_marked = mpi_sum(num_marked)

        if mpi_rank() == 0:
            mpi_print(
                ("Old mesh size: {}\n"
                "Marked cells:  {}\t({:.3f}%)\n"
                "New mesh size: {}\t({:.3f}x)").format(prev_size, num_marked, float(100*num_marked)/prev_size, new_size, float(new_size)/prev_size))
