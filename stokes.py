import dolfin as df
import numpy as np
from utils import mpi_print, mpi_max, mpi_min, Top, Btm, Boundary, SideWallsY, SideWallsZ, SideWallsX
import os
from mpi4py import MPI
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Solve Stokes velocity")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    parser.add_argument("--direction", type=str, default='z', help="x or z direction of flow")
    parser.add_argument("--tol", type=float, default=df.DOLFIN_EPS_LARGE, help="tol for subdomains")
    return parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":

    args = parse_args()

    direction = args.direction

    # Create the mesh
    mpi_print('importing mesh')
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), args.mesh+'/mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh, "mesh", False)
    mpi_print('mesh done')

    tol = args.tol

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print(x_max, x_min)

    # Define function spaces
    V = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    VP = df.MixedElement([V, P])
    W = df.FunctionSpace(mesh , VP)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)
                
    grains = Boundary()
    if direction == 'z':
        sidewalls_x = SideWallsX(x_min, x_max, tol)
    if direction == 'x':
        sidewalls_z = SideWallsZ(x_min, x_max, tol)
    sidewalls_y = SideWallsY(x_min, x_max, tol)
    top = Top(x_min, x_max, tol, direction)
    btm = Btm(x_min, x_max, tol, direction)

    grains.mark(subd, 3)
    if direction == 'z':
        sidewalls_x.mark(subd, 4)
    if direction == 'x':
        sidewalls_z.mark(subd, 4)
    sidewalls_y.mark(subd, 5)
    top.mark(subd, 1)
    btm.mark(subd, 2)
 
    with df.XDMFFile(mesh.mpi_comm(), args.mesh+"subd.xdmf") as xdmff:
        xdmff.write(subd)

    noslip = df.Constant((0.0, 0.0, 0.0))
    bc_porewall = df.DirichletBC(W.sub(0), noslip, subd, 3)
    bc_slip_y = df.DirichletBC(W.sub(0).sub(1), df.Constant(0.), subd, 5)
    bc_top = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 1)
    bc_bottom = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 2)
    if direction == 'x':
        bc_slip_z = df.DirichletBC(W.sub(0).sub(2), df.Constant(0.), subd, 4)
        bcs = [bc_porewall, bc_slip_y, bc_slip_z, bc_top, bc_bottom]
    if direction == 'z':
        bc_slip_x = df.DirichletBC(W.sub(0).sub(0), df.Constant(0.), subd, 4)
        bcs = [bc_porewall, bc_slip_x, bc_slip_y, bc_top, bc_bottom]

    if direction == 'x':
        f = df.Constant((1.0, 0.0, 0.0))
    if direction == 'z':
        f = df.Constant((0.0, 0.0, -1.0))

    # Define variational problem
    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)

    a = df.inner(df.grad(u), df.grad(v))*df.dx + df.div(v)*p*df.dx + q*df.div(u)*df.dx
    L = df.inner(f, v)*df.dx

    # Form for use in constructing preconditioner matrix
    b = df.inner(df.grad(u), df.grad(v))*df.dx + p*q*df.dx

    # Assemble system
    A, bb = df.assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = df.assemble_system(b, L, bcs)


    # Create Krylov solver and AMG preconditioner
    solver = df.KrylovSolver("minres", "hypre_amg")
    solver.parameters["monitor_convergence"] = True
    solver.parameters["relative_tolerance"] = 1e-12
    #solver.parameters["maximum_iterations"] = 13000

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    mpi_print("computing volume")
    vol = df.assemble(df.Constant(1.) * df.dx(domain=mesh))

    mpi_print('solving')
    # Solve
    U = df.Function(W)

    solver.solve(U.vector(), bb)
    mpi_print('solving done')

    # Get sub-functions
    u_, p_ = U.split(deepcopy=True)

    dir_index = 2 if direction == "z" else 0
    ui_mean = abs(df.assemble(u_[dir_index] * df.dx))/vol
    u_.vector()[:] /= ui_mean

    # Create XDMF files for visualization output
    xdmffile_u = df.XDMFFile(args.vel+'v_show.xdmf')
    xdmffile_u.parameters["flush_output"] = True
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_)

    with df.HDF5File(mesh.mpi_comm(), args.vel+"v_hdffive.h5", "w") as h5f:
        h5f.write(u_, "u")


