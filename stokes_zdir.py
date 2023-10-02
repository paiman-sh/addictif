import dolfin as df
from dolfin import *
import numpy as np
import os
from mpi4py import MPI
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Solve Stokes velocity")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--vel", type=str, default='velocity/', help="path to the velocity")
    return parser.parse_args()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class GenSubDomain(df.SubDomain):
    def __init__(self, x_min, x_max, tol=df.DOLFIN_EPS_LARGE):
        self.x_min = x_min
        self.x_max = x_max
        self.tol = tol
        super().__init__()

class Top(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] > self.x_max[2] - self.tol

class Btm(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] < self.x_min[2] + self.tol
    
class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class SideWallsX(GenSubDomain):
    def inside(self, x, on_boundary): 
        return on_boundary and bool(
            x[0] < self.x_min[0] + self.tol or x[0] > self.x_max[0] - self.tol)
            
class SideWallsY(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and bool( 
            x[1] < self.x_min[1] + self.tol or x[1] > self.x_max[1] - self.tol)

def mpi_max(x):
    x_max_loc = x.max(axis=0)
    x_max = np.zeros_like(x_max_loc)
    comm.Allreduce(x_max_loc, x_max, op=MPI.MAX)
    return x_max

def mpi_min(x):
    x_min_loc = x.min(axis=0)
    x_min = np.zeros_like(x_min_loc)
    comm.Allreduce(x_min_loc, x_min, op=MPI.MIN)
    return x_min

def mpi_print(*args):
    if rank == 0:
        print(*args)

if __name__ == "__main__":

    args = parse_args()

    # Create the mesh
    mpi_print('importing mesh')
    mesh = Mesh()
    with df.HDF5File(mesh.mpi_comm(), args.mesh+'mesh.h5', "r") as h5f_up:
        h5f_up.read(mesh, "mesh", False)
    mpi_print('mesh done')
    
    tol = 1e-2

    x = mesh.coordinates()[:]

    x_min = mpi_min(x)
    x_max = mpi_max(x)

    mpi_print(x_max, x_min)

    # Define function spaces
    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    VP = MixedElement([V, P])
    W = FunctionSpace(mesh , VP)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)

    grains = Boundary()
    sidewalls_x = SideWallsX(x_min, x_max, tol)
    sidewalls_y = SideWallsY(x_min, x_max, tol)
    top = Top(x_min, x_max, tol)
    btm = Btm(x_min, x_max, tol)

    grains.mark(subd, 3)
    sidewalls_x.mark(subd, 4)
    sidewalls_y.mark(subd, 5)
    top.mark(subd, 1)
    btm.mark(subd, 2)
 
    with XDMFFile(mesh.mpi_comm(), args.mesh+"subd.xdmf") as xdmff:
        xdmff.write(subd)

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc_porewall = DirichletBC(W.sub(0), noslip, subd, 3)
    bc_slip_x = DirichletBC(W.sub(0).sub(0), Constant(0.), subd, 4)
    bc_slip_y = DirichletBC(W.sub(0).sub(1), Constant(0.), subd, 5)
    bc_top = DirichletBC(W.sub(1), Constant(0.), subd, 1)
    bc_bottom = DirichletBC(W.sub(1), Constant(0.), subd, 2)

    # Collect boundary conditions
    bcs = [bc_porewall, bc_slip_x, bc_slip_y, bc_top, bc_bottom]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0, -1.0))
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)


    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver("minres", "hypre_amg")
    solver.parameters["monitor_convergence"] = True
    solver.parameters["relative_tolerance"] = 1e-9 


    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    mpi_print('solving')
    
    # Solve
    U = Function(W)


    solver.solve(U.vector(), bb)


    # Get sub-functions
    u, p = U.split(deepcopy=True)



    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile(args.vel+'v_show.xdmf')
    xdmffile_u.parameters["flush_output"] = True
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u)


    with df.HDF5File(mesh.mpi_comm(), args.vel+"v_hdffive.h5", "w") as h5f:
        h5f.write(u, "u")


