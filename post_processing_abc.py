import argparse
import numpy as np
import dolfin as df
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(description="Post-processing")
    parser.add_argument("--it", type=int, default=0, help="Iteration")
    parser.add_argument("--D", type=float, default=1e-2, help="Diffusion")
    parser.add_argument("--L", type=float, default=1, help="Pore size")
    parser.add_argument("--mesh", type=str, default='mesh/', help="path to the mesh")
    parser.add_argument("--con", type=str, default='concentration/', help="path to the concentration")
    return parser.parse_args()

def mpi_print(*args):
    if rank == 0:
        print(*args)

if __name__ == "__main__":
    args = parse_args()
    
    D=args.D
    pore_size = args.L
    it = args.it
    Pe__ = pore_size / D


    print("loading mesh")
    mesh_u = df.Mesh()
    with df.HDF5File(mesh_u.mpi_comm(), args.mesh+'mesh.h5', "r") as h5f_mesh:
        h5f_mesh.read(mesh_u, "mesh", False)

    if it == 0: 
      mesh = mesh_u
    else: 
      mesh = df.Mesh()
      with df.HDF5File(mesh.mpi_comm(), args.mesh+"refined_mesh/mesh_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f_up:
          h5f_up.read(mesh, "mesh", False)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    conc_ = df.Function(S, name="delta")
    conc_.vector()[:] = conc_.vector()[:]

    print("loading delta")

    with df.HDF5File(mesh.mpi_comm(), args.con + "delta/con_Pe{}_it{}.h5".format(Pe__, it), "r") as h5f:
        h5f.read(conc_, 'delta')

    a_ = df.Function(S, name= "a")
    b_ = df.Function(S, name="b") 
    c_ = df.Function(S, name="c") 

    a_.vector()[:] = np.maximum(0, conc_.vector()[:])
    b_.vector()[:] = np.maximum(0, -conc_.vector()[:])
    c_.vector()[:] = (1 - abs(conc_.vector()[:]))/2

    mpi_print("Saving abc")

    with df.XDMFFile(mesh.mpi_comm(), args.con + "abc/con_show_Pe{}_it{}.xdmf".format(Pe__, it)) as xdmff:
        xdmff.parameters.update({"functions_share_mesh": True,"rewrite_function_mesh": False})
        xdmff.write(conc_, 0.)
        xdmff.write(a_, 0.)
        xdmff.write(b_, 0.)
        xdmff.write(c_, 0.)


    with df.HDF5File(mesh.mpi_comm(), args.con + "abc/con_Pe{}_it{}.h5".format(Pe__, it), "w") as h5f:
        h5f.write(conc_, "delta")
        h5f.write(a_, "a")
        h5f.write(b_, "b")
        h5f.write(c_, "c")