import dolfin as df
import os

if __name__ == "__main__":
    mesh = df.UnitCubeMesh(10, 10, 10)

    filedir = os.path.dirname(__file__)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(filedir, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")