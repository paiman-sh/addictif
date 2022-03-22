import dolfin as df
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
import os

class PBC(df.SubDomain):
    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        super().__init__()

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0) and on_boundary)

    def map(self, x, y):
        if df.near(x[0], 5*self.Lx):
            y[0] = x[0] - 5*self.Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0] - 10000.
            y[1] = x[1] - 10000.
            y[2] = x[2] - 10000.

class Walls(df.SubDomain):
    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        super().__init__()

    def inside(self, x, on_boundary):
        return bool(on_boundary and 
                    (x[1] < df.DOLFIN_EPS_LARGE or x[1] > self.Ly - df.DOLFIN_EPS_LARGE or
                     x[2] < df.DOLFIN_EPS_LARGE or x[2] > self.Lz - df.DOLFIN_EPS_LARGE or
                     (x[0] > df.DOLFIN_EPS_LARGE and x[0] < 5*self.Lx - df.DOLFIN_EPS_LARGE)))

class Inlet(df.SubDomain):
    def __init__(self):
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and df.near(x[0], 0)

def mesh(**params):
    mesh = df.Mesh()
    fname = "meshes/baker.h5"
    with df.HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0 and not os.path.exists("output"):
        os.makedirs("output")

    Lx = 1
    Ly = 1
    Lz = 1
    fx = 1.0
    fy = 0.0
    fz = 0.0

    res = 0.02
    k = 1
    relative_tolerance = 1e-9

    mesh = mesh()
    dim = mesh.topology().dim()

    pbc = PBC(Lx, Ly, Lz)
    wall = Walls(Lx, Ly, Lz)

    subd = df.MeshFunction("size_t", mesh, dim-1)
    subd.set_all(0)
    wall.mark(subd, 1)
    pbc.mark(subd, 2)

    with df.XDMFFile(mesh.mpi_comm(), "output/subd3d.xdmf") as xdmff:
        xdmff.write(subd)

    f0 = df.Constant((fx, fy, fz))

    # Flow (to be split?)
    U_el = df.VectorElement("Lagrange", mesh.ufl_cell(), k+1)
    P_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), k)
    UP_el = df.MixedElement([U_el, P_el])
    UP = df.FunctionSpace(mesh, UP_el, constrained_domain=pbc)

    up_ = df.Function(UP)
    u_, p_ = df.split(up_)
    up = df.TrialFunction(UP)
    u, p = df.TrialFunctions(UP)
    v, q = df.TestFunctions(UP)

    a = df.inner(df.sym(df.grad(u)), df.sym(df.grad(v))) * df.dx() \
        - df.inner(p, df.div(v)) * df.dx() - df.inner(q, df.div(u)) * df.dx()

    L = df.dot(f0, v) * df.dx()

    bcu_wall = df.DirichletBC(UP.sub(0), df.Constant((0.,)*dim), subd, 1)
    bcs_up = [bcu_wall]

    A = df.assemble(a)
    b = df.assemble(L)
    for bc in bcs_up:
        bc.apply(A)
        bc.apply(b, up_.vector())

    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    
    df.PETScOptions.set('ksp_view')
    df.PETScOptions.set('ksp_monitor_true_residual')
    df.PETScOptions.set('pc_type', 'fieldsplit')
    df.PETScOptions.set('ksp_rtol', relative_tolerance)

    df.PETScOptions.set('pc_fieldsplit_type', 'additive')
    df.PETScOptions.set('pc_fieldsplit_detect_saddle_point')
    df.PETScOptions.set('fieldsplit_u_ksp_type', 'preonly')
    df.PETScOptions.set('fieldsplit_u_pc_type', 'lu')
    df.PETScOptions.set('fieldsplit_p_ksp_type', 'preonly')
    df.PETScOptions.set('fieldsplit_p_pc_type', 'jacobi')
    
    """
    df.PETScOptions.set('ksp_error_if_not_converged')
    df.PETScOptions.set('ksp_monitor_true_residual')
    df.PETScOptions.set('ksp_atol', 1e-10) 
    df.PETScOptions.set("pc_use_amat")
    df.PETScOptions.set("pc_type", "fieldsplit")
    df.PETScOptions.set("pc_fieldsplit_type", "schur")
    df.PETScOptions.set("pc_fieldsplit_schur_fact_type", "full")
    df.PETScOptions.set("pc_fieldsplit_schur_precondition", "a11")
    df.PETScOptions.set("pc_fieldsplit_off_diag_use_amat")
    df.PETScOptions.set("fieldsplit_velocity_pc_type", "lu")
    df.PETScOptions.set("fieldsplit_pressure_ksp_rtol", 1e-10)
    df.PETScOptions.set("fieldsplit_pressure_pc_type", "lu")
    """

    ksp.setFromOptions()
    if rank == 0:
        print('Solving with:', ksp.getType())

    A_ = df.as_backend_type(A).mat()
    b_ = df.as_backend_type(b).vec()

    ksp.setOperators(A_)

    pc = ksp.getPC()
    is0 = PETSc.IS().createGeneral(UP.sub(0).dofmap().dofs())
    is1 = PETSc.IS().createGeneral(UP.sub(1).dofmap().dofs())
    fields = [('u', is0), ('p', is1)]
    pc.setFieldSplitIS(*fields)

    ksp.setUp()

    x_, _ = A_.createVecs()

    xdmf_params = dict(
        functions_share_mesh=True,
        rewrite_function_mesh=False,
        flush_output=True
    )

    xdmff = df.XDMFFile(mesh.mpi_comm(), "output/fields3d.xdmf")
    xdmff.parameters.update(xdmf_params)

    if rank == 0:
        print("Solving...")
    ksp.solve(b_, x_)
    up_.vector()[:] = x_
    for bc in bcs_up:
        bc.apply(up_.vector())

    u__, p__ = up_.split(deepcopy=True)
    u__.rename("u", "tmp")
    p__.rename("p", "tmp")

    xdmff.write(u__, 0)
    xdmff.write(p__, 0)

    with df.HDF5File(mesh.mpi_comm(), "output/mesh.h5", "w") as h5f:
        h5f.write(mesh, "mesh")

    fname = "up_0.h5"
    with df.HDF5File(mesh.mpi_comm(), "output/{}".format(fname), "w") as h5f:
        h5f.write(u__, "u")
        h5f.write(p__, "p")

    if rank == 0:
        with open(os.path.join("output", "timestamps.dat"), "w") as ofile:
            for t in [0, 1000000]:
                ofile.write("{} {}\n".format(t, fname))
            
        with open(os.path.join("output", "dolfin_params.dat"), "w") as ofile:
            txt = """velocity_space=P{}
pressure_space=P{}
timestamps=timestamps.dat
mesh=mesh.h5
periodic_x=true
periodic_y=false
periodic_z=false
rho=1.0""".format(k+1, k)
            ofile.write(txt)

    inlet = Inlet()
    inlet.mark(subd, 3)
    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    flux_inlet = df.assemble(u__[0] * ds(3))
    flux_vol = df.assemble(u__[0] * df.dx())
    if rank == 0:
        print("{} should be close to {}".format(flux_inlet, flux_vol/(5*Lx)))