 # AdDiCTIF
This repository contains an implementation of *Ad*vection-*Di*ffusion-*C*hemistry in a *T*ime-*I*ndependent *F*ramework.
The methodology is described in [our preprint](https://dx.doi.org/10.2139/ssrn.4783198). In particular, the workflow relies on the finite element method (via FEniCS) to solve steady advection-diffusion equation in arbitrary porous geometries in order to assess transverse mixing.
Assuming instantanous reaction kinetics we can then, by simple post-processing, infer the concentration of chemical species entering into various chemical reaction networks, allowing the estimation of the relevant reaction rates. We also supply several scripts to analyze the conservative and reactive dynamics.

### Installation
To install, execute:
```
pip install .
```

We can proceed with the following steps given a mesh:

### Velocity from Stokes
1. To get the Stokes velocity field, run ```addictif stokes```:
```
mpiexec -np [NUM_PROCESSES] addictif stokes --mesh path_to_the/mesh_file.h5 -o path_to_the/output_folder/ [--direction [x/y/*z*]] [--tol [tolerance]] [--sidewall_bc [noslip/*freeslip*]]
```
We assume that the HDF5 file of the mesh is stored in path_to_the_mesh_file/. Depending on the desired flow direction, you can change it.

### Conservative advection-diffusion
2. Run ```addictif ade_steady``` to solve the conservative transport to get the concentration of delta:
```
mpiexec -np [NUM_PROCESSES] addictif ade_steady -i path_to_the/output_folder/ -D [diffusion coefficient] --it [iteration 0/1/...]
```
We may define the diffusion coefficient ```D``` and the iteration number. The latter should be 0 at the first run.

3. Run ```addictif refine``` to refine given a conservative field.
```
addictif refine ...
```
4. Then go back to 2 and increase `it` by 1.

### Post-processing
3. Run with ```it = 0, 1, 2, etc.``` to have the concentration with refined mesh at the area with a high gradient of concentration.
4. If we assume we have the reaction ```a + b -> c``` with an infinite Damkholer number, we can derive the concentration of ```a```, ```b```, and ```c``` by running ```post_processing_abc.py``` :
```
mpiexec -np [NUM_PROCESSES] python3 post_processing_abc.py --mesh path_to_the_mesh_file/ --con path_where_concentration_will_be_stored/ --D 0.01 --L 0.2 --it 4
```
Note that we run the post_processing code on the results of the final iteration.
