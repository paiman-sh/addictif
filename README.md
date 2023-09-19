# project_rock_mixing
Mixing in rocks.
This is my (Paiman's) project now ;)

### Baker's benchmark
1. Get [meshtools](https://github.com/gautelinga/meshtools) and install: ```pip install . ```
2. Run meshtools' ```examples/bakersmap.py```: ```python3 bakersmap.py baker.h5 -res 12```
3. Copy the resulting mesh to the folder ```meshes``` here.
4. Run ```stokes3d.py```: ```mpiexec -np [NUM_PROCESSES] python3 stokes3d.py```
5. Get [partrac](https://github.com/gautelinga/partrac) and compile and install with Dolfin support. Global install: ```make && sudo make install```
6. Run partrac with these settings in the folder ```output``` here: 
```
partrac ./dolfin_params.dat mode=tet dt=0.1 T=10000 init_mode=uniform_y x0=0 y0=0.5 z0=0.5 int_order=2 Nrw=1000 Nrw_max=1e7 refine=true refine_intv=1.0 stat_intv=1.0 dump_intv=1.0 verbose=true ds_max=0.02 ds_min=0.0 ds_init=0.01 Dm=0.0
```
