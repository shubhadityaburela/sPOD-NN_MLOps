import argparse
from time import perf_counter

import numpy as np
from Shifts import Shifts_1D, Shifts_2D
from Wildfire import Wildfire


def parse_arguments():
    p = argparse.ArgumentParser(description="Input the problem type.")
    p.add_argument("type_of_problem", type=str, choices=["1D", "2D", "2D_ww"], help="Choose the problem type")
    return p.parse_args()


args = parse_arguments()

Dimension = args.type_of_problem

if Dimension == "1D":
    Lx = 1000
    Ly = 1000
    Nx = 3000
    Ny = 1
    Nt = 6000
    step = 10
    v_x = 0
    v_y = 0
    cfl = 0.7
    c = 1.0
elif Dimension == "2D":
    Lx = 500
    Ly = 500
    Nx = 500
    Ny = 500
    Nt = 900
    step = 10
    v_x = 0
    v_y = 0
    cfl = 1.0
    c = 1.0
    edge_detect = False
elif Dimension == "2D_ww":
    Lx = 500
    Ly = 500
    Nx = 500
    Ny = 500
    Nt = 500
    step = 10
    v_x = 0.2
    v_y = 0
    cfl = 0.8
    c = 1.0
    edge_detect = True


for beta in list([540, 550, 558.49, 560, 570, 580]):
    t_1 = perf_counter()
    wf = Wildfire(
        Lxi=Lx,
        Leta=Ly,
        Nxi=Nx,
        Neta=Ny,
        timesteps=Nt,
        v_x=v_x,
        v_y=v_y,
        cfl=cfl,
        c=c,
        beta=beta,
        select_every_n_timestep=step,
    )

    wf.solver()
    t_2 = perf_counter()
    print(f"Time consumption in solving wildfire PDE : {t_2 - t_1:0.4f} seconds")
    print(wf.qs.shape, wf.qs.dtype, wf.qs.nbytes)
    print(np.abs(wf.qs.astype(np.float32) - wf.qs).max() / np.abs(wf.qs).max())

    t_3 = perf_counter()
    if Dimension == "1D":
        deltaNew = Shifts_1D(SnapShotMatrix=wf.qs, X=wf.X, t=wf.t)
    else:
        deltaNew = np.asarray(Shifts_2D(SnapShotMatrix=wf.qs, X=wf.X, Y=wf.Y, t=wf.t, edge_detect=edge_detect))
    t_4 = perf_counter()
    print(f"Time consumption in computing the shifts : {t_4 - t_3:0.4f} seconds")
    print(deltaNew.shape, deltaNew.dtype, deltaNew.nbytes)
