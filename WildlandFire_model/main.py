from Wildfire import Wildfire
from Shifts import Shifts_1D, Shifts_2D
from Plots import PlotFlow, PlotFOM2D
import time
import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)


Dimension = "2D_ww"
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
    direc_plot = "./plots/FOM_1D/"
    direc_data = "./data/FOM_1D/"
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
    direc_plot = "./plots/FOM_2D/video/"
    direc_data = "./data/FOM_2D/"
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
    direc_plot = "./plots/FOM_2D_ww/video/"
    direc_data = "./data/FOM_2D_ww/"
else:
    print("Choosing the 1D model....")
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
    direc_plot = "./plots/FOM_1D/"
    direc_data = "./data/FOM_1D/"


os.makedirs(direc_data, exist_ok=True)
os.makedirs(direc_plot, exist_ok=True)

for beta in list([540, 550, 558.49, 560, 570, 580]):

    tic = time.process_time()
    wf = Wildfire(Lxi=Lx, Leta=Ly, Nxi=Nx, Neta=Ny, timesteps=Nt, v_x=v_x, v_y=v_y,
                  cfl=cfl, c=c, beta=beta, select_every_n_timestep=step)

    wf.solver()
    toc = time.process_time()
    print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

    # %%
    # Create the Shifts for the Wildfire model. This function will only be called once and then the results will be
    # stored. (DEPENDENT on the problem setup)
    if Dimension == "1D":
        # Plot the Full Order Model (FOM)
        PlotFlow(SnapMat=wf.qs, X=wf.X, Y=wf.Y, X_2D=wf.X_2D, Y_2D=wf.Y_2D, t=wf.t, direc=direc_plot)

        deltaNew = Shifts_1D(SnapShotMatrix=wf.qs, X=wf.X, t=wf.t)
    else:
        # Plot the Full Order Model (FOM)
        PlotFOM2D(SnapMat=wf.qs, X=wf.X, Y=wf.Y, X_2D=wf.X_2D, Y_2D=wf.Y_2D, t=wf.t, directory=direc_plot,
                  plot_every=10, plot_at_all=False)

        deltaNew = Shifts_2D(SnapShotMatrix=wf.qs, X=wf.X, Y=wf.Y, t=wf.t, edge_detect=edge_detect)


    # Save the Snapshot matrix, grid and the time array
    print('Saving the matrix and the grid data')
    np.save(direc_data + 'SnapShotMatrix' + str(beta) + '.npy', wf.qs)
    np.save(direc_data + '1D_Grid.npy', [wf.X, wf.Y])
    np.save(direc_data + 'Time.npy', wf.t)
    np.save(direc_data + '2D_Grid.npy', [wf.X_2D, wf.Y_2D])
    np.save(direc_data + 'Shifts' + str(beta) + '.npy', deltaNew)
