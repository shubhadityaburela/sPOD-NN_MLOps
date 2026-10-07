import numpy as np
from matplotlib import pyplot as plt

from Plots import polar_cross_section_2D, polar_cross_section_2D_nonlinear
from Shifts import cartesian_to_polar


# for 2D without wind model
impath = "./data/FOM_2D/"
immpath = "./plots/FOM_2D/"

q = np.load(impath + 'SnapShotMatrix558.49.npy')
XY_1D = np.load(impath + '1D_Grid.npy', allow_pickle=True)
t = np.load(impath + 'Time.npy')
XY_2D = np.load(impath + '2D_Grid.npy', allow_pickle=True)

X = XY_1D[0]
Y = XY_1D[1]
X_2D = XY_2D[0]
Y_2D = XY_2D[1]

Nx = len(X)
Ny = len(Y)
Nt = len(t)

SnapShotMatrix = np.reshape(q, newshape=[Nx, Ny, 2, Nt])
T = np.reshape(SnapShotMatrix[:, :, 0, :], newshape=[Nx, Ny, 1, Nt])
S = np.reshape(SnapShotMatrix[:, :, 1, :], newshape=[Nx, Ny, 1, Nt])

# Plotting the cross-section for the 2D without wind model
polar_cross_section_2D(immpath, T, X, Y, t, var_name="T", fv=0)
polar_cross_section_2D(immpath, S, X, Y, t, var_name="S", fv=1)



# for 2D with wind model
impath = "./data/FOM_2D_ww/"
immpath = "./plots/FOM_2D_ww/"

q = np.load(impath + 'SnapShotMatrix558.49.npy')
XY_1D = np.load(impath + '1D_Grid.npy', allow_pickle=True)
t = np.load(impath + 'Time.npy')
XY_2D = np.load(impath + '2D_Grid.npy', allow_pickle=True)

X = XY_1D[0]
Y = XY_1D[1]
X_2D = XY_2D[0]
Y_2D = XY_2D[1]

Nx = len(X)
Ny = len(Y)
Nt = len(t)

SnapShotMatrix = np.reshape(np.transpose(q), newshape=[Nt, 2, Nx, Ny], order="F")
T = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 0, :, :]), newshape=[Nt, -1], order="F"))
S = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 1, :, :]), newshape=[Nt, -1], order="F"))
S = np.reshape(S, newshape=[Nx, Ny, 1, Nt], order="F")
# Plotting the cross-section for the 2D with wind model
polar_cross_section_2D_nonlinear(immpath, S, X, Y, t, var_name="S", fv=1)
