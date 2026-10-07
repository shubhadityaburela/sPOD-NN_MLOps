import numpy as np
import matplotlib;
from matplotlib.ticker import FormatStrFormatter

from Shifts import cartesian_to_polar, edge_detection

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import moviepy.video.io.ImageSequenceClip
import glob
from math import pi

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


class PlotFlow:
    def __init__(self, SnapMat, X, Y, X_2D, Y_2D, t, direc) -> None:

        self.__Nx = int(np.size(X))
        self.__Ny = int(np.size(Y))
        self.__Nt = int(np.size(t))

        # Prepare the space-time grid
        [self.__X_grid, self.__t_grid] = np.meshgrid(X, t)
        self.__X_grid = self.__X_grid.T
        self.__t_grid = self.__t_grid.T

        self.__X_2D = X_2D
        self.__Y_2D = Y_2D

        immpath = direc
        os.makedirs(immpath, exist_ok=True)

        T = SnapMat[:self.__Nx, :]
        S = SnapMat[self.__Nx:, :]

        # Plot the snapshot matrix for conserved variables for original model
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(self.__X_grid, self.__t_grid, T, cmap='YlOrRd')
        ax1.axis('off')
        ax1.axis('auto')
        ax1.set_title(r"$T(x, t)$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(self.__X_grid, self.__t_grid, S, cmap='YlGn')
        ax2.axis('off')
        ax2.axis('auto')
        ax2.set_title(r"$S(x, t)$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        fig.supylabel(r"time $t$")
        fig.supxlabel(r"space $x$")

        save_fig(filepath=immpath + 'Variable', figure=fig)

        print('All the plots for the ORIGINAL MODEL saved')


def PlotFOM2D(SnapMat, X, Y, X_2D, Y_2D, t, directory, plot_every=9, plot_at_all=False):
    Nx = int(np.size(X))
    Ny = int(np.size(Y))
    Nt = int(np.size(t))
    SnapMat = np.reshape(np.transpose(SnapMat), newshape=[Nt, 2, Nx, Ny], order="F")

    if plot_at_all:
        immpath = directory
        os.makedirs(immpath, exist_ok=True)
        for n in range(Nt):
            if n % plot_every == 0:
                min_T = np.min(SnapMat[n, 0, :, :])
                max_T = np.max(SnapMat[n, 0, :, :])
                min_S = np.min(SnapMat[n, 1, :, :])
                max_S = np.max(SnapMat[n, 1, :, :])

                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                im1 = ax1.pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 0, :, :]), vmin=min_T, vmax=max_T,
                                     cmap='YlOrRd')
                ax1.axis('scaled')
                ax1.set_title(r"$T(x, y)$")
                ax1.set_yticks([], [])
                ax1.set_xticks([], [])
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='10%', pad=0.08)
                fig.colorbar(im1, cax=cax, orientation='vertical')

                ax2 = fig.add_subplot(122)
                im2 = ax2.pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 1, :, :]), vmin=min_S, vmax=max_S,
                                     cmap='YlGn')
                ax2.axis('scaled')
                ax2.set_title(r"$S(x, y)$")
                ax2.set_yticks([], [])
                ax2.set_xticks([], [])
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size='10%', pad=0.08)
                fig.colorbar(im2, cax=cax, orientation='vertical')

                fig.supylabel(r"space $y$")
                fig.supxlabel(r"space $x$")

                fig.savefig(immpath + "Var" + str(n), dpi=300, transparent=True)
                fig.savefig(immpath + "Var" + str(n) + ".pdf", format="pdf", bbox_inches="tight", transparent=True)
                plt.close(fig)

        fps = 1
        image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(immpath + "Var_2D" + '.mp4')
    else:
        pass



def polar_cross_section_2D(impath, q, X, Y, t, var_name, fv):
    q_polar, theta_i, r_i, _ = cartesian_to_polar(q, X, Y, t, fill_val=fv)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    if var_name == "T":
        cmap = "YlOrRd"
        ymin = 0
        ymax = 900
        offset = 300
        xy_1 = (175, 800)
        xytext_1 = (-2, 790)
        xy_2 = (255, 600)
        xytext_2 = (135, 590)
    else:
        cmap = "YlGn"
        ymin = 0
        ymax = 1
        offset = 0
        xy_1 = (175, 0.8)
        xytext_1 = (-2, 0.79)
        xy_2 = (255, 0.6)
        xytext_2 = (135, 0.590)


    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    (subfig_t, subfig_b) = fig.subfigures(1, 2, hspace=0.05, wspace=0.1)

    # put 3 axis in the top subfigure
    gs_t = subfig_t.add_gridspec(nrows=4, ncols=1)
    ax1 = subfig_t.add_subplot(gs_t[0:4, 0])

    min = np.min(q[..., 0, -1])
    max = np.max(q[..., 0, -1])
    ax1.plot(X, np.squeeze(q[:, Ny // 2, 0, 29]), color="green", linestyle="-", label=r"$t=t_1$")
    ax1.plot(X, np.squeeze(q[:, Ny // 2, 0, -1]), color="green", linestyle="-.", label=r"$t=t_2$")
    ax1.vlines(x=250, ymin=ymin, ymax=ymax, colors='black', linestyles="--")
    ax1.annotate(r"$\Delta$", xy=xy_1,
                 xytext=xytext_1,
                 xycoords='data',
                 textcoords='data',
                 arrowprops=dict(arrowstyle='<|-|>',
                                 color='blue',
                                 lw=2.5,
                                 ls='--'),
                 # fontsize=18
                 )
    ax1.annotate(r"$R$", xy=xy_2,
                 xytext=xytext_2,
                 xycoords='data',
                 textcoords='data',
                 arrowprops=dict(arrowstyle='<|-|>',
                                 color='red',
                                 lw=2.5,
                                 ls='--'),
                 # fontsize=18
                 )
    ax1.set_ylim(bottom=min, top=max + offset)
    # ax1.axis('auto')
    ax1.legend()
    ax1.grid()

    subfig_t.supylabel(r"$" + var_name + "$")
    subfig_t.supxlabel(r"space $x$")

    gs_b = subfig_b.add_gridspec(nrows=4, ncols=1)
    ax4 = subfig_b.add_subplot(gs_b[0:2, 0])
    ax5 = subfig_b.add_subplot(gs_b[2:4, 0], sharex=ax4, sharey=ax4)

    ax5.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, 29]), cmap=cmap)
    ax5.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax5.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    ax4.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, -1]), cmap=cmap)
    ax4.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax4.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    subfig_b.supylabel(r"$\mathrm{R}$")
    subfig_b.supxlabel(r"$\theta$(rad)")

    fig.savefig(impath + 'polar_cs_' + var_name, dpi=300, transparent=True)
    fig.savefig(impath + 'polar_cs_' + var_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)




def polar_cross_section_2D_nonlinear(impath, q, X, Y, t, var_name, fv):
    q_polar, theta_i, r_i, _ = cartesian_to_polar(q, X, Y, t, fill_val=fv)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    cmap = "YlGn"
    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)

    # Perform edge detection
    edge = edge_detection(q=q_polar)
    refvalue_front = np.amax(edge[..., 0, -1] * r_grid, axis=0)
    is_zero = np.where(refvalue_front != 0)[0]
    if np.any(is_zero):
        refvalue_front = np.interp(x=theta_i, xp=theta_i[is_zero], fp=refvalue_front[is_zero])
    front = np.amax(edge[..., 0, 19] * r_grid, axis=0)
    is_zero = np.where(front != 0)[0]
    if np.any(is_zero):
        front = np.interp(x=theta_i, xp=theta_i[is_zero], fp=front[is_zero])

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    (subfig_t, subfig_b) = fig.subfigures(1, 2, hspace=0.05, wspace=0.1)

    # put 3 axis in the top subfigure
    gs_t = subfig_t.add_gridspec(nrows=4, ncols=1)
    ax1 = subfig_t.add_subplot(gs_t[0:4, 0])

    ax1.plot(theta_i, front, color="green", linestyle="-", label=r"$t=t'_1$")
    ax1.plot(theta_i, refvalue_front, color="orange", linestyle="-.", label=r"$t=t'_2$")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.annotate(r"$\Delta^{\theta_1}$", xy=(1.6, 100),
                 xytext=(1.35, 240),
                 xycoords='data',
                 textcoords='data',
                 arrowprops=dict(arrowstyle='<|-|>',
                                 color='blue',
                                 lw=2.0,
                                 ls='--')
                 )
    ax1.annotate(r"$\Delta^{\theta_2}$", xy=(1.0, 84),
                 xytext=(0.8, 197),
                 xycoords='data',
                 textcoords='data',
                 arrowprops=dict(arrowstyle='<|-|>',
                                 color='red',
                                 lw=2.0,
                                 ls='--')
                 )
    ax1.legend()
    ax1.grid()

    subfig_t.supylabel(r"$\mathrm{R}$")
    subfig_t.supxlabel(r"$\theta$(rad)")

    gs_b = subfig_b.add_gridspec(nrows=4, ncols=1)
    ax4 = subfig_b.add_subplot(gs_b[0:2, 0])
    ax5 = subfig_b.add_subplot(gs_b[2:4, 0], sharex=ax4, sharey=ax4)

    ax5.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, 19]), cmap=cmap)
    ax5.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax5.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    ax4.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, -1]), cmap=cmap)
    ax4.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax4.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    subfig_b.supylabel(r"$\mathrm{R}$")
    subfig_b.supxlabel(r"$\theta$(rad)")

    fig.savefig(impath + 'polar_cs_nl_' + var_name, dpi=300, transparent=True)
    fig.savefig(impath + 'polar_cs_nl_' + var_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)