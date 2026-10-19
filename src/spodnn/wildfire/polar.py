"""Polar resampling and edge detection for the 2D wildfire fields.

Ported from WildlandFire_model/Shifts.py. The polar centre is given in pixels
(Nx // 2, Ny // 2) while r and theta come from the physical grid, so the two
agree only for dx = dy = 1; the config enforces that whenever edge_detect is on.
"""

import numpy as np
import polarTransform
from skimage import feature


def to_polar(fields, x, y, fill_value=0):
    """Resample each frame of fields (Nx, Ny, Nt) onto a regular (r, theta) grid.

    Returns the polar fields (N_r, N_theta, Nt) with N_r = Nx and N_theta = Ny,
    the radii r (N_r,), the angles theta (N_theta,) and the polarTransform
    settings, which can map polar images back with settings.convertToCartesianImage.
    """
    nx, ny, nt = fields.shape
    x_c, y_c = nx // 2, ny // 2  # in pixels, see the module docstring
    xx, yy = np.meshgrid(x - x_c, y - y_c)
    r_all = np.sqrt(xx**2 + yy**2)
    theta_all = np.arctan2(yy, xx)
    r = np.linspace(r_all.min(), r_all.max(), nx)
    theta = np.linspace(theta_all.min(), theta_all.max(), ny)

    polar = np.zeros((nx, ny, nt))
    settings = None
    for k in range(nt):
        image, settings = polarTransform.convertToPolarImage(
            fields[..., k],
            radiusSize=nx,
            angleSize=ny,
            initialRadius=r.min(),
            finalRadius=r.max(),
            initialAngle=theta.min(),
            finalAngle=theta.max(),
            center=(x_c, y_c),
            borderVal=fill_value,
        )
        polar[..., k] = image.T
    return polar, r, theta, settings


def detect_edges(fields, sigma=2):
    """Canny edges of each frame of fields (..., Nt), as 0/1 integers of the same shape."""
    edges = np.zeros(fields.shape, dtype=int)
    for k in range(fields.shape[-1]):
        edges[..., k] = feature.canny(fields[..., k], sigma=sigma)
    return edges
