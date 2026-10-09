"""Min-max scaling of the network's inputs and outputs.

The network sees parameters and predicts outputs in [0, 1]. The constants are
fitted on the training data before training (Scaling.fit), applied to inputs and
training targets, and inverted on the predictions to get physical values back.
A trained network is only valid together with the scaling it was trained with,
so the constants are saved with the model (Scaling.save / Scaling.load).

Output layout, as in the paper: the first n_amplitudes rows are time amplitudes,
the remaining rows (if any) are shifts. All amplitudes share one min/max and all
shifts share one min/max; the parameters (rows of the input, e.g. [t, mu]) are
scaled row by row:

    x_scaled = (x - x_min) / (x_max - x_min),    x = x_min + (x_max - x_min) * x_scaled

Ported from DL-ROM/LIB/DFNN.py (scale_data, scale_params, the inverse scaling in
test_model) and Utilities.py, without their in-place updates.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "spodnn.scaling/1"


def _scale(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def _unscale(x_scaled, x_min, x_max):
    return (x_max - x_min) * x_scaled + x_min


def _bounds(x):
    """Min and max of x as floats; a constant x can't be scaled."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max == x_min:
        raise ValueError(f"cannot scale constant data (min = max = {x_min})")
    return x_min, x_max


@dataclass(frozen=True)
class Scaling:
    """Fitted min-max constants for one network.

    Attributes:
        n_amplitudes: Number of amplitude rows in the outputs; later rows are shifts.
        amplitude_min, amplitude_max: Bounds shared by all amplitude rows.
        shift_min, shift_max: Bounds shared by all shift rows, or None if the
            network predicts no shifts (POD-NN).
        param_min, param_max: Bounds per parameter row, shape (n_params,).
    """

    n_amplitudes: int
    amplitude_min: float
    amplitude_max: float
    shift_min: float | None
    shift_max: float | None
    param_min: np.ndarray
    param_max: np.ndarray

    @property
    def has_shifts(self) -> bool:
        return self.shift_min is not None

    # --- fit -------------------------------------------------------------

    @classmethod
    def fit(cls, outputs: np.ndarray, params: np.ndarray, n_amplitudes: int) -> "Scaling":
        """Fit the constants on training data.

        Args:
            outputs: Training targets, shape (n_outputs, n_samples): amplitudes
                in the first n_amplitudes rows, shifts in the rest.
            params: Training inputs, shape (n_params, n_samples), unscaled.
            n_amplitudes: Number of amplitude rows.
        """
        amplitude_min, amplitude_max = _bounds(outputs[:n_amplitudes])
        shift_min = shift_max = None
        if outputs.shape[0] > n_amplitudes:
            shift_min, shift_max = _bounds(outputs[n_amplitudes:])
        param_bounds = [_bounds(row) for row in params]
        return cls(
            n_amplitudes=n_amplitudes,
            amplitude_min=amplitude_min,
            amplitude_max=amplitude_max,
            shift_min=shift_min,
            shift_max=shift_max,
            param_min=np.array([lo for lo, _ in param_bounds]),
            param_max=np.array([hi for _, hi in param_bounds]),
        )

    # --- transform and inverse -------------------------------------------

    def scale_params(self, params: np.ndarray) -> np.ndarray:
        """Network inputs: scale each parameter row into [0, 1]. Returns a new array."""
        return _scale(params, self.param_min[:, None], self.param_max[:, None])

    def unscale_params(self, params_scaled: np.ndarray) -> np.ndarray:
        """Inverse of scale_params."""
        return _unscale(params_scaled, self.param_min[:, None], self.param_max[:, None])

    def scale_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Training targets: scale amplitudes and shifts into [0, 1]. Returns a new array."""
        return self._map_outputs(outputs, _scale)

    def unscale_outputs(self, outputs_scaled: np.ndarray) -> np.ndarray:
        """Network predictions back to physical amplitudes and shifts.

        Computed in float64 and returned in the input's dtype, as the paper did
        (the network predicts in float32).
        """
        return self._map_outputs(outputs_scaled, _unscale)

    def _map_outputs(self, outputs, fn):
        n, rows = self.n_amplitudes, outputs.shape[0]
        if (rows <= n) if self.has_shifts else (rows != n):
            expected = f"more than {n}" if self.has_shifts else str(n)
            raise ValueError(f"expected {expected} output rows for this scaling, got {rows}")
        result = np.empty(outputs.shape, dtype=np.float64)
        result[:n] = fn(outputs[:n], self.amplitude_min, self.amplitude_max)
        if self.has_shifts:
            result[n:] = fn(outputs[n:], self.shift_min, self.shift_max)
        return result.astype(outputs.dtype, copy=False)

    # --- save and load -----------------------------------------------------

    def save(self, path) -> None:
        """Write the constants as JSON (human-readable; floats round-trip exactly)."""
        shifts = None if not self.has_shifts else {"min": self.shift_min, "max": self.shift_max}
        Path(path).write_text(json.dumps({
            "format": FORMAT,
            "n_amplitudes": self.n_amplitudes,
            "amplitudes": {"min": self.amplitude_min, "max": self.amplitude_max},
            "shifts": shifts,
            "params": {"min": self.param_min.tolist(), "max": self.param_max.tolist()},
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path) -> "Scaling":
        """Read constants written by save()."""
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        shifts = data["shifts"]
        return cls(
            n_amplitudes=data["n_amplitudes"],
            amplitude_min=data["amplitudes"]["min"],
            amplitude_max=data["amplitudes"]["max"],
            shift_min=None if shifts is None else shifts["min"],
            shift_max=None if shifts is None else shifts["max"],
            param_min=np.array(data["params"]["min"]),
            param_max=np.array(data["params"]["max"]),
        )

    @classmethod
    def from_paper_npy(cls, path, n_amplitudes: int) -> "Scaling":
        """Read a scaling.npy saved by the paper code (DL-ROM run_model).

        The file is a pickled object array [amplitude_max, amplitude_min,
        shift_max, shift_min, param_max, param_min]; the paper stores the integer
        0 for both shift entries when the network has no shift outputs.
        allow_pickle can run code from the file, so only use this on trusted,
        committed files.

        Note: the paper's POD-NN files store param bounds 0 and 1 (identity),
        because its training parameters had already been scaled in place by the
        sPOD-NN run; the POD-NN was effectively trained on sPOD-scaled inputs.
        """
        a_max, a_min, s_max, s_min, p_max, p_min = np.load(path, allow_pickle=True)
        no_shifts = isinstance(s_max, int) and isinstance(s_min, int)
        return cls(
            n_amplitudes=n_amplitudes,
            amplitude_min=float(np.asarray(a_min).squeeze()),
            amplitude_max=float(np.asarray(a_max).squeeze()),
            shift_min=None if no_shifts else float(np.asarray(s_min).squeeze()),
            shift_max=None if no_shifts else float(np.asarray(s_max).squeeze()),
            param_min=np.asarray(p_min, dtype=float).ravel(),
            param_max=np.asarray(p_max, dtype=float).ravel(),
        )
