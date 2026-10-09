"""Capture golden values from the paper code for the synthetic case.

Runs the pipeline of paper/synthetic.ipynb headlessly: the pinned trained weights
are loaded (no training) and no figures are drawn. Every error the paper code
reports is written to a JSON file, together with the environment it ran in.

Run it with the original paper environment (Python 3.9, numpy 1.22): paper/ is a
frozen reference and is not patched for Python 3.12, so under the project
environment it stops inside paper/Helper.py by design. The glue code here still
avoids version-specific numpy calls, uses only the standard library and numpy,
and never imports spodnn.

Usage, from the repository root:
    /opt/homebrew/Caskroom/miniforge/base/envs/PyTorch/bin/python \
        scripts/make_golden.py --out tests/golden/synthetic.json
"""

import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"


def setup_paper_imports():
    """Make the paper code importable the same way the notebooks do."""
    os.environ["MPLBACKEND"] = "Agg"  # draw nothing on screen; no display needed
    os.chdir(PAPER)  # the paper code uses paths relative to paper/
    sys.path[:0] = [str(PAPER), str(REPO / "sPOD" / "src" / "spod"), str(REPO / "DL-ROM" / "LIB")]


def fingerprint(a):
    """A few numbers that summarise an array, so an intermediate result can be
    checked without storing the whole array."""
    a = np.asarray(a, dtype=float)
    return {
        "shape": list(a.shape),
        "norm": float(np.linalg.norm(a)),
        "sum": float(a.sum()),
        "abs_max": float(np.abs(a).max()),
        "abs_mean": float(np.abs(a).mean()),
    }


def amplitude_fingerprint(ta):
    """Sign-safe summary of a (modes x samples) amplitude matrix.

    An SVD mode is only defined up to sign (u and -u are equally valid), so the
    amplitudes are summarised by row norms, which don't depend on the sign. The
    signs themselves are recorded separately as `orientation` (the sign of each
    row's largest entry): the trained networks only fit the original orientation.
    """
    ta = np.asarray(ta, dtype=float)
    largest = np.argmax(np.abs(ta), axis=1)
    return {
        "shape": list(ta.shape),
        "row_norms": [float(v) for v in np.linalg.norm(ta, axis=1)],
        "orientation": [int(np.sign(ta[i, j])) for i, j in enumerate(largest)],
    }


def run_synthetic():
    """Run the synthetic pipeline of paper/synthetic.ipynb and return its results.

    Returns a dict with:
        "test_model_rel_err_sPOD": float, the rel_err returned by test_model for sPOD-NN
        "test_model_rel_err_POD":  float, the rel_err returned by test_model for POD-NN
        "errors":     the list of 3 arrays returned by OnlinePredictionAnalysis
        "errors_inf": the list of 3 arrays returned by OnlinePredictionAnalysis
        "intermediates": fingerprints of intermediate results, grouped by pipeline stage
    """

    import synthetic_sup as sup_module
    from DFNN import scale_params, test_model
    from synthetic_sup import synthetic_sup

    # fingerprint: spy on the sPOD-I helpers. synthetic_sup did `from Helper import *`,
    # so it looks these names up in its own namespace at call time; replacing them there
    # with a wrapper records their outputs without changing what they compute.
    spied = {}

    def spy_on(name):
        original = getattr(sup_module, name)

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            spied[name] = out
            return out

        setattr(sup_module, name, wrapper)

    spy_on("my_delta_interpolate")
    spy_on("my_interpolated_state")

    # Run the sPOD offline phase
    df = synthetic_sup(
        training_samples=[0.1, 0.15, 0.2, 0.25, 0.3],
        testing_sample=[0.23],
        nmodes=8,
        spod_iter=300,
        plot_offline_data=False,
    )

    # fingerprint: data generation and decomposition, before anything else touches df
    D = df.D
    intermediates = {
        "data": {
            "q_train": fingerprint(df.q_train),
            "q_test": fingerprint(df.q_test),
            "q1_test": fingerprint(df.q1_test),
            "q2_test": fingerprint(df.q2_test),
            "shifts_train_frame1": fingerprint(df.shifts_train[0]),
            "shifts_train_frame2": fingerprint(df.shifts_train[1]),
            "shifts_test_frame1": fingerprint(df.shifts_test[0]),
            "shifts_test_frame2": fingerprint(df.shifts_test[1]),
            "params_train": fingerprint(df.params_train),
            "params_test": fingerprint(df.params_test),  # copied now: scale_params scales it in place later
        },
        "decomposition": {
            "pod_reconstruction": fingerprint(df.U_POD_TRAIN @ df.TA_POD_TRAIN),
            "pod_amplitudes_train": amplitude_fingerprint(df.TA_POD_TRAIN),
            "pod_amplitudes_test": amplitude_fingerprint(df.TA_POD_TEST),
        },
    }
    stage = intermediates["decomposition"]
    for k in range(df.NumFrames):
        amplitudes = df.TA_TRAIN[k * D : (k + 1) * D]
        stage["frame%d_reconstruction" % (k + 1)] = fingerprint(df.U_list[k] @ amplitudes)
        stage["frame%d_amplitudes_train" % (k + 1)] = amplitude_fingerprint(amplitudes)
        stage["frame%d_amplitudes_test" % (k + 1)] = amplitude_fingerprint(df.TA_TEST[k * D : (k + 1) * D])

    # We collect the time amplitudes, shifts and the parameters for the training as well as the testing data.
    TA_TRAIN = df.TA_TRAIN
    SHIFTS_TRAIN = df.SHIFTS_TRAIN
    TA_TEST = df.TA_TEST
    SHIFTS_TEST = df.SHIFTS_TEST
    PARAMS_TEST = df.PARAMS_TEST
    TA_POD_TRAIN = df.TA_POD_TRAIN
    TA_POD_TEST = df.TA_POD_TEST

    # Assemble the matrices for the training and testing data
    # The shape is passed positionally: the `newshape=` keyword used in the notebook
    # was removed in numpy 2.x, while the positional form works in every version.
    shifts_train = np.concatenate((np.reshape(SHIFTS_TRAIN[0], [1, -1]), np.reshape(SHIFTS_TRAIN[1], [1, -1])), axis=0)
    shifts_test = np.concatenate((np.reshape(SHIFTS_TEST[0], [1, -1]), np.reshape(SHIFTS_TEST[1], [1, -1])), axis=0)

    ta_train = np.concatenate((TA_TRAIN, shifts_train), axis=0)
    ta_test = np.concatenate((TA_TEST, shifts_test), axis=0)

    # Prepare the parameters:
    params_sPOD = {
        "scaling": True,
        "full_order_model_dimension": df.Nx,
        "reduced_order_model_dimension": ta_train.shape[0],
        "totalModes": ta_train.shape[0] - df.NumFrames,
        "num_early_stop": 100000,
    }
    params_POD = {
        "scaling": True,
        "full_order_model_dimension": df.Nx,
        "reduced_order_model_dimension": TA_POD_TRAIN.shape[0],
        "totalModes": TA_POD_TRAIN.shape[0],
        "num_early_stop": 500,
    }

    # Testing the trained model
    test = {"typeOfTest": "full", "test_sample": 200}
    log_folder_base_sPOD = "DNN_result/synthetic/training_results_sPOD/"
    log_folder_trained_model_sPOD = sorted(Path(log_folder_base_sPOD).glob("*/"), key=os.path.getmtime)[-1]
    PATH_sPOD = str(log_folder_trained_model_sPOD) + "/trained_weights/" + "weights.pt"

    log_folder_base_POD = "DNN_result/synthetic/training_results_POD/"
    log_folder_trained_model_POD = sorted(Path(log_folder_base_POD).glob("*/"), key=os.path.getmtime)[-1]
    PATH_POD = str(log_folder_trained_model_POD) + "/trained_weights/" + "weights.pt"

    PATH_sPOD = "DNN_result/synthetic/training_results_sPOD/2025_06_26__13-27-59/trained_weights/weights.pt"
    PATH_POD = "DNN_result/synthetic/training_results_POD/2025_06_26__14-24-05/trained_weights/weights.pt"

    # Scale the parameters before prediction
    if "/trained_weights/weights.pt" in PATH_sPOD:
        address_sPOD = PATH_sPOD.replace("/trained_weights/weights.pt", "")
    scaling_sPOD = np.load(address_sPOD + "/variables/" + "scaling.npy", allow_pickle=True)

    if "/trained_weights/weights.pt" in PATH_POD:
        address_POD = PATH_POD.replace("/trained_weights/weights.pt", "")
    scaling_POD = np.load(address_POD + "/variables/" + "scaling.npy", allow_pickle=True)

    PARAMS_TEST_sPOD = scale_params(PARAMS_TEST, params_sPOD, scaling_sPOD)
    PARAMS_TEST_POD = scale_params(PARAMS_TEST, params_POD, scaling_POD)

    rel_err_sPOD, results_predicted_sPOD = test_model(
        ta_test,
        PARAMS_TEST_sPOD,
        saved_model=True,
        PATH_TO_WEIGHTS=PATH_sPOD,
        params=params_sPOD,
        scaling=scaling_sPOD,
        batch_size=50,
    )
    rel_err_POD, results_predicted_POD = test_model(
        TA_POD_TEST,
        PARAMS_TEST_POD,
        saved_model=True,
        PATH_TO_WEIGHTS=PATH_POD,
        params=params_POD,
        scaling=scaling_POD,
        batch_size=50,
    )

    TA_sPOD_pred = results_predicted_sPOD[:-2, :]
    shifts_sPOD_pred = results_predicted_sPOD[-2:, :]
    TA_POD_pred = results_predicted_POD

    # fingerprint: network inputs (after the in-place scaling) and raw network outputs
    intermediates["network"] = {
        "params_test_scaled": fingerprint(PARAMS_TEST),
        "prediction_sPOD": fingerprint(results_predicted_sPOD),
        "prediction_POD": fingerprint(results_predicted_POD),
    }

    errors, errors_inf = df.OnlinePredictionAnalysis(
        TA_sPOD_pred, shifts_sPOD_pred, TA_POD_pred, plot_online=False, test_type=test
    )

    # fingerprint: sPOD-I outputs captured by the spies during OnlinePredictionAnalysis
    shifts_interp = spied["my_delta_interpolate"]
    q_interp, amplitudes_interp = spied["my_interpolated_state"]
    intermediates["interpolation"] = {
        "shifts_frame1": fingerprint(shifts_interp[0]),
        "shifts_frame2": fingerprint(shifts_interp[1]),
        "state": fingerprint(q_interp),
        "amplitudes_frame1": amplitude_fingerprint(amplitudes_interp[0]),
        "amplitudes_frame2": amplitude_fingerprint(amplitudes_interp[1]),
    }

    return {
        "test_model_rel_err_sPOD": rel_err_sPOD,
        "test_model_rel_err_POD": rel_err_POD,
        "errors": errors,
        "errors_inf": errors_inf,
        "intermediates": intermediates,
    }


# Lines like "Relative ... (sPOD-NN) is 0.0753..." or "... error =  3.2648e-05"
_REPORTED = re.compile(r"^(.*?)\s*(?:\bis|=)\s+([-+]?[0-9][0-9.eE+-]*)\s*$", re.MULTILINE)


def parse_reported_values(log):
    """Collect every 'name is value' line the paper code printed."""
    return {m.group(1).strip(): float(m.group(2)) for m in _REPORTED.finditer(log)}


def provenance():
    """Describe exactly where and with what these numbers were produced."""
    versions = {}
    for name in ["numpy", "scipy", "sklearn", "torch", "numba", "matplotlib"]:
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": git("rev-parse", "HEAD"),
        # True if tracked files had uncommitted edits, i.e. the code that produced these
        # numbers is not exactly git_commit. Untracked files (such as this JSON) don't count.
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="JSON file to write")
    args = parser.parse_args()
    out = args.out.resolve()  # resolve before setup_paper_imports changes directory

    setup_paper_imports()
    print("Running the synthetic pipeline (about 2-3 minutes)...", file=sys.stderr)
    log = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(log):
        result = run_synthetic()
    runtime = time.time() - start

    values = parse_reported_values(log.getvalue())
    values["test_model rel_err (sPOD-NN)"] = float(result["test_model_rel_err_sPOD"])
    values["test_model rel_err (POD-NN)"] = float(result["test_model_rel_err_POD"])

    methods = ["sPOD-NN", "POD-NN", "sPOD-I"]  # order returned by OnlinePredictionAnalysis
    curves = {}
    for m, e, e_inf in zip(methods, result["errors"], result["errors_inf"]):  # noqa: B905 (components of same size, so zip is safe)
        curves["E_j " + m] = [float(v) for v in e]
        curves["E_j_inf " + m] = [float(v) for v in e_inf]

    golden = {
        "case": "synthetic",
        "provenance": provenance(),
        "runtime_seconds": round(runtime, 1),
        "values": values,
        "curves": curves,
        "intermediates": result["intermediates"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2) + "\n")

    for name, value in values.items():
        print(f"{value:<24.17g} {name}")
    print(f"\n{len(values)} values and {len(curves)} curves written to {out} ({runtime:.0f} s)")


if __name__ == "__main__":
    main()
