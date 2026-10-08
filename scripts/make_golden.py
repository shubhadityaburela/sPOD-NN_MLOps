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


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"


def setup_paper_imports():
    """Make the paper code importable the same way the notebooks do."""
    os.environ["MPLBACKEND"] = "Agg"  # draw nothing on screen; no display needed
    os.chdir(PAPER)  # the paper code uses paths relative to paper/
    sys.path[:0] = [str(PAPER), str(REPO / "sPOD" / "lib"), str(REPO / "DL-ROM" / "LIB")]


def run_synthetic():
    """Run the synthetic pipeline of paper/synthetic.ipynb and return its results.

    Returns a dict with:
        "test_model_rel_err_sPOD": float, the rel_err returned by test_model for sPOD-NN
        "test_model_rel_err_POD":  float, the rel_err returned by test_model for POD-NN
        "errors":     the list of 3 arrays returned by OnlinePredictionAnalysis
        "errors_inf": the list of 3 arrays returned by OnlinePredictionAnalysis
    """

    from synthetic_sup import synthetic_sup
    from DFNN import scale_params, test_model
    import numpy as np

    # Run the sPOD offline phase
    df = synthetic_sup(training_samples=[0.1, 0.15, 0.2, 0.25, 0.3], testing_sample=[0.23], 
                   nmodes=8, spod_iter=300, plot_offline_data=False)


    # We collect the time amplitudes, shifts and the parameters for the training as well as the testing data.
    TA_TRAIN = df.TA_TRAIN
    SHIFTS_TRAIN = df.SHIFTS_TRAIN
    PARAMS_TRAIN = df.PARAMS_TRAIN
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
        'scaling': True,  
        'full_order_model_dimension': df.Nx,  
        'reduced_order_model_dimension': ta_train.shape[0], 
        'totalModes': ta_train.shape[0] - df.NumFrames,  
        'num_early_stop': 100000  
    
    }
    params_POD = {
            'scaling': True,  
            'full_order_model_dimension': df.Nx, 
            'reduced_order_model_dimension': TA_POD_TRAIN.shape[0], 
            'totalModes': TA_POD_TRAIN.shape[0],  
            'num_early_stop': 500  
    }


    # Testing the trained model
    test = {
    'typeOfTest': "full",
    'test_sample': 200
    }
    log_folder_base_sPOD = 'DNN_result/synthetic/training_results_sPOD/'
    log_folder_trained_model_sPOD = sorted(Path(log_folder_base_sPOD).glob('*/'), key=os.path.getmtime)[-1]
    PATH_sPOD = str(log_folder_trained_model_sPOD) + '/trained_weights/' + 'weights.pt'

    log_folder_base_POD = 'DNN_result/synthetic/training_results_POD/'
    log_folder_trained_model_POD = sorted(Path(log_folder_base_POD).glob('*/'), key=os.path.getmtime)[-1]
    PATH_POD = str(log_folder_trained_model_POD) + '/trained_weights/' + 'weights.pt'

    PATH_sPOD = 'DNN_result/synthetic/training_results_sPOD/2025_06_26__13-27-59/trained_weights/weights.pt'
    PATH_POD = 'DNN_result/synthetic/training_results_POD/2025_06_26__14-24-05/trained_weights/weights.pt'

    # Scale the parameters before prediction
    if '/trained_weights/weights.pt' in PATH_sPOD: address_sPOD = PATH_sPOD.replace('/trained_weights/weights.pt', '')
    scaling_sPOD = np.load(address_sPOD + '/variables/' + 'scaling.npy', allow_pickle=True)

    if '/trained_weights/weights.pt' in PATH_POD: address_POD = PATH_POD.replace('/trained_weights/weights.pt', '')
    scaling_POD = np.load(address_POD + '/variables/' + 'scaling.npy', allow_pickle=True)

    PARAMS_TEST_sPOD = scale_params(PARAMS_TEST, params_sPOD, scaling_sPOD)
    PARAMS_TEST_POD = scale_params(PARAMS_TEST, params_POD, scaling_POD)


    rel_err_sPOD, results_predicted_sPOD = test_model(ta_test, PARAMS_TEST_sPOD, saved_model=True, 
                                                  PATH_TO_WEIGHTS=PATH_sPOD, params=params_sPOD,
                                                  scaling=scaling_sPOD, batch_size=50) 
    rel_err_POD, results_predicted_POD = test_model(TA_POD_TEST, PARAMS_TEST_POD, saved_model=True,
                                                    PATH_TO_WEIGHTS=PATH_POD, params=params_POD,
                                                    scaling=scaling_POD, batch_size=50)


    TA_sPOD_pred = results_predicted_sPOD[:-2, :]
    shifts_sPOD_pred = results_predicted_sPOD[-2:, :]
    TA_POD_pred = results_predicted_POD

    errors, errors_inf = df.OnlinePredictionAnalysis(TA_sPOD_pred, shifts_sPOD_pred, TA_POD_pred, plot_online=False, test_type=test)


    return { "test_model_rel_err_sPOD": rel_err_sPOD,
            "test_model_rel_err_POD":  rel_err_POD,
            "errors":     errors,
            "errors_inf": errors_inf}


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
    for m, e, e_inf in zip(methods, result["errors"], result["errors_inf"]):
        curves["E_j " + m] = [float(v) for v in e]
        curves["E_j_inf " + m] = [float(v) for v in e_inf]

    golden = {
        "case": "synthetic",
        "provenance": provenance(),
        "runtime_seconds": round(runtime, 1),
        "values": values,
        "curves": curves,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2) + "\n")

    for name, value in values.items():
        print(f"{value:<24.17g} {name}")
    print(f"\n{len(values)} values and {len(curves)} curves written to {out} ({runtime:.0f} s)")


if __name__ == "__main__":
    main()
