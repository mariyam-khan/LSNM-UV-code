"""
Experiment runner -- LSNM-UV-X Section 6.1 simulations.

Compares:
    LSNM-UV-X  (this paper)
    CAM-UV     (Maeda & Shimizu 2021,  lingam package)
    FCI        (Spirtes et al. 2000,   causal-learn package)
    BANG       (Wang et al.,           R package ngBap)

Usage
-----
    python run_experiments.py                    # run all 4 function configs
    python run_experiments.py --config A         # run single config
    python run_experiments.py --n-trials 10      # quick test

Requirements: numpy, pandas, joblib, lingam, pygam, causal-learn
Optional:     rpy2  +  R package ngBap  (for BANG)
"""

import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lsnm_data_gen import gen_lsnm_experiment, FUNC_CONFIGS, DEFAULT_FUNC_CONFIG
from lsnm_uv_x    import LSNMUV_X
from eval_metrics  import (
    directed_metrics, bidirected_metrics,
    parse_camuv_result, parse_fci_result, parse_bang_result,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def run_lsnm_uv_x(X, alpha=0.01, d=3, max_regress_size=2):
    model = LSNMUV_X(alpha=alpha, num_explanatory_vals=d,
                     max_regress_size=max_regress_size)
    model.fit(X)
    return parse_camuv_result(model)


def run_camuv(X, alpha=0.01, d=3):
    from lingam import CAMUV
    model = CAMUV(alpha=alpha, num_explanatory_vals=d)
    model.fit(X)
    return parse_camuv_result(model)


def run_fci(X, alpha=0.01):
    import io, contextlib
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    p = X.shape[1]
    with contextlib.redirect_stderr(io.StringIO()), \
         contextlib.redirect_stdout(io.StringIO()):
        pag, _ = fci(X, independence_test_method=fisherz, alpha=alpha,
                      verbose=False, show_progress=False)
    return parse_fci_result(pag, p)


def run_bang(X):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        bang_r_path = os.path.join(script_dir, "bang", "bang.R")
        if not os.path.exists(bang_r_path):
            bang_r_path = os.path.join(script_dir, "..", "bang", "bang.R")
        ro.r.source(bang_r_path)

        conv = numpy2ri.converter
        n, p = X.shape
        with conv.context():
            r_X = ro.r.matrix(ro.FloatVector(X.flatten()),
                              nrow=n, ncol=p, byrow=True)
            result = ro.r.bang(r_X, K=3, level=0.01,
                               verbose=False, restrict=1,
                               testType="dhsic")
            A_est, B_est = parse_bang_result(result, p)

        return A_est, B_est

    except Exception as e:
        log.warning("[BANG] skipped (%s)", e)
        return None, None


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(n, seed, alpha=0.01, d=3, include_bang=True,
                     func_config=None):
    X, A_true, B_true, _ = gen_lsnm_experiment(
        n=n, seed=seed, func_config=func_config,
    )

    methods = {
        "LSNM-UV-X": lambda: run_lsnm_uv_x(X, alpha=alpha, d=d),
        "CAM-UV":    lambda: run_camuv(X, alpha=alpha, d=d),
        "FCI":       lambda: run_fci(X, alpha=alpha),
    }
    if include_bang:
        methods["BANG"] = lambda: run_bang(X)

    rows = []
    for name, fn in methods.items():
        t0 = time.perf_counter()
        A_est = B_est = None
        failed = False
        try:
            A_est, B_est = fn()
        except Exception as e:
            log.warning("  [%s] n=%d seed=%d: %s", name, n, seed, e)
            failed = True
        runtime = time.perf_counter() - t0

        if failed or A_est is None or B_est is None:
            continue

        prec_d, rec_d, f1_d = directed_metrics(A_est, A_true)
        prec_b, rec_b, f1_b = bidirected_metrics(B_est, B_true)

        rows.append(dict(
            method        = name,
            n             = n,
            seed          = seed,
            prec_directed = prec_d,
            rec_directed  = rec_d,
            f1_directed   = f1_d,
            prec_bidir    = prec_b,
            rec_bidir     = rec_b,
            f1_bidir      = f1_b,
            runtime_sec   = runtime,
        ))

    return rows


# ---------------------------------------------------------------------------
# Full experiment grid for one function config
# ---------------------------------------------------------------------------

def run_all_experiments(
    n_list=None, n_trials=100, alpha=0.01, d=3,
    include_bang=True, n_jobs=-1,
    save_path="results_section6.csv",
    func_config=None,
):
    if n_list is None:
        n_list = [200, 400, 600, 800, 1000]

    tasks = [(n, seed) for n in n_list for seed in range(n_trials)]
    log.info("Launching %d trials (%d sizes x %d trials) ...",
             len(tasks), len(n_list), n_trials)

    all_rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_single_trial)(
            n, seed, alpha=alpha, d=d,
            include_bang=include_bang,
            func_config=func_config,
        )
        for n, seed in tasks
    )

    df = pd.DataFrame([row for trial in all_rows for row in trial])
    df.to_csv(save_path, index=False)
    log.info("Saved -> %s  (%d rows)", save_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Multi-config runner: run all 4 function configs
# ---------------------------------------------------------------------------

def run_all_func_configs(
    n_list=None, n_trials=100, alpha=0.01, d=3,
    include_bang=True, n_jobs=-1,
    output_dir=".",
):
    import os
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for config_name, func_config in FUNC_CONFIGS.items():
        log.info("=== Config %s: %s ===", config_name, func_config)
        save_path = os.path.join(output_dir, f"results_config_{config_name}.csv")
        df = run_all_experiments(
            n_list=n_list, n_trials=n_trials, alpha=alpha, d=d,
            include_bang=include_bang, n_jobs=n_jobs,
            save_path=save_path, func_config=func_config,
        )
        df["config"] = config_name
        results[config_name] = df

    df_all = pd.concat(results.values(), ignore_index=True)
    combined_path = os.path.join(output_dir, "results_all_configs.csv")
    df_all.to_csv(combined_path, index=False)
    log.info("Combined results -> %s  (%d rows)", combined_path, len(df_all))
    return df_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df, save_path="figure_section6_directed.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = df["method"].unique()
    n_list  = sorted(df["n"].unique())
    cols    = ["prec_directed", "rec_directed", "f1_directed"]
    titles  = ["Average Precision", "Average Recall", "Average F-measure"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, col, title in zip(axes, cols, titles):
        for method in methods:
            sub = df[df["method"] == method].groupby("n")[col].mean()
            ax.plot(sub.index, sub.values, marker="s", label=method)
        ax.set_xlabel("sample size")
        ax.set_ylabel(title.lower())
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_bidir_results(df, save_path="figure_section6_bidir.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = df["method"].unique()
    cols    = ["prec_bidir", "rec_bidir", "f1_bidir"]
    titles  = ["Average Precision (UBP/UCP)", "Average Recall (UBP/UCP)",
               "Average F-measure (UBP/UCP)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, col, title in zip(axes, cols, titles):
        for method in methods:
            sub = df[df["method"] == method].groupby("n")[col].mean()
            ax.plot(sub.index, sub.values, marker="s", label=method)
        ax.set_xlabel("sample size")
        ax.set_ylabel(title.lower())
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multiconfig(df, save_dir="."):
    """Generate one 2x3 figure per config (precision/recall/F1 x directed/bidirected)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    configs = sorted(df["config"].unique())
    methods = sorted(df["method"].unique())
    colors  = {"LSNM-UV-X": "#e41a1c", "CAM-UV": "#377eb8",
               "FCI": "#4daf4a", "BANG": "#984ea3"}
    markers = {"LSNM-UV-X": "o", "CAM-UV": "s", "FCI": "^", "BANG": "D"}
    labels  = {"LSNM-UV-X": "LSNM-UV", "CAM-UV": "CAM-UV",
               "FCI": "FCI", "BANG": "BANG"}

    row_specs = [
        ("Directed edges",  ["prec_directed", "rec_directed", "f1_directed"]),
        ("Bidirected edges", ["prec_bidir",    "rec_bidir",    "f1_bidir"]),
    ]
    col_titles = ["Precision", "Recall", "F1"]

    for config_name in configs:
        sub = df[df["config"] == config_name]
        cfg = FUNC_CONFIGS[config_name]
        suptitle = (f"Config {config_name}: "
                    f"f1={cfg['f1']}, g1={cfg['g1']}, "
                    f"f2={cfg['f2']}, g2={cfg['g2']}")

        fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
        fig.suptitle(suptitle, fontsize=10)

        for row_idx, (row_label, cols) in enumerate(row_specs):
            for col_idx, (col, col_title) in enumerate(zip(cols, col_titles)):
                ax = axes[row_idx][col_idx]
                for method in methods:
                    msub = sub[sub["method"] == method]
                    if msub.empty:
                        continue
                    means = msub.groupby("n")[col].mean()
                    ax.plot(means.index, means.values,
                            marker=markers.get(method, "o"),
                            color=colors.get(method, "gray"),
                            label=labels.get(method, method),
                            linewidth=1.5, markersize=5)
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=9, fontweight="bold")
                if row_idx == 1:
                    ax.set_xlabel("sample size")
                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)
                if row_idx == 0 and col_idx == 2:
                    ax.legend(fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"figure_config_{config_name}.png"), dpi=150)
        plt.close()


def plot_runtime(df, save_path="figure_runtime.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in df["method"].unique():
        sub = df[df["method"] == method].groupby("n")["runtime_sec"].mean()
        ax.plot(sub.index, sub.values, marker="s", label=method)
    ax.set_xlabel("sample size")
    ax.set_ylabel("average run time (seconds)")
    ax.set_title("Average runtime")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="LSNM-UV-X experiments")
    parser.add_argument("--config", type=str, default=None,
                        help="Run single config (A/B/C/D) or omit for all")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-bang", action="store_true",
                        help="Skip BANG (if ngBap not installed)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.expanduser("~/LSNM_simulations"))
    args = parser.parse_args()

    include_bang = not args.no_bang
    n_list = [200, 400, 600, 800, 1000]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.config:
        cfg = FUNC_CONFIGS[args.config]
        log.info("Running config %s: %s", args.config, cfg)
        df = run_all_experiments(
            n_list=n_list, n_trials=args.n_trials,
            include_bang=include_bang, n_jobs=args.n_jobs,
            save_path=f"{args.output_dir}/results_config_{args.config}.csv",
            func_config=cfg,
        )
        plot_results(df, save_path=f"{args.output_dir}/figure_directed_{args.config}.png")
        plot_bidir_results(df, save_path=f"{args.output_dir}/figure_bidir_{args.config}.png")
    else:
        df_all = run_all_func_configs(
            n_list=n_list, n_trials=args.n_trials,
            include_bang=include_bang, n_jobs=args.n_jobs,
            output_dir=args.output_dir,
        )
        plot_multiconfig(df_all, save_dir=args.output_dir)
        plot_runtime(df_all, save_path=f"{args.output_dir}/figure_runtime.png")
