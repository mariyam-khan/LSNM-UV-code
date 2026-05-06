"""
LSNM Data Generation -- Section 6.1 simulation setup.
======================================================

Overview
--------
This file generates synthetic data for evaluating the LSNM-UV-X algorithm.
Each call to gen_lsnm_experiment() produces one simulation trial: a dataset of
observed variables, and the ground-truth ADMG that the algorithm is expected to
recover.  The graph structure follows Maeda & Shimizu (2021) Section 5.1, and
the data model extends it to the Location-Scale Noise Model of our paper (Eq. 1).


Step 1 -- Graph Construction (_build_full_graph)
-------------------------------------------------
We construct a full latent DAG over both observed and hidden variables.
Variable indices are laid out as:

    0 ... p-1          ->  observed variables  x_0, ..., x_{p-1}
    p ... p+n_cc-1     ->  hidden common causes  (root nodes, no parents)
    p+n_cc ... end     ->  hidden intermediates

With default parameters: p=10 observed variables, n_cc=2 hidden common causes,
n_int=2 hidden intermediates (14 nodes total).

Observed skeleton.
    Direct edges among the p observed variables are drawn from an Erdos-Renyi
    DAG with edge probability 0.3, following Maeda21 Section 5.1 exactly.
    Node index order is the topological order (edge x_j -> x_i only if j < i).

Hidden common causes (UBPs).
    For each hidden common cause u_k (a root node with no parents of its own):
      - We select a pair of observed variables (x_a, x_b) with no direct edge
        between them.
      - We add edges u_k -> x_a and u_k -> x_b.
      - In the projected ADMG this becomes a bidirected edge x_a <-> x_b,
        representing an Unobserved Backdoor Path (UBP).

Hidden intermediates (UCPs).
    For each hidden intermediate y_k:
      - We select an existing observed-to-observed edge x_j -> x_i.
      - We replace it with x_j -> y_k -> x_i (the direct edge is removed).
      - In the projected ADMG this becomes a bidirected edge x_j <-> x_i,
        representing an Unobserved Causal Path (UCP).

Bow-free guarantee.
    The construction ensures no observed variable simultaneously has a direct
    edge and a bidirected edge to the same other variable, satisfying the
    bow-free ADMG condition required by the paper (Definition 2.5).


Step 2 -- Data Generation (_gen_lsnm_variable)
-----------------------------------------------
All variables (observed and hidden) are generated in topological order so that
when we generate variable v_i, all its parents already have values.

For each variable v_i we implement the two-level LSNM from Equation (1):

    v_i  =  f_i^1(K_i)  +  g_i^1(K_i) * eta_i          (Layer 1)
    eta_i  =  f_i^2(Q_i)  +  g_i^2(Q_i) * eps_i         (Layer 2)

where K_i are the observed parents of v_i, Q_i are the hidden parents of v_i,
and eps_i ~ N(0,1) are mutually independent across all variables.


Step 3 -- Ground-Truth ADMG (compute_true_admg)
------------------------------------------------
After generating data for all variables, we project the full latent DAG down
to the p observed variables to obtain the ground-truth ADMG (A, B).

Step 4 -- Final Output (gen_lsnm_experiment)
---------------------------------------------
A random column permutation is applied before returning.

Requirements: numpy, networkx
"""

import numpy as np
import networkx as nx


# -----------------------------------------------------------------------------
# Nonlinear function families
# -----------------------------------------------------------------------------

def nlfunc_polynomial(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Maeda & Shimizu (2021) Eq. (8): (x + a)^c + b"""
    a = rng.uniform(-5.0, 5.0)
    b = rng.uniform(-1.0, 1.0)
    c = rng.choice([2, 3])
    return (x + a) ** c + b


def nlfunc_sigmoid(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scaled sigmoid: a * sigmoid(b*x + c) + d"""
    a = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-2.0, 2.0)
    d = rng.uniform(-1.0, 1.0)
    return a / (1.0 + np.exp(-(b * x + c))) + d


def nlfunc_trigonometric(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal: a * sin(b*x + c) + d"""
    a = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-np.pi, np.pi)
    d = rng.uniform(-1.0, 1.0)
    return a * np.sin(b * x + c) + d


def nlfunc_rbf(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Radial basis function: a * exp(-b*(x - c)^2) + d"""
    a = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    b = rng.uniform(0.1, 1.0)
    c = rng.uniform(-2.0, 2.0)
    d = rng.uniform(-1.0, 1.0)
    return a * np.exp(-b * (x - c) ** 2) + d


def nlfunc_tanh(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hyperbolic tangent: a * tanh(b*x + c) + d"""
    a = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-2.0, 2.0)
    d = rng.uniform(-1.0, 1.0)
    return a * np.tanh(b * x + c) + d


def nlfunc_softplus(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Softplus: a * log(1 + exp(b*x + c)) + d"""
    a = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-2.0, 2.0)
    d = rng.uniform(-1.0, 1.0)
    return a * np.log1p(np.exp(np.clip(b * x + c, -20, 20))) + d


def nlfunc_logarithmic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Logarithmic: a * sign(b*x + c) * log(|b*x + c| + 1) + d"""
    a = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-2.0, 2.0)
    d = rng.uniform(-1.0, 1.0)
    z = b * x + c
    return a * np.sign(z) * np.log(np.abs(z) + 1.0) + d


def nlfunc_quadratic_sine(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Quadratic-sine (growing oscillation): a * x * sin(b*x + c) + d"""
    a = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-np.pi, np.pi)
    d = rng.uniform(-1.0, 1.0)
    return a * x * np.sin(b * x + c) + d


_FUNC_REGISTRY = {
    "polynomial": nlfunc_polynomial,
    "sigmoid": nlfunc_sigmoid,
    "trigonometric": nlfunc_trigonometric,
    "rbf": nlfunc_rbf,
    "tanh": nlfunc_tanh,
    "softplus": nlfunc_softplus,
    "logarithmic": nlfunc_logarithmic,
    "quadratic_sine": nlfunc_quadratic_sine,
}


def nlfunc_by_type(x: np.ndarray, rng: np.random.Generator,
                   func_type: str = "polynomial") -> np.ndarray:
    fn = _FUNC_REGISTRY.get(func_type)
    if fn is None:
        raise ValueError(f"Unknown func_type '{func_type}'. "
                         f"Choose from {list(_FUNC_REGISTRY.keys())}")
    return fn(x, rng)


# Backward-compatible alias
random_nlfunc = nlfunc_polynomial


# -----------------------------------------------------------------------------
# Preset function configurations
# Each config assigns a different function family to each LSNM component:
#   f1 = Layer 1 location, g1 = Layer 1 scale,
#   f2 = Layer 2 location, g2 = Layer 2 scale
# -----------------------------------------------------------------------------

_FAMILY_ORDER = [
    "polynomial", "sigmoid", "trigonometric", "rbf",
    "tanh", "softplus", "logarithmic", "quadratic_sine",
]

FUNC_CONFIGS = {
    chr(ord("A") + k): {
        "f1": _FAMILY_ORDER[k % 8],
        "g1": _FAMILY_ORDER[(k + 1) % 8],
        "f2": _FAMILY_ORDER[(k + 2) % 8],
        "g2": _FAMILY_ORDER[(k + 3) % 8],
    }
    for k in range(8)
}

DEFAULT_FUNC_CONFIG = {"f1": "polynomial", "g1": "polynomial",
                       "f2": "polynomial", "g2": "polynomial"}


# -----------------------------------------------------------------------------
# Variable generator -- Layer 2 + Layer 1 (LSNM)
# -----------------------------------------------------------------------------

def _gen_lsnm_variable(
    obs_parent_vals: list,
    hid_parent_vals: list,
    rng: np.random.Generator,
    n: int,
    func_config: dict = None,
) -> np.ndarray:
    """
    Generate one variable v_i from the two-level LSNM (paper Eq. 1 / Eq. 2).

    Parameters
    ----------
    func_config : dict with keys "f1", "g1", "f2", "g2" mapping to function
                  family names. If None, uses polynomial for all.
    """
    if func_config is None:
        func_config = DEFAULT_FUNC_CONFIG

    # -------------------------------------------------------------------------
    # Layer 2 -- paper Eq. (7):  eta_i = f_i^2(Q_i) + g_i^2(Q_i) * eps_i
    # -------------------------------------------------------------------------
    eps_i = rng.standard_normal(n)

    if len(hid_parent_vals) == 0:
        eta_i = eps_i
    else:
        f_i2 = np.zeros(n)
        for u_k in hid_parent_vals:
            f_i2 = f_i2 + nlfunc_by_type(u_k, rng, func_config["f2"])

        log_g_i2 = np.zeros(n)
        for u_k in hid_parent_vals:
            log_g_i2 = log_g_i2 + nlfunc_by_type(u_k, rng, func_config["g2"])
        s = np.std(log_g_i2)
        if s > 1e-8:
            log_g_i2 = log_g_i2 / s
        g_i2 = np.clip(np.exp(log_g_i2), 0.1, 10.0)

        eta_i = f_i2 + g_i2 * eps_i

    eta_i = (eta_i - np.mean(eta_i)) / (np.std(eta_i) + 1e-8)

    # -------------------------------------------------------------------------
    # Layer 1 -- paper Eq. (2):  v_i = f_i^1(K_i) + g_i^1(K_i) * eta_i
    # -------------------------------------------------------------------------
    if len(obs_parent_vals) == 0:
        v_i = eta_i
    else:
        f_i1 = np.zeros(n)
        for x_k in obs_parent_vals:
            f_i1 = f_i1 + nlfunc_by_type(x_k, rng, func_config["f1"])

        log_g_i1 = np.zeros(n)
        for x_k in obs_parent_vals:
            log_g_i1 = log_g_i1 + nlfunc_by_type(x_k, rng, func_config["g1"])
        s = np.std(log_g_i1)
        if s > 1e-8:
            log_g_i1 = log_g_i1 / s
        g_i1 = np.clip(np.exp(log_g_i1), 0.1, 10.0)

        v_i = f_i1 + g_i1 * eta_i

    return (v_i - np.mean(v_i)) / (np.std(v_i) + 1e-8)


# -----------------------------------------------------------------------------
# Step 1 -- Graph construction
# -----------------------------------------------------------------------------

def gen_er_dag(p: int, er_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi DAG over p observed variables (Maeda21 Section 5.1)."""
    G = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i):
            if rng.random() < er_prob:
                G[i, j] = 1
    return G


def _build_full_graph(rng, p=10, n_cc=2, n_int=2, er_prob=0.3):
    """Build the full latent DAG over observed and hidden variables (Step 1)."""
    n_total = p + n_cc + n_int
    G_obs   = gen_er_dag(p, er_prob, rng)

    G_full  = np.zeros((n_total, n_total), dtype=int)
    G_full[:p, :p] = G_obs

    cc_pairs  = []
    int_pairs = []

    used_bidir_pairs: set = set()
    for k in range(n_cc):
        hc = p + k
        candidates = [
            (i, j) for i in range(p) for j in range(i + 1, p)
            if G_full[i, j] == 0 and G_full[j, i] == 0
            and (i, j) not in used_bidir_pairs
        ]
        if not candidates:
            continue
        idx  = int(rng.integers(0, len(candidates)))
        a, b = candidates[idx]
        G_full[a, hc] = 1
        G_full[b, hc] = 1
        cc_pairs.append((a, b))
        used_bidir_pairs.add((a, b))

    for k in range(n_int):
        hi = p + n_cc + k
        edges = [(i, j) for i in range(p) for j in range(p) if G_full[i, j] == 1]
        if not edges:
            break
        idx  = int(rng.integers(0, len(edges)))
        i, j = edges[idx]
        G_full[i, j]  = 0
        G_full[hi, j] = 1
        G_full[i, hi] = 1
        int_pairs.append((j, i))

    return G_full, cc_pairs, int_pairs


def compute_true_admg(G_full: np.ndarray, p: int, cc_pairs: list, int_pairs: list):
    """Project the full latent DAG onto the ground-truth ADMG (Step 3)."""
    A = np.zeros((p, p), dtype=int)
    B = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if G_full[i, j] == 1:
                A[i, j] = 1

    for (a, b) in cc_pairs:
        B[a, b] = 1;  B[b, a] = 1

    for (j, i) in int_pairs:
        B[i, j] = 1;  B[j, i] = 1

    overlap = int(np.sum((A == 1) & (B == 1)))
    assert overlap == 0, (
        f"Bow-free violation: {overlap} pairs have both directed and bidirected edges."
    )

    return A, B


# -----------------------------------------------------------------------------
# Step 4 -- Main experiment generator
# -----------------------------------------------------------------------------

def gen_lsnm_experiment(
    n:           int,
    seed:        int   = None,
    p:           int   = 10,
    n_cc:        int   = 2,
    n_int:       int   = 2,
    er_prob:     float = 0.3,
    func_config: dict  = None,
):
    """
    Generate one simulation trial (Steps 1-4).

    Parameters
    ----------
    n           : number of i.i.d. samples
    seed        : integer random seed (None for non-reproducible)
    p           : number of observed variables                     (default 10)
    n_cc        : number of hidden common causes (UBPs)            (default 2)
    n_int       : number of hidden intermediates (UCPs)            (default 2)
    er_prob     : Erdos-Renyi edge probability                     (default 0.3)
    func_config : dict mapping component names ("f1","g1","f2","g2") to function
                  family names. If None, uses polynomial for all components.

    Returns
    -------
    X_perm  : (n, p) ndarray
    A_true  : (p, p) ndarray
    B_true  : (p, p) ndarray
    perm    : (p,) ndarray
    """
    if func_config is None:
        func_config = DEFAULT_FUNC_CONFIG

    rng     = np.random.default_rng(seed)
    n_total = p + n_cc + n_int

    G_full, cc_pairs, int_pairs = _build_full_graph(
        rng, p=p, n_cc=n_cc, n_int=n_int, er_prob=er_prob
    )

    G_nx       = nx.from_numpy_array(G_full.T, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G_nx))

    data = np.zeros((n, n_total))
    for v in topo_order:
        parent_indices  = np.where(G_full[v, :] == 1)[0]
        obs_parent_vals = [data[:, par] for par in parent_indices if par < p]
        hid_parent_vals = [data[:, par] for par in parent_indices if par >= p]
        data[:, v]      = _gen_lsnm_variable(
            obs_parent_vals, hid_parent_vals, rng, n,
            func_config=func_config,
        )

    X_obs = data[:, :p]

    perm   = np.arange(p)
    rng.shuffle(perm)
    X_perm = X_obs[:, perm]

    A_raw, B_raw = compute_true_admg(G_full, p, cc_pairs, int_pairs)
    A_true = A_raw[np.ix_(perm, perm)]
    B_true = B_raw[np.ix_(perm, perm)]

    return X_perm, A_true, B_true, perm
