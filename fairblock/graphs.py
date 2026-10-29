"""Network generation."""

import networkx as nx
import numpy as np


def two_group_sbm(n_high, n_low, p_in, p_out, seed=None):
    """Two-community SBM. Returns (G, groups) where groups[v] is "H" or "L".

    Nodes 0..n_high-1 are group H (high acceptance), the rest group L.
    """
    G = nx.stochastic_block_model(
        [n_high, n_low], [[p_in, p_out], [p_out, p_in]], seed=seed
    )
    groups = np.array(["H"] * n_high + ["L"] * n_low)
    return G, groups
