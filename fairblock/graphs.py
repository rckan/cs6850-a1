"""Network generation and loaders.

Every function returns (G, groups): G is a simple undirected graph with nodes
0..n-1 (what model.sample_worlds needs), and groups[v] is "H" or "L", the
label model.node_acceptance uses to pick a mean acceptance.
"""

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


def sbm_fixed_degree(n_high, n_low, ratio, mean_degree, seed=None):
    """Two-group SBM with p_out = ratio * p_in, scaled so the expected mean degree
    is mean_degree whatever the ratio. Lets a homophily sweep change only how
    edges are split between and within groups, not how dense the network is.
    """
    p_in, p_out = sbm_probs(n_high, n_low, ratio, mean_degree)
    return two_group_sbm(n_high, n_low, p_in, p_out, seed=seed)


def sbm_probs(n_high, n_low, ratio, mean_degree):
    """(p_in, p_out) with p_out / p_in = ratio and expected mean degree mean_degree."""
    n = n_high + n_low
    within = n_high * (n_high - 1) / 2 + n_low * (n_low - 1) / 2
    between = n_high * n_low
    p_in = mean_degree * n / 2 / (within + ratio * between)
    if not 0 < ratio * p_in <= p_in <= 1:
        raise ValueError(f"no valid SBM for ratio={ratio}, mean_degree={mean_degree}")
    return p_in, ratio * p_in


def from_labels(G, labels, low):
    """Largest connected component of G as a simple undirected graph on 0..n-1.

    labels maps each node of G to a category; nodes whose label is `low` become
    group L, everything else group H. Node order follows sorted(G.nodes) so the
    numbering is reproducible.
    """
    U = nx.Graph(G.to_undirected())  # drops parallel edges and direction
    U.remove_edges_from(nx.selfloop_edges(U))
    cc = max(nx.connected_components(U), key=len)
    nodes = sorted(cc)
    index = {v: i for i, v in enumerate(nodes)}
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    H.add_edges_from((index[u], index[v]) for u, v in U.subgraph(cc).edges)
    groups = np.array(["L" if labels[v] == low else "H" for v in nodes])
    return H, groups


def karate_club(low="Officer"):
    """Zachary's karate club, split by the club each member joined (17 / 17)."""
    G = nx.karate_club_graph()
    return from_labels(G, nx.get_node_attributes(G, "club"), low)
