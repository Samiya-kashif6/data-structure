"""
Graph Algorithms & Topological Link Prediction Metrics
Implements Common Neighbors, Jaccard Index, Adamic-Adar,
Resource Allocation, Preferential Attachment, and Personalized PageRank.
"""
import math
from typing import Set, Dict, List, Tuple
from .graph import Graph


def common_neighbors(graph: Graph, u: str, v: str) -> Set[str]:
    """
    Computes the set of common neighbors between nodes u and v:
    CN(u, v) = Γ(u) ∩ Γ(v)
    Time Complexity: O(min(deg(u), deg(v))) using set intersection.
    """
    if u not in graph.nodes or v not in graph.nodes:
        return set()
    return graph.neighbors(u).intersection(graph.neighbors(v))


def jaccard_similarity(graph: Graph, u: str, v: str) -> float:
    """
    Computes the Jaccard similarity coefficient:
    J(u, v) = |Γ(u) ∩ Γ(v)| / |Γ(u) ∪ Γ(v)|
    Values range from 0.0 (no shared friends) to 1.0 (identical friend circles).
    """
    nu = graph.neighbors(u)
    nv = graph.neighbors(v)
    if not nu and not nv:
        return 0.0

    intersection_len = len(nu.intersection(nv))
    union_len = len(nu.union(nv))
    if union_len == 0:
        return 0.0

    return intersection_len / union_len


def adamic_adar_index(graph: Graph, u: str, v: str) -> float:
    """
    Computes the Adamic-Adar link prediction metric:
    AA(u, v) = ∑_{z ∈ Γ(u) ∩ Γ(v)} 1 / log(deg(z))

    Key Insight:
    Penalizes common neighbors with extremely high degrees (hubs/celebrities),
    giving significantly higher weight to tight-knit personal friends.
    If deg(z) <= 1, log(deg(z)) is 0, so we clamp minimum degree to 2.
    """
    mutuals = common_neighbors(graph, u, v)
    score = 0.0

    for z in mutuals:
        deg = graph.degree(z)
        if deg > 1:
            score += 1.0 / math.log(deg)
        else:
            # Fallback for boundary case
            score += 1.0 / math.log(2.0)

    return score


def resource_allocation_index(graph: Graph, u: str, v: str) -> float:
    """
    Computes the Resource Allocation index:
    RA(u, v) = ∑_{z ∈ Γ(u) ∩ Γ(v)} 1 / deg(z)

    Similar to Adamic-Adar but with a steeper penalty for high-degree nodes.
    """
    mutuals = common_neighbors(graph, u, v)
    score = 0.0
    for z in mutuals:
        deg = graph.degree(z)
        if deg > 0:
            score += 1.0 / deg
    return score


def preferential_attachment(graph: Graph, u: str, v: str) -> int:
    """
    Computes Preferential Attachment score:
    PA(u, v) = deg(u) * deg(v)

    Assumes nodes with high degree are more likely to form new connections.
    """
    return graph.degree(u) * graph.degree(v)


def interest_similarity(graph: Graph, u: str, v: str) -> float:
    """
    Attribute Homophily: Jaccard similarity over user interest sets.
    Captures content/semantic affinity independent of graph topology.
    """
    node_u = graph.get_node(u)
    node_v = graph.get_node(v)
    if not node_u or not node_v:
        return 0.0

    iu = node_u.interests
    iv = node_v.interests
    if not iu or not iv:
        return 0.0

    common_interests = iu.intersection(iv)
    total_interests = iu.union(iv)
    return len(common_interests) / len(total_interests) if total_interests else 0.0


def personalized_pagerank(
    graph: Graph,
    source: str,
    alpha: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-5
) -> Dict[str, float]:
    """
    Personalized PageRank (Random Walk with Restart) centered at `source`.

    Equation:
    p = (1 - alpha) * e_source + alpha * (p * P)

    Where:
    - alpha is the probability of following a random outgoing edge (0.85)
    - (1 - alpha) is the restart probability jumping back to `source`
    - P is the transition probability matrix

    Returns:
    Dictionary of node_id -> stationary probability score.
    """
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    if n == 0 or source not in graph.nodes:
        return {}

    # Initialize probability distribution vector
    p: Dict[str, float] = {node_id: 0.0 for node_id in nodes}
    p[source] = 1.0

    # Power Iteration
    for iteration in range(max_iter):
        p_next: Dict[str, float] = {node_id: 0.0 for node_id in nodes}

        # Flow from neighbors
        for node_id in nodes:
            current_prob = p[node_id]
            if current_prob == 0.0:
                continue

            deg = graph.degree(node_id)
            if deg > 0:
                out_prob = (alpha * current_prob) / deg
                for neighbor in graph.neighbors(node_id):
                    p_next[neighbor] += out_prob
            else:
                # Dead-end node teleports back to source
                p_next[source] += alpha * current_prob

        # Teleportation back to source
        p_next[source] += (1.0 - alpha)

        # Check convergence using L1 norm
        diff = sum(abs(p_next[node_id] - p[node_id]) for node_id in nodes)
        p = p_next
        if diff < tol:
            break

    return p

