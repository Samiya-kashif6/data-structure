"""
Performance & Scalability Benchmark Suite for GraphPulse Engine
Measures throughput, candidate retrieval latency, and algorithm execution times.
"""
import time
import random
from engine.graph import Graph, Node
from engine.algorithms import (
    common_neighbors,
    jaccard_similarity,
    adamic_adar_index,
    personalized_pagerank
)
from engine.recommender import FriendRecommender


def build_benchmark_graph(num_nodes: int = 1000, avg_degree: int = 20) -> Graph:
    print(f"\n[+] Synthesizing Barabasi-Albert Scale-Free Graph with {num_nodes:,} nodes (avg degree ~{avg_degree})...")
    g = Graph()
    # Seed nodes
    m = avg_degree // 2
    for i in range(m):
        g.add_node(Node(id=f"u_{i}", name=f"User {i}", interests={"Python", "Data"}))
        for j in range(i):
            g.add_edge(f"u_{i}", f"u_{j}")

    # Preferential attachment generation
    repeated_nodes = []
    for i in range(m):
        for j in range(i):
            repeated_nodes.extend([f"u_{i}", f"u_{j}"])

    for i in range(m, num_nodes):
        new_id = f"u_{i}"
        g.add_node(Node(id=new_id, name=f"User {i}", interests={"Tech", "AI", "Cloud"}))
        # Pick m targets with preferential attachment
        targets = set()
        while len(targets) < m and len(repeated_nodes) > 0:
            targets.add(random.choice(repeated_nodes))

        for target in targets:
            g.add_edge(new_id, target)
            repeated_nodes.extend([new_id, target])

    return g


def run_benchmark():
    g = build_benchmark_graph(num_nodes=1000, avg_degree=20)
    stats = g.graph_statistics()
    print(f"    Graph generated: {stats['total_users']:,} nodes, {stats['total_friendships']:,} edges.")
    print(f"    Avg Degree: {stats['average_degree']}, Graph Density: {stats['graph_density']}")

    rec_engine = FriendRecommender(g)
    test_users = [f"u_{random.randint(10, 950)}" for _ in range(50)]

    print("\n[+] Benchmarking 2-Hop Candidate Retrieval across 50 random nodes...")
    t0 = time.perf_counter()
    total_candidates = 0
    for uid in test_users:
        cands = g.two_hop_neighbors(uid)
        total_candidates += len(cands)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    print(f"    Total candidates found: {total_candidates:,}")
    print(f"    Total time: {elapsed_ms:.2f} ms | Avg per user: {elapsed_ms / 50:.3f} ms")

    print("\n[+] Benchmarking Adamic-Adar vs Jaccard Link Prediction (1,000 pairs)...")
    node_pairs = [(random.choice(test_users), random.choice(test_users)) for _ in range(1000)]

    t0 = time.perf_counter()
    for u, v in node_pairs:
        adamic_adar_index(g, u, v)
    t1 = time.perf_counter()
    aa_ms = (t1 - t0) * 1000
    print(f"    1,000 Adamic-Adar calculations: {aa_ms:.2f} ms ({aa_ms / 1000 * 1000:.2f} µs/pair)")

    t0 = time.perf_counter()
    for u, v in node_pairs:
        jaccard_similarity(g, u, v)
    t1 = time.perf_counter()
    jc_ms = (t1 - t0) * 1000
    print(f"    1,000 Jaccard calculations: {jc_ms:.2f} ms ({jc_ms / 1000 * 1000:.2f} µs/pair)")

    print("\n[+] Benchmarking Personalized PageRank (Power Iteration on 1,000 nodes)...")
    t0 = time.perf_counter()
    ppr = personalized_pagerank(g, source=test_users[0], alpha=0.85, max_iter=25)
    t1 = time.perf_counter()
    ppr_ms = (t1 - t0) * 1000
    print(f"    PPR Convergence time: {ppr_ms:.2f} ms across all 1,000 nodes")

    print("\n[+] Benchmarking Full End-to-End Recommendation Pipeline (Top-10 for 20 users)...")
    t0 = time.perf_counter()
    for uid in test_users[:20]:
        rec_engine.recommend(uid, limit=10)
    t1 = time.perf_counter()
    full_ms = (t1 - t0) * 1000
    print(f"    20 Full Recommendations: {full_ms:.2f} ms | Avg latency: {full_ms / 20:.2f} ms per request")
    print(f"    Throughput: {20 / (full_ms / 1000):.1f} queries/sec on single CPU core")


if __name__ == "__main__":
    run_benchmark()

