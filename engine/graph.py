"""
Graph Data Structures for Social Network Analysis
Optimized for fast topological link prediction and two-hop candidate retrieval.
"""
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Node:
    """
    DATA STRUCTURE: Node Object (Vertex)
    Holds profile attributes, interests (Hash Set), and community tags.
    """
    id: str
    name: str
    city: str = ""
    interests: Set[str] = field(default_factory=set)
    community: str = "General"
    avatar: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "city": self.city,
            "interests": sorted(list(self.interests)),
            "community": self.community,
            "avatar": self.avatar or f"https://api.dicebear.com/7.x/avataaars/svg?seed={self.id}",
            "metadata": self.metadata
        }


class Graph:
    """
    ====================================================================
    DATA STRUCTURE: Undirected Graph via Adjacency List
    --------------------------------------------------------------------
    Representation:
      _adj is a Hash Map (Python Dict) where:
        - Key: User ID (string)
        - Value: Hash Set of Neighbor IDs (set of strings)
    
    Why this Data Structure? (Beginner DSA Breakdown):
      1. Edge Lookup has_edge(u, v):
         - Adjacency Matrix: O(1) time, but takes O(V^2) memory (wastes space on sparse graphs).
         - Adjacency List with Sets: O(1) time, and only takes O(V + E) memory!
      2. Finding Neighbors neighbors(u):
         - Takes O(1) time to access the set.
      3. Finding Mutual Friends:
         - Hash Set Intersection set_u & set_v takes O(min(deg(u), deg(v))).
    ====================================================================
    """

    def __init__(self):
        # 1. HASH MAP of nodes: O(1) lookup by ID
        self._nodes: Dict[str, Node] = {}
        # 2. ADJACENCY LIST (Hash Map of Hash Sets): O(1) neighbor lookups
        self._adj: Dict[str, Set[str]] = {}
        # 3. HASH SET for tracking dismissed recommendations per user
        self._dismissed: Dict[str, Set[str]] = {}

    def add_node(self, node: Node) -> None:
        """Add a user node (vertex) to the graph. Time Complexity: O(1)."""
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            self._adj[node.id] = set()
            self._dismissed[node.id] = set()

    def get_node(self, node_id: str) -> Optional[Node]:
        """Fetch node by identifier. Time Complexity: O(1)."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node exists. Time Complexity: O(1)."""
        return node_id in self._nodes

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    @property
    def num_nodes(self) -> int:
        """Total vertices |V|."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Total edges |E|."""
        return sum(len(neighbors) for neighbors in self._adj.values()) // 2

    def add_edge(self, u: str, v: str) -> bool:
        """
        Add an undirected friendship edge between u and v.
        Updates both sets since friendship is reciprocal.
        Time Complexity: O(1).
        """
        if u == v or u not in self._nodes or v not in self._nodes:
            return False
        if v in self._adj[u]:
            return False

        self._adj[u].add(v)
        self._adj[v].add(u)
        return True

    def remove_edge(self, u: str, v: str) -> bool:
        """Remove friendship edge between u and v. Time Complexity: O(1)."""
        if u in self._adj and v in self._adj[u]:
            self._adj[u].remove(v)
            self._adj[v].remove(u)
            return True
        return False

    def has_edge(self, u: str, v: str) -> bool:
        """O(1) check whether an edge exists between u and v using Hash Set lookup."""
        return u in self._adj and v in self._adj[u]

    def neighbors(self, node_id: str) -> Set[str]:
        """Return the Hash Set of direct friends (degree-1 neighbors). Time Complexity: O(1)."""
        return self._adj.get(node_id, set())

    def degree(self, node_id: str) -> int:
        """Return the degree (number of friends). Time Complexity: O(1)."""
        return len(self._adj.get(node_id, set()))

    def two_hop_neighbors(self, node_id: str) -> Set[str]:
        """
        Retrieve all 'friends of friends' strictly at distance 2.
        Excludes self, direct friends, and dismissed candidates.
        Time Complexity: O(deg(u) * avg_deg).
        """
        if node_id not in self._adj:
            return set()

        direct_friends = self._adj[node_id]
        dismissed = self._dismissed.get(node_id, set())
        two_hop_candidates = set()

        for friend in direct_friends:
            for fof in self._adj.get(friend, set()):
                if fof != node_id and fof not in direct_friends and fof not in dismissed:
                    two_hop_candidates.add(fof)

        return two_hop_candidates

    def bfs_two_hop_trace(self, start_id: str) -> Dict[str, Any]:
        """
        ====================================================================
        DATA STRUCTURE: QUEUE (FIFO) - BREADTH-FIRST SEARCH (BFS)
        --------------------------------------------------------------------
        A step-by-step educational BFS trace showing:
          1. Queue state at each step (enqueue / dequeue)
          2. Visited set (Hash Set) preventing cycles
          3. Level-order separation:
             - Level 0: Start User
             - Level 1: Direct Friends
             - Level 2: Friends-of-Friends (Recommendation Candidates!)
        Time Complexity: O(V + E)
        Space Complexity: O(V) for Queue and Visited Set
        ====================================================================
        """
        if start_id not in self._nodes:
            return {"error": "Start node not found"}

        # QUEUE Data Structure (FIFO)
        queue = deque([(start_id, 0)])  # stores (node_id, distance_level)
        # HASH SET for O(1) membership check
        visited = {start_id}
        
        steps_log = []
        level_0 = [start_id]
        level_1_friends = []
        level_2_candidates = []

        step_counter = 1

        while queue and step_counter <= 40:
            current_node, level = queue.popleft()
            current_name = self._nodes[current_node].name if current_node in self._nodes else current_node

            neighbors_list = list(self.neighbors(current_node))
            enqueued_this_step = []

            if level < 2:
                for neighbor in neighbors_list:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level + 1))
                        enqueued_this_step.append(neighbor)
                        if level + 1 == 1:
                            level_1_friends.append(neighbor)
                        elif level + 1 == 2:
                            level_2_candidates.append(neighbor)

            steps_log.append({
                "step": step_counter,
                "action": f"DEQUEUE node '{current_name}' (ID: {current_node}) at Level {level}",
                "level": level,
                "current_node": current_node,
                "queue_size_after": len(queue),
                "enqueued_neighbors": enqueued_this_step,
                "total_visited": len(visited)
            })
            step_counter += 1

        return {
            "start_node": start_id,
            "data_structure_used": "collections.deque (FIFO Queue) + set (Hash Set)",
            "level_0_target": level_0,
            "level_1_direct_friends_count": len(level_1_friends),
            "level_1_direct_friends": level_1_friends,
            "level_2_candidates_count": len(level_2_candidates),
            "level_2_candidates": level_2_candidates,
            "trace_steps": steps_log
        }

    def dismiss_candidate(self, user_id: str, candidate_id: str) -> None:
        """Record dismissal in Hash Set. Time Complexity: O(1)."""
        if user_id in self._dismissed:
            self._dismissed[user_id].add(candidate_id)

    def is_dismissed(self, user_id: str, candidate_id: str) -> bool:
        """Check if dismissed via Hash Set lookup. Time Complexity: O(1)."""
        return candidate_id in self._dismissed.get(user_id, set())

    def local_clustering_coefficient(self, node_id: str) -> float:
        """
        Triangles check using Hash Set intersections.
        C(u) = (2 * E_u) / (k * (k - 1))
        """
        neighbors = list(self.neighbors(node_id))
        k = len(neighbors)
        if k < 2:
            return 0.0

        edges_between_neighbors = 0
        for i in range(k):
            for j in range(i + 1, k):
                if self.has_edge(neighbors[i], neighbors[j]):
                    edges_between_neighbors += 1

        possible_edges = k * (k - 1) / 2
        return edges_between_neighbors / possible_edges

    def graph_statistics(self) -> Dict[str, Any]:
        n = self.num_nodes
        m = self.num_edges
        avg_deg = (2 * m / n) if n > 0 else 0.0
        max_possible_edges = (n * (n - 1)) / 2 if n > 1 else 1
        density = m / max_possible_edges if n > 1 else 0.0

        sampled_nodes = list(self._nodes.keys())[:100]
        avg_clustering = (
            sum(self.local_clustering_coefficient(u) for u in sampled_nodes) / len(sampled_nodes)
            if sampled_nodes else 0.0
        )

        return {
            "total_users": n,
            "total_friendships": m,
            "average_degree": round(avg_deg, 2),
            "graph_density": round(density, 4),
            "average_clustering_coefficient": round(avg_clustering, 4)
        }

    def to_json_graph(self, max_nodes: int = 150) -> Dict[str, Any]:
        nodes_list = []
        node_ids = set()

        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda x: self.degree(x.id),
            reverse=True
        )[:max_nodes]

        for node in sorted_nodes:
            node_ids.add(node.id)
            nodes_list.append({
                **node.to_dict(),
                "degree": self.degree(node.id)
            })

        edges_list = []
        seen_edges = set()
        for u in node_ids:
            for v in self.neighbors(u):
                if v in node_ids:
                    edge_key = tuple(sorted([u, v]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges_list.append({
                            "source": u,
                            "target": v
                        })

        return {
            "nodes": nodes_list,
            "edges": edges_list
        }


