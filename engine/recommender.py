"""
Two-Stage Hybrid Friend Recommendation Pipeline
Combines topological link prediction, Personalized PageRank,
and attribute homophily with cold-start fallback handling.
"""
import heapq
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .graph import Graph
from .algorithms import (
    common_neighbors,
    jaccard_similarity,
    adamic_adar_index,
    interest_similarity,
    personalized_pagerank
)


@dataclass
class Recommendation:
    user_id: str
    name: str
    city: str
    community: str
    avatar: str
    score: float
    mutual_friends_count: int
    mutual_friends_sample: List[Dict[str, str]]
    shared_interests: List[str]
    metrics: Dict[str, float]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "city": self.city,
            "community": self.community,
            "avatar": self.avatar,
            "score": round(self.score, 2),
            "mutual_friends_count": self.mutual_friends_count,
            "mutual_friends_sample": self.mutual_friends_sample,
            "shared_interests": self.shared_interests,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "reasons": self.reasons
        }


class FriendRecommender:
    """
    ====================================================================
    BEGINNER DSA PIPELINE: Two-Stage Recommendation Engine
    --------------------------------------------------------------------
    1. STAGE 1 (Candidate Generation):
       - Uses BREADTH-FIRST SEARCH (BFS) via Queue (collections.deque)
       - Explores Level 1 (Friends) and Level 2 (Friends-of-Friends).
       - Uses Hash Sets for O(1) visited checks and filtering.
    
    2. STAGE 2 (Top-K Ranking via Min-Heap):
       - DATA STRUCTURE: MIN-HEAP (heapq) / PRIORITY QUEUE
       - Instead of sorting all N candidates in O(N log N) time,
         we maintain a Min-Heap of size K.
       - Time Complexity: O(N log K), optimal when N is large and K is small (e.g. K=10)!
    ====================================================================
    """

    def __init__(
        self,
        graph: Graph,
        weight_adamic_adar: float = 0.35,
        weight_jaccard: float = 0.20,
        weight_ppr: float = 0.20,
        weight_interests: float = 0.15,
        weight_context: float = 0.10
    ):
        self.graph = graph
        self.w_aa = weight_adamic_adar
        self.w_jc = weight_jaccard
        self.w_ppr = weight_ppr
        self.w_int = weight_interests
        self.w_ctx = weight_context

    def _generate_candidates(self, user_id: str, min_candidates: int = 15) -> List[str]:
        """
        Stage 1: Candidate Generation via BFS 2-hop search.
        Falls back to Hash Set interest overlap if candidate pool is sparse (Cold Start).
        """
        candidates = set(self.graph.two_hop_neighbors(user_id))

        # Cold-start fallback: if 2-hop pool is insufficient
        if len(candidates) < min_candidates:
            user_node = self.graph.get_node(user_id)
            user_interests = user_node.interests if user_node else set()
            user_city = user_node.city if user_node else ""
            user_community = user_node.community if user_node else ""
            direct_friends = self.graph.neighbors(user_id)

            fallback_candidates = []
            for other_id, other_node in self.graph.nodes.items():
                if other_id == user_id or other_id in direct_friends or self.graph.is_dismissed(user_id, other_id):
                    continue

                if other_id not in candidates:
                    # Hash Set intersection for interest overlap
                    overlap = len(user_interests.intersection(other_node.interests))
                    same_city = 1 if user_city and other_node.city == user_city else 0
                    same_comm = 1 if user_community and other_node.community == user_community else 0
                    fallback_score = overlap * 2.0 + same_city * 1.5 + same_comm * 1.0 + (self.graph.degree(other_id) * 0.05)
                    fallback_candidates.append((other_id, fallback_score))

            fallback_candidates.sort(key=lambda x: x[1], reverse=True)
            for cid, _ in fallback_candidates[:(min_candidates - len(candidates))]:
                candidates.add(cid)

        return list(candidates)

    def recommend(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """
        Calculates scores and extracts Top-K using a Min-Heap (Priority Queue).
        """
        if not self.graph.has_node(user_id):
            return []

        user_node = self.graph.get_node(user_id)
        candidates = self._generate_candidates(user_id)
        if not candidates:
            return []

        # Personalized PageRank (fast approximation)
        ppr_scores = personalized_pagerank(self.graph, user_id, alpha=0.85, max_iter=15, tol=1e-4)
        max_ppr = max((ppr_scores.get(c, 0.0) for c in candidates), default=1e-6)
        if max_ppr == 0:
            max_ppr = 1e-6

        raw_results = []
        max_aa = 1e-6

        for cid in candidates:
            c_node = self.graph.get_node(cid)
            if not c_node:
                continue

            mutuals = common_neighbors(self.graph, user_id, cid)
            aa_val = adamic_adar_index(self.graph, user_id, cid)
            jc_val = jaccard_similarity(self.graph, user_id, cid)
            ppr_val = ppr_scores.get(cid, 0.0)
            int_sim = interest_similarity(self.graph, user_id, cid)

            ctx_bonus = 0.0
            if user_node and c_node.city and user_node.city == c_node.city:
                ctx_bonus += 0.5
            if user_node and c_node.community and user_node.community == c_node.community:
                ctx_bonus += 0.5

            if aa_val > max_aa:
                max_aa = aa_val

            raw_results.append({
                "cid": cid,
                "node": c_node,
                "mutuals": mutuals,
                "aa": aa_val,
                "jc": jc_val,
                "ppr": ppr_val,
                "int_sim": int_sim,
                "ctx": ctx_bonus
            })

        # -------------------------------------------------------------
        # DATA STRUCTURE: MIN-HEAP / PRIORITY QUEUE for Top-K Selection
        # -------------------------------------------------------------
        # We push tuples: (score, counter, Recommendation) into a min-heap.
        # If heap exceeds size K, heapq.heappop removes the smallest score.
        # This gives O(N log K) time complexity instead of O(N log N) sorting!
        # -------------------------------------------------------------
        top_k_heap: List[Tuple[float, int, Recommendation]] = []
        tie_breaker = 0

        for item in raw_results:
            cid = item["cid"]
            c_node = item["node"]
            mutuals = item["mutuals"]

            norm_aa = item["aa"] / max_aa if max_aa > 0 else 0.0
            norm_jc = item["jc"]
            norm_ppr = item["ppr"] / max_ppr if max_ppr > 0 else 0.0
            norm_int = item["int_sim"]
            norm_ctx = item["ctx"]

            composite_score = (
                self.w_aa * norm_aa +
                self.w_jc * norm_jc +
                self.w_ppr * norm_ppr +
                self.w_int * norm_int +
                self.w_ctx * norm_ctx
            ) * 100.0

            mutual_samples = []
            for m_id in list(mutuals)[:3]:
                m_node = self.graph.get_node(m_id)
                if m_node:
                    mutual_samples.append({"id": m_id, "name": m_node.name})

            shared_interests = sorted(list(
                user_node.interests.intersection(c_node.interests)
            )) if user_node else []

            reasons = []
            if len(mutuals) > 0:
                sample_names = ", ".join(m["name"] for m in mutual_samples)
                extra = f" (+{len(mutuals) - len(mutual_samples)} more)" if len(mutuals) > len(mutual_samples) else ""
                reasons.append(f"{len(mutuals)} mutual friend{'s' if len(mutuals) > 1 else ''}: {sample_names}{extra}")

            if shared_interests:
                reasons.append(f"Shared interests: {', '.join(shared_interests[:3])}")

            if user_node and c_node.city and user_node.city == c_node.city:
                reasons.append(f"Lives in {c_node.city}")

            if user_node and c_node.community and user_node.community == c_node.community:
                reasons.append(f"Part of {c_node.community} community")

            if not reasons:
                reasons.append("Active member in your extended network")

            rec = Recommendation(
                user_id=cid,
                name=c_node.name,
                city=c_node.city,
                community=c_node.community,
                avatar=c_node.avatar,
                score=composite_score,
                mutual_friends_count=len(mutuals),
                mutual_friends_sample=mutual_samples,
                shared_interests=shared_interests,
                metrics={
                    "adamic_adar": item["aa"],
                    "jaccard": item["jc"],
                    "ppr_score": item["ppr"],
                    "interest_similarity": item["int_sim"]
                },
                reasons=reasons
            )

            # Push into Min-Heap
            tie_breaker += 1
            if len(top_k_heap) < limit:
                heapq.heappush(top_k_heap, (composite_score, tie_breaker, rec))
            else:
                if composite_score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (composite_score, tie_breaker, rec))

        # Extract from heap in descending order of score
        result = [item[2] for item in sorted(top_k_heap, key=lambda x: x[0], reverse=True)]
        return result

