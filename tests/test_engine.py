"""
Comprehensive Unit Tests for GraphPulse Engine
"""
import unittest
import math
from engine.graph import Graph, Node
from engine.algorithms import (
    common_neighbors,
    jaccard_similarity,
    adamic_adar_index,
    resource_allocation_index,
    preferential_attachment,
    interest_similarity,
    personalized_pagerank
)
from engine.recommender import FriendRecommender
from engine.dataset import generate_synthetic_social_network


class TestGraphEngine(unittest.TestCase):

    def setUp(self):
        self.g = Graph()
        # Create a simple toy graph:
        # A - B - C
        # |   |
        # D - E
        nodes = ["A", "B", "C", "D", "E"]
        for nid in nodes:
            self.g.add_node(Node(id=nid, name=f"User {nid}"))

        self.g.add_edge("A", "B")
        self.g.add_edge("B", "C")
        self.g.add_edge("A", "D")
        self.g.add_edge("B", "E")
        self.g.add_edge("D", "E")

    def test_basic_graph_properties(self):
        self.assertEqual(self.g.num_nodes, 5)
        self.assertEqual(self.g.num_edges, 5)
        self.assertTrue(self.g.has_edge("A", "B"))
        self.assertTrue(self.g.has_edge("B", "A"))
        self.assertFalse(self.g.has_edge("A", "C"))
        self.assertEqual(self.g.degree("B"), 3)  # A, C, E
        self.assertEqual(self.g.degree("C"), 1)  # B

    def test_common_neighbors(self):
        # A and C share neighbor B
        cn_ac = common_neighbors(self.g, "A", "C")
        self.assertEqual(cn_ac, {"B"})

        # A and E share B and D
        cn_ae = common_neighbors(self.g, "A", "E")
        self.assertEqual(cn_ae, {"B", "D"})

    def test_jaccard_similarity(self):
        # Neighbors of A: {B, D} (size 2)
        # Neighbors of E: {B, D} (size 2)
        # Intersection: {B, D} (size 2), Union: {B, D} (size 2)
        # Jaccard = 2 / 2 = 1.0
        j_ae = jaccard_similarity(self.g, "A", "E")
        self.assertAlmostEqual(j_ae, 1.0)

        # Neighbors of A: {B, D}
        # Neighbors of C: {B}
        # Intersection: {B} (1), Union: {B, D} (2)
        # Jaccard = 1 / 2 = 0.5
        j_ac = jaccard_similarity(self.g, "A", "C")
        self.assertAlmostEqual(j_ac, 0.5)

    def test_adamic_adar_supernode_penalty(self):
        # Create a hub node H connected to 50 nodes
        # Create an intimate friend F connected to only 2 nodes
        test_g = Graph()
        for i in range(60):
            test_g.add_node(Node(id=f"n_{i}", name=f"Node {i}"))

        # User X and Candidate Y
        test_g.add_node(Node(id="X", name="User X"))
        test_g.add_node(Node(id="Y", name="Candidate Y"))
        test_g.add_node(Node(id="Z", name="Candidate Z"))

        # F has degree 2 (X and Y)
        test_g.add_node(Node(id="F", name="Intimate Friend"))
        test_g.add_edge("X", "F")
        test_g.add_edge("Y", "F")

        # H has degree 50 (X, Z, and 48 others)
        test_g.add_node(Node(id="H", name="Celebrity Hub"))
        test_g.add_edge("X", "H")
        test_g.add_edge("Z", "H")
        for i in range(48):
            test_g.add_edge("H", f"n_{i}")

        # Both (X, Y) and (X, Z) share exactly 1 common neighbor
        self.assertEqual(len(common_neighbors(test_g, "X", "Y")), 1)
        self.assertEqual(len(common_neighbors(test_g, "X", "Z")), 1)

        # Adamic-Adar of (X, Y) should be much HIGHER than (X, Z) because F is not a hub!
        aa_xy = adamic_adar_index(test_g, "X", "Y")
        aa_xz = adamic_adar_index(test_g, "X", "Z")

        # 1 / log(2) ≈ 1.44 vs 1 / log(50) ≈ 0.25
        self.assertGreater(aa_xy, aa_xz * 3.0)

    def test_personalized_pagerank(self):
        ppr = personalized_pagerank(self.g, source="A", alpha=0.85)
        # Source A should have a high score due to restart probability
        self.assertIn("A", ppr)
        self.assertGreater(ppr["B"], ppr["C"])  # Direct neighbor B higher than 2-hop C

    def test_recommender_pipeline_and_feedback(self):
        rec_engine = FriendRecommender(self.g)
        recs = rec_engine.recommend("A", limit=5)

        # A is friends with B and D. 2-hop candidates should be C and E.
        rec_ids = [r.user_id for r in recs]
        self.assertIn("C", rec_ids)
        self.assertIn("E", rec_ids)

        # Verify reasons are populated
        self.assertTrue(len(recs[0].reasons) > 0)

        # Test dismissal feedback loop: dismiss 'E'
        self.g.dismiss_candidate("A", "E")
        self.assertTrue(self.g.is_dismissed("A", "E"))

        recs_after_dismiss = rec_engine.recommend("A", limit=5)
        rec_ids_after = [r.user_id for r in recs_after_dismiss]
        self.assertNotIn("E", rec_ids_after)

    def test_bfs_trace_data_structure(self):
        trace = self.g.bfs_two_hop_trace("A")
        self.assertEqual(trace["start_node"], "A")
        self.assertEqual(trace["level_1_direct_friends_count"], 2)  # B and D
        self.assertIn("C", trace["level_2_candidates"])
        self.assertIn("E", trace["level_2_candidates"])
        self.assertGreater(len(trace["trace_steps"]), 0)

    def test_cold_start_fallback(self):
        # Generate full synthetic network with cold-start users
        full_g = generate_synthetic_social_network()
        rec_engine = FriendRecommender(full_g)

        # newbie_jess has 0 friends
        self.assertEqual(full_g.degree("newbie_jess"), 0)
        recs = rec_engine.recommend("newbie_jess", limit=5)

        # Should not crash and should return relevant recommendations based on interests / community
        self.assertGreater(len(recs), 0)
        top_rec = recs[0]
        # Should have shared interests or community
        self.assertTrue(len(top_rec.reasons) > 0)


if __name__ == "__main__":
    unittest.main()

