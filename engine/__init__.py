"""
GraphPulse Engine - Core Data Structures and Algorithms
"""
from .graph import Graph, Node
from .algorithms import (
    common_neighbors,
    jaccard_similarity,
    adamic_adar_index,
    resource_allocation_index,
    preferential_attachment,
    personalized_pagerank,
    interest_similarity
)
from .recommender import FriendRecommender, Recommendation
from .dataset import generate_synthetic_social_network

__all__ = [
    "Graph",
    "Node",
    "common_neighbors",
    "jaccard_similarity",
    "adamic_adar_index",
    "resource_allocation_index",
    "preferential_attachment",
    "personalized_pagerank",
    "interest_similarity",
    "FriendRecommender",
    "Recommendation",
    "generate_synthetic_social_network",
]

