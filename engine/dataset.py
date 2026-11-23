"""
Realistic Social Network Synthesizer
Generates community-structured graph topology with homophily,
supernode hubs, bridge nodes, and cold-start newcomers.
"""
import random
from typing import List, Dict, Set
from .graph import Graph, Node

COMMUNITIES = {
    "AI & Engineering": {
        "cities": ["San Francisco", "Seattle", "Bengaluru", "Berlin"],
        "interests": ["Python", "Machine Learning", "System Design", "Cloud Computing", "Rust", "DevOps", "Graph Neural Networks"],
        "names": [
            "Aarav Sharma", "Elena Rostova", "Marcus Chen", "Sophia Taylor",
            "Liam Vance", "Priya Nair", "Alexander Wright", "Hana Tanaka",
            "Julian Becker", "Maya Patel", "David Kim", "Chloe Dupont"
        ]
    },
    "Design & Product": {
        "cities": ["New York", "London", "Stockholm", "Amsterdam"],
        "interests": ["UI/UX Design", "Product Management", "Design Systems", "Figma", "User Research", "Typography", "Branding"],
        "names": [
            "Isabella Rossi", "Lucas Morales", "Zoe Jenkins", "Oliver Smith",
            "Mia Lindqvist", "Noah Dubois", "Ava Johansson", "Ethan Brooks",
            "Camila Silva", "Leo Martin", "Amara Okafor", "Felix Weber"
        ]
    },
    "Fintech & Crypto": {
        "cities": ["Chicago", "Zurich", "Singapore", "Hong Kong"],
        "interests": ["Algorithmic Trading", "Blockchain", "Quantitative Finance", "DeFi", "Macroeconomics", "Risk Analysis"],
        "names": [
            "Benjamin Harris", "Seraphina Lin", "Viktor Novak", "Astrid Hansen",
            "Gabriel Mendoza", "Emily Zhang", "Klaus Fischer", "Tariq Mansour"
        ]
    },
    "BioTech & Science": {
        "cities": ["Boston", "Cambridge", "Oxford", "Tokyo"],
        "interests": ["Genomics", "Bioinformatics", "Neuroscience", "CRISPR", "Data Science", "Molecular Biology"],
        "names": [
            "Dr. Aris Thorne", "Dr. Rachel Green", "Dr. Kenji Sato", "Dr. Fatimah Al-Sayed",
            "Dr. Samuel Levin", "Dr. Beatrice Bell"
        ]
    }
}

COLD_START_USERS = [
    {
        "id": "newbie_sam",
        "name": "Sam Rivers (New User)",
        "city": "San Francisco",
        "community": "AI & Engineering",
        "interests": {"Machine Learning", "Python", "Cloud Computing"}
    },
    {
        "id": "newbie_jess",
        "name": "Jessica Day (New User)",
        "city": "New York",
        "community": "Design & Product",
        "interests": {"UI/UX Design", "Figma", "User Research"}
    }
]


def generate_synthetic_social_network(seed: int = 42) -> Graph:
    """
    Generates a realistic social graph with:
    1. Homophilic community clusters
    2. Inter-community bridges
    3. High-degree hub accounts (celebrities/influencers)
    4. Cold-start isolated or near-isolated users
    """
    random.seed(seed)
    graph = Graph()

    community_nodes: Dict[str, List[str]] = {comm: [] for comm in COMMUNITIES}
    node_counter = 1

    # 1. Create nodes for each community
    for comm_name, data in COMMUNITIES.items():
        for name in data["names"]:
            uid = f"user_{node_counter:03d}"
            city = random.choice(data["cities"])
            # Sample 3-5 interests
            user_interests = set(random.sample(data["interests"], k=min(4, len(data["interests"]))))

            node = Node(
                id=uid,
                name=name,
                city=city,
                interests=user_interests,
                community=comm_name
            )
            graph.add_node(node)
            community_nodes[comm_name].append(uid)
            node_counter += 1

    # 2. Add intra-community edges (dense connections)
    for comm_name, uids in community_nodes.items():
        k = len(uids)
        for i in range(k):
            # Connect each person to 3-6 other members within community
            num_connections = random.randint(3, min(6, k - 1))
            potential_targets = [u for u in uids if u != uids[i]]
            targets = random.sample(potential_targets, k=min(num_connections, len(potential_targets)))
            for target in targets:
                graph.add_edge(uids[i], target)

    # 3. Create a "Supernode" / Hub in AI & Engineering (High-degree influencer)
    hub_id = community_nodes["AI & Engineering"][0]
    hub_node = graph.get_node(hub_id)
    if hub_node:
        hub_node.name = f"{hub_node.name} (Tech Keynote Speaker)"
        hub_node.metadata["is_hub"] = True
        # Connect hub to 80% of members across all communities
        for comm, uids in community_nodes.items():
            for uid in uids:
                if uid != hub_id and random.random() < 0.65:
                    graph.add_edge(hub_id, uid)

    # 4. Add inter-community bridge edges (sparse cross-discipline connections)
    all_uids = list(graph.nodes.keys())
    for _ in range(25):
        u1, u2 = random.sample(all_uids, 2)
        if graph.get_node(u1).community != graph.get_node(u2).community:
            graph.add_edge(u1, u2)

    # 5. Add Cold-Start Users
    for cold_data in COLD_START_USERS:
        cold_node = Node(
            id=cold_data["id"],
            name=cold_data["name"],
            city=cold_data["city"],
            community=cold_data["community"],
            interests=cold_data["interests"],
            metadata={"cold_start": True}
        )
        graph.add_node(cold_node)

    # Connect newbie_sam to exactly 1 friend in AI to simulate single-edge sparse state
    if community_nodes["AI & Engineering"]:
        single_friend = community_nodes["AI & Engineering"][2]
        graph.add_edge("newbie_sam", single_friend)

    # newbie_jess has 0 friends initially to test 0-hop cold-start fallback

    return graph

