"""
================================================================================
🎓 BEGINNER DSA TUTORIAL: How Social Network Friend Suggestions Work
================================================================================
This script breaks down the 4 core Data Structures used in real social networks
(like Facebook, LinkedIn, Instagram) in simple, beginner-friendly terms:

  1. GRAPH (Adjacency List) -> Represents users and friendships
  2. QUEUE (FIFO via collections.deque) -> Explores friends-of-friends (BFS)
  3. HASH SET (set) -> Computes mutual friends instantly in O(1) average time
  4. MIN-HEAP (heapq) -> Keeps top 3 recommendations without sorting everything

Run this file directly:
  python beginner_dsa_tutorial.py
================================================================================
"""
from collections import deque
import heapq
import math


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ==============================================================================
# LESSON 1: THE GRAPH (ADJACENCY LIST)
# ==============================================================================
section("LESSON 1: The Graph Representation (Adjacency List)")

print("""
A Graph consists of:
  - VERTICES (Nodes): The people in the social network (e.g. Alice, Bob)
  - EDGES (Links): Friendships connecting two people

How do we store this in code?
We use a Python DICTIONARY (Hash Map) where:
  - Key = Person's Name
  - Value = A SET of their direct friends
""")

social_graph = {
    "Alice":   {"Bob", "Charlie", "David"},
    "Bob":     {"Alice", "David", "Elena"},
    "Charlie": {"Alice", "David", "Frank"},
    "David":   {"Alice", "Bob", "Charlie", "Grace"},
    "Elena":   {"Bob", "Grace", "Henry"},
    "Frank":   {"Charlie", "Henry"},
    "Grace":   {"David", "Elena", "Henry"},
    "Henry":   {"Elena", "Frank", "Grace"}
}

for person, friends in social_graph.items():
    print(f"  {person:8} has {len(friends)} friends -> {friends}")

print("\n💡 DSA Insight:")
print("  Checking if Alice is friends with Bob takes O(1) time: 'Bob' in social_graph['Alice']")
print(f"  Is Alice friends with Bob? -> {'Bob' in social_graph['Alice']}")
print(f"  Is Alice friends with Elena? -> {'Elena' in social_graph['Alice']}")


# ==============================================================================
# LESSON 2: BFS (BREADTH-FIRST SEARCH) USING A QUEUE
# ==============================================================================
section("LESSON 2: Finding Candidates with BFS (Queue Data Structure)")

print("""
To recommend friends to Alice, we look at 'Friends of Friends' (Distance = 2).
We use BREADTH-FIRST SEARCH (BFS) with a QUEUE (First-In, First-Out):
  - Level 0: Alice (Distance 0)
  - Level 1: Alice's Direct Friends (Distance 1)
  - Level 2: Friends of Friends (Distance 2) -> OUR CANDIDATES!
""")

start_user = "Alice"
queue = deque([(start_user, 0)])  # Queue holds: (person, distance_level)
visited = {start_user}            # Hash Set to avoid visiting the same person twice
candidates = set()

step = 1
print(f"Starting BFS from '{start_user}'...")
while queue:
    current_person, level = queue.popleft()  # Dequeue from the front
    print(f"  Step {step}: Dequeued '{current_person}' (Level {level})")

    # If we are at level 1, explore their friends (which will be level 2)
    if level < 2:
        for neighbor in social_graph[current_person]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))  # Enqueue to the back
                if level + 1 == 2:
                    candidates.add(neighbor)
                    print(f"    --> Discovered Level 2 Candidate: '{neighbor}'!")
    step += 1

print(f"\nFinal Candidates for {start_user} (at distance 2): {candidates}")


# ==============================================================================
# LESSON 3: HASH SET INTERSECTION FOR MUTUAL FRIENDS
# ==============================================================================
section("LESSON 3: Finding Mutual Friends with Set Intersection")

print("""
How do we find who Alice and a candidate both know?
We use HASH SET INTERSECTION:
  mutual_friends = social_graph['Alice'] & social_graph[candidate]

This is extremely fast: O(min(|A|, |B|))!
""")

scored_candidates = []

for candidate in candidates:
    # 1. Mutual friends via set intersection
    mutuals = social_graph[start_user] & social_graph[candidate]

    # 2. Jaccard Similarity = Mutual Friends / Total Unique Friends
    union_friends = social_graph[start_user] | social_graph[candidate]
    jaccard = len(mutuals) / len(union_friends)

    # 3. Adamic-Adar Score = Sum of 1 / log(degree of mutual friend)
    # Penalizes popular hubs!
    adamic_adar = sum(1.0 / math.log(max(len(social_graph[m]), 2)) for m in mutuals)

    scored_candidates.append({
        "name": candidate,
        "mutual_count": len(mutuals),
        "mutual_friends": mutuals,
        "jaccard": round(jaccard, 3),
        "adamic_adar": round(adamic_adar, 3)
    })

    print(f"  Candidate: {candidate:8} | Mutual Friends: {len(mutuals)} ({mutuals}) | Adamic-Adar: {adamic_adar:.3f}")


# ==============================================================================
# LESSON 4: TOP-K SELECTION USING A MIN-HEAP (PRIORITY QUEUE)
# ==============================================================================
section("LESSON 4: Extracting Top-2 Recommendations via Min-Heap")

print("""
Suppose we have 100,000 candidates. Sorting them all takes O(N log N).
Instead, we maintain a MIN-HEAP of size K (e.g., K = 2).
  - Time Complexity: O(N log K) -> Much faster!
  - If a new candidate has a higher score than the heap's minimum, we replace it.
""")

K = 2
min_heap = []  # stores (score, candidate_name)

for c in scored_candidates:
    score = c["adamic_adar"]
    name = c["name"]

    if len(min_heap) < K:
        heapq.heappush(min_heap, (score, name))
        print(f"  Heap size < {K}: Pushed ({score}, '{name}')")
    else:
        # If score is better than the smallest item in the heap
        if score > min_heap[0][0]:
            removed = heapq.heapreplace(min_heap, (score, name))
            print(f"  Higher score found! Replaced {removed} with ({score}, '{name}')")
        else:
            print(f"  Score {score} for '{name}' is not higher than heap min {min_heap[0][0]}, skipped.")

# Extract top K sorted
top_recommendations = sorted(min_heap, key=lambda x: x[0], reverse=True)

print(f"\n🏆 Final Top-{K} Friend Recommendations for {start_user}:")
for rank, (score, name) in enumerate(top_recommendations, 1):
    print(f"   #{rank}: {name} (Adamic-Adar Score: {score})")


# ==============================================================================
# SUMMARY TABLE OF DATA STRUCTURES & COMPLEXITIES
# ==============================================================================
section("SUMMARY: Data Structures & Complexity Reference Card")

print("""
+---------------------------+-----------------------+------------------+------------------------------+
| Data Structure            | Python Type           | Time Complexity  | Purpose in Friend System     |
+---------------------------+-----------------------+------------------+------------------------------+
| Graph (Adjacency List)    | dict of sets          | O(1) lookup      | Fast neighbor retrieval      |
| Queue (FIFO)              | collections.deque     | O(1) push/pop    | BFS 2-hop candidate search   |
| Hash Set (Visited)        | set                   | O(1) membership  | Prevents infinite loops/dups |
| Hash Set (Intersection)   | set_A & set_B         | O(min(|A|, |B|)) | Finds mutual friends         |
| Min-Heap (Priority Queue) | heapq (list)          | O(log K) push    | Top-K recommendation ranking |
+---------------------------+-----------------------+------------------+------------------------------+
""")
print("✅ Tutorial completed! You now understand the core DSA of friend suggestion systems.\n")

