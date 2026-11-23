"""
Integration tests for GraphPulse REST API server endpoints
"""
import urllib.request
import json
import threading
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server import run_server

def test_api():
    t = threading.Thread(target=run_server, args=(8088,), daemon=True)
    t.start()
    time.sleep(1.2)

    # 1. Analytics
    with urllib.request.urlopen("http://localhost:8088/api/analytics") as resp:
        data = json.loads(resp.read().decode())
        print(f"[OK] Analytics: {data['total_users']} users, {data['total_friendships']} edges, clustering: {data['average_clustering_coefficient']}")
        assert data['total_users'] > 0

    # 2. Users list
    with urllib.request.urlopen("http://localhost:8088/api/users") as resp:
        data = json.loads(resp.read().decode())
        print(f"[OK] Users endpoint: {len(data['users'])} users returned")
        assert len(data['users']) > 0

    # 3. Recommendations for user
    with urllib.request.urlopen("http://localhost:8088/api/recommendations/user_002?limit=3") as resp:
        data = json.loads(resp.read().decode())
        print(f"[OK] Recommendations for user_002: {data['total_recommendations']} recs. Top: {data['recommendations'][0]['name']} (Score: {data['recommendations'][0]['score']}%)")
        assert len(data['recommendations']) > 0

    # 4. Cold-start user
    with urllib.request.urlopen("http://localhost:8088/api/recommendations/newbie_jess?limit=3") as resp:
        data = json.loads(resp.read().decode())
        print(f"[OK] Cold-start for newbie_jess: {data['total_recommendations']} recs. Reasons: {data['recommendations'][0]['reasons']}")
        assert len(data['recommendations']) > 0

    # 5. Algorithm Comparison
    with urllib.request.urlopen("http://localhost:8088/api/compare?u=user_001&v=user_002") as resp:
        data = json.loads(resp.read().decode())
        print(f"[OK] Comparison: Common Neighbors={data['common_neighbors_count']}, Adamic-Adar={data['adamic_adar_index']}, Jaccard={data['jaccard_similarity']}")
        assert "adamic_adar_index" in data

    # 6. Static HTML
    with urllib.request.urlopen("http://localhost:8088/") as resp:
        html = resp.read().decode()
        print(f"[OK] Static Web Dashboard: {len(html)} bytes served with HTTP 200")
        assert "<title>GraphPulse" in html

    print("\n>>> ALL API & INTEGRATION CHECKS PASSED WITH 100% SUCCESS! <<<")
    os._exit(0)

if __name__ == "__main__":
    test_api()
