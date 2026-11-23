"""
GraphPulse REST API Server
Zero external dependencies (uses standard library http.server).
Provides RESTful endpoints for graph queries, recommendations, friend creation,
and real-time feedback loop.
"""
import http.server
import socketserver
import json
import os
import urllib.parse
from typing import Dict, Any

from engine.dataset import generate_synthetic_social_network
from engine.recommender import FriendRecommender
from engine.algorithms import (
    common_neighbors,
    jaccard_similarity,
    adamic_adar_index,
    resource_allocation_index,
    preferential_attachment,
    personalized_pagerank,
    interest_similarity
)

# Global State
GRAPH = generate_synthetic_social_network()
RECOMMENDER = FriendRecommender(GRAPH)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


class GraphPulseHandler(http.server.SimpleHTTPRequestHandler):

    def send_json(self, data: Any, status: int = 200):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Static file routing
        if path == "/" or path == "/index.html":
            index_path = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # API Endpoints
        if path == "/api/users":
            users = []
            for uid, node in GRAPH.nodes.items():
                users.append({
                    **node.to_dict(),
                    "degree": GRAPH.degree(uid),
                    "is_cold_start": node.metadata.get("cold_start", False),
                    "is_hub": node.metadata.get("is_hub", False)
                })
            # Sort by name
            users.sort(key=lambda u: u["name"])
            self.send_json({"users": users})
            return

        if path.startswith("/api/user/"):
            user_id = path.replace("/api/user/", "").strip()
            node = GRAPH.get_node(user_id)
            if not node:
                self.send_json({"error": "User not found"}, status=404)
                return

            friends = [
                {
                    "id": fid,
                    "name": GRAPH.get_node(fid).name,
                    "city": GRAPH.get_node(fid).city,
                    "community": GRAPH.get_node(fid).community
                }
                for fid in GRAPH.neighbors(user_id) if GRAPH.get_node(fid)
            ]

            data = {
                **node.to_dict(),
                "degree": GRAPH.degree(user_id),
                "friends": friends,
                "clustering_coefficient": round(GRAPH.local_clustering_coefficient(user_id), 4)
            }
            self.send_json(data)
            return

        if path == "/api/graph":
            max_nodes = int(query.get("limit", [100])[0])
            json_graph = GRAPH.to_json_graph(max_nodes=max_nodes)
            self.send_json(json_graph)
            return

        if path.startswith("/api/recommendations/"):
            user_id = path.replace("/api/recommendations/", "").strip()
            limit = int(query.get("limit", [8])[0])

            if not GRAPH.has_node(user_id):
                self.send_json({"error": "User not found"}, status=404)
                return

            recs = RECOMMENDER.recommend(user_id, limit=limit)
            self.send_json({
                "user_id": user_id,
                "total_recommendations": len(recs),
                "recommendations": [r.to_dict() for r in recs]
            })
            return

        if path.startswith("/api/dsa/trace/"):
            user_id = path.replace("/api/dsa/trace/", "").strip()
            if not GRAPH.has_node(user_id):
                self.send_json({"error": "User not found"}, status=404)
                return
            trace = GRAPH.bfs_two_hop_trace(user_id)
            self.send_json(trace)
            return

        if path.startswith("/api/dsa/adj/"):
            user_id = path.replace("/api/dsa/adj/", "").strip()
            if not GRAPH.has_node(user_id):
                self.send_json({"error": "User not found"}, status=404)
                return
            neighbors = list(GRAPH.neighbors(user_id))
            subgraph = {
                user_id: neighbors
            }
            for n in neighbors:
                subgraph[n] = list(GRAPH.neighbors(n))
            self.send_json({
                "target_user": user_id,
                "data_structure": "Dictionary of Sets (Adjacency List)",
                "adjacency_list": subgraph
            })
            return

        if path == "/api/analytics":
            stats = GRAPH.graph_statistics()
            self.send_json(stats)
            return

        if path == "/api/compare":
            u = query.get("u", [None])[0]
            v = query.get("v", [None])[0]
            if not u or not v or not GRAPH.has_node(u) or not GRAPH.has_node(v):
                self.send_json({"error": "Provide valid u and v parameters"}, status=400)
                return

            mutuals = list(common_neighbors(GRAPH, u, v))
            mutual_names = [GRAPH.get_node(m).name for m in mutuals if GRAPH.get_node(m)]

            data = {
                "u": GRAPH.get_node(u).to_dict(),
                "v": GRAPH.get_node(v).to_dict(),
                "has_direct_edge": GRAPH.has_edge(u, v),
                "common_neighbors_count": len(mutuals),
                "common_neighbors": mutual_names,
                "jaccard_similarity": round(jaccard_similarity(GRAPH, u, v), 4),
                "adamic_adar_index": round(adamic_adar_index(GRAPH, u, v), 4),
                "resource_allocation_index": round(resource_allocation_index(GRAPH, u, v), 4),
                "preferential_attachment": preferential_attachment(GRAPH, u, v),
                "interest_similarity": round(interest_similarity(GRAPH, u, v), 4)
            }
            self.send_json(data)
            return

        # 404
        self.send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        global GRAPH, RECOMMENDER
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/friend":
            u = payload.get("user_id")
            v = payload.get("target_id")
            if not u or not v:
                self.send_json({"error": "Missing user_id or target_id"}, status=400)
                return

            success = GRAPH.add_edge(u, v)
            self.send_json({
                "success": success,
                "message": f"Friendship established between {u} and {v}" if success else "Edge could not be created",
                "u_degree": GRAPH.degree(u),
                "v_degree": GRAPH.degree(v)
            })
            return

        if path == "/api/dismiss":
            user_id = payload.get("user_id")
            candidate_id = payload.get("candidate_id")
            if not user_id or not candidate_id:
                self.send_json({"error": "Missing user_id or candidate_id"}, status=400)
                return

            GRAPH.dismiss_candidate(user_id, candidate_id)
            self.send_json({
                "success": True,
                "message": f"Candidate {candidate_id} dismissed for {user_id}"
            })
            return

        if path == "/api/reset":
            GRAPH = generate_synthetic_social_network()
            RECOMMENDER = FriendRecommender(GRAPH)
            self.send_json({"success": True, "message": "Graph reset to initial synthetic state."})
            return

        self.send_json({"error": "Not Found"}, status=404)


def run_server(port: int = 8000):
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ("", port)
    httpd = socketserver.TCPServer(server_address, GraphPulseHandler)
    print(f"============================================================")
    print(f"  GraphPulse Server running at: http://localhost:{port}/")
    print(f"  REST API available at:       http://localhost:{port}/api/")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
