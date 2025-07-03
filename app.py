import pickle
import asyncio
import numpy as np
import networkx as nx
from flask import Flask, render_template, jsonify, request
import os
import json
import threading
import queue
from concurrent.futures import Future
import time

# --- NEW: Lightweight libraries for HTTP calls and Authentication ---
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# --- Environment and Vertex AI Configuration ---
# All Google Cloud client libraries are removed to save memory.
# We only need the configuration values.

CREDENTIALS_FILENAME = "commanding-fact-441820-j9-0e1712201ab6.json"
RENDER_CREDENTIALS_PATH = f"/etc/secrets/{CREDENTIALS_FILENAME}"
# RENDER_CREDENTIALS_PATH = CREDENTIALS_FILENAME
GOOGLE_APPLICATION_CREDENTIALS = ""

if os.path.exists(RENDER_CREDENTIALS_PATH):
    GOOGLE_APPLICATION_CREDENTIALS = RENDER_CREDENTIALS_PATH
    print(f"✅ Credentials found on Render at: {GOOGLE_APPLICATION_CREDENTIALS}")
else:
    # Fallback for local development
    if os.path.exists(CREDENTIALS_FILENAME):
        GOOGLE_APPLICATION_CREDENTIALS = CREDENTIALS_FILENAME
        print(f"✅ Credentials found locally: {GOOGLE_APPLICATION_CREDENTIALS}")
    else:
        print(f"❌ FATAL ERROR: Credentials file not found at {RENDER_CREDENTIALS_PATH} or locally.")
        exit()

PROJECT_ID = "commanding-fact-441820-j9"
LOCATION = "us-central1" # Important for constructing REST URLs
MODEL_ID = "text-multilingual-embedding-002"

# Vector Search Configuration
API_ENDPOINT = "1028163771.us-central1-856708660097.vdb.vertexai.goog"
INDEX_ENDPOINT_PATH = "projects/856708660097/locations/us-central1/indexEndpoints/6254692840882831360"
DEPLOYED_INDEX_ID = "my_narrs_emb_index_display_1751478778959"


# --- NEW: Lightweight Authentication Handler ---
_cached_token = None
_token_expiry = 0
_token_lock = threading.Lock()

def get_gcloud_token():
    """
    Gets a Google Cloud access token, caching it until it expires.
    This is thread-safe and highly efficient.
    """
    global _cached_token, _token_expiry
    with _token_lock:
        # If token is valid for at least 1 more minute, return it
        if _cached_token and _token_expiry > time.time() + 60:
            return _cached_token

        print("🚀 GCloud token expired or not found. Refreshing...")
        try:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            creds = service_account.Credentials.from_service_account_file(
                GOOGLE_APPLICATION_CREDENTIALS, scopes=scopes
            )
            creds.refresh(Request())
            _cached_token = creds.token
            _token_expiry = creds.expiry.timestamp()
            print("✅ New GCloud token acquired.")
            return _cached_token
        except Exception as e:
            print(f"❌ FATAL ERROR: Could not get Google Cloud auth token: {e}")
            raise


# --- NEW: Async HTTP-based Embedding Function ---
async def embed_text_http(texts: list, task: str = "RETRIEVAL_QUERY"):
    """
    Gets text embeddings by calling the Vertex AI REST API directly.
    """
    try:
        token = get_gcloud_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        # Note: The embedding model for search/retrieval is different than for clustering
        # We will use RETRIEVAL_QUERY for search, and RETRIEVAL_DOCUMENT for indexing
        # but for this app's purpose, query is fine.
        payload = {
            "instances": [{"task_type": task, "content": text} for text in texts]
        }
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predict"

        # Use a session for potential connection pooling
        async with requests.AsyncSession() as session:
            response = await session.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            json_response = response.json()
            embeddings = [np.array(p['embeddings']['values']) for p in json_response['predictions']]
            return embeddings

    except Exception as e:
        print(f"❌ Error during HTTP embedding call: {e}")
        # In a real app, you might want to check response.text for more detailed error messages from the API
        raise

# --- Thread-Safe Async Embedding Setup (Modified to use HTTP) ---
embedding_queue = queue.Queue()

def embedding_worker():
    # We need to import requests here for the async version to work in a new thread
    import httpx
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def aembed_text(texts: list, task: str = "RETRIEVAL_QUERY"):
        try:
            token = get_gcloud_token()
            headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8" }
            payload = { "instances": [{"task_type": task, "content": text} for text in texts] }
            url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predict"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                json_response = response.json()
                embeddings = [p['embeddings']['values'] for p in json_response['predictions']]
                return np.array(embeddings)
        except Exception as e:
            print(f"Error during async HTTP embedding call: {e}")
            raise

    async def main_loop():
        while True:
            future, texts = embedding_queue.get()
            print(f"Worker received job to embed {len(texts)} text(s) via HTTP.")
            try:
                # Use the new HTTP-based async function
                result = await aembed_text(texts)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    loop.run_until_complete(main_loop())

worker_thread = threading.Thread(target=embedding_worker, daemon=True)
worker_thread.start()
print("✅ Embedding worker thread started (using lightweight HTTP).")


# --- App Initialization and Data Loading (Using lightweight graph) ---
app = Flask(__name__)

LIGHTWEIGHT_GRAPH_FILE = "saved_components_tree_processed_full.pkl"
try:
    with open(LIGHTWEIGHT_GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    print(f"✅ Successfully loaded lightweight graph data from '{LIGHTWEIGHT_GRAPH_FILE}'.")
except FileNotFoundError:
    print(f"❌ ERROR: '{LIGHTWEIGHT_GRAPH_FILE}' not found. Please run the pruning script first.")
    # Create a dummy graph to allow the app to run
    G = nx.DiGraph()
    G.add_node("outter")

print(f"✅ Graph loaded with {len(G.nodes())} nodes.")

def get_node_display_text(graph, node_id):
    if not graph.has_node(node_id):
        return f"Unknown Node: {node_id}"
    node_data = graph.nodes[node_id]
    base_text = node_data.get('text', node_id)
    if 'uploaded_at_avg' in node_data and node_data['uploaded_at_avg']:
        date_str = str(node_data['uploaded_at_avg']).split(' ')[0]
        return f"{base_text} ({date_str})"
    return base_text

# --- API Endpoints (Largely Unchanged) ---

@app.route('/')
def index():
    return render_template('index.html')

def create_node_dict(graph, node_id):
    is_branch = graph.out_degree(node_id) > 0
    node_data = graph.nodes[node_id]
    payload_data = {k: str(v) if v is not None else '' for k, v in node_data.items()}

    return {
        'id': node_id,
        'text': get_node_display_text(graph, node_id),
        'has_children': is_branch,
        'data': payload_data
    }

@app.route('/api/nodes/')
def get_root_nodes():
    children_data = [create_node_dict(G, child_id) for child_id in G.successors('outter')]
    return jsonify(children_data)

@app.route('/api/nodes/<path:node_id>')
def get_children(node_id):
    if G.has_node(node_id):
        sorted_children = sorted(list(G.successors(node_id)), key=lambda id: get_node_display_text(G, id))
        children_data = [create_node_dict(G, child_id) for child_id in sorted_children]
        return jsonify(children_data)
    return jsonify([])

@app.route('/api/path_to_node/<path:node_id>')
def get_path_to_node(node_id):
    if not G.has_node(node_id):
        return jsonify({'error': 'Node not found'}), 404
    try:
        path = nx.shortest_path(G, source='outter', target=node_id)
        return jsonify(path)
    except nx.NetworkXNoPath:
        return jsonify({'error': f'No path found from outter to {node_id}'}), 404

# =====================================================================
# === SEARCH ENDPOINT (REWRITTEN TO USE HTTP) =========================
# =====================================================================
@app.route('/api/search', methods=['POST'])
def search_nodes():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is empty'}), 400

    try:
        # Step 1: Get embedding for the query using the background worker (which now uses HTTP)
        future = Future()
        embedding_queue.put((future, [query]))
        query_embedding_vector = future.result(timeout=30)[0]

        # Step 2: Use Vector Search by calling its REST API directly
        token = get_gcloud_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        
        # Construct the payload for the findNeighbors REST endpoint
        payload = {
            "deployed_index_id": DEPLOYED_INDEX_ID,
            "queries": [
                {
                    "neighbor_count": 20,
                    "datapoint": {
                        "feature_vector": query_embedding_vector.tolist()
                    }
                }
            ]
        }
        
        url = f"https://{API_ENDPOINT}/v1/{INDEX_ENDPOINT_PATH}:findNeighbors"
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Step 3: Process the response and format for the frontend
        results = []
        json_response = response.json()
        if json_response and 'nearestNeighbors' in json_response and json_response['nearestNeighbors']:
            for neighbor in json_response['nearestNeighbors'][0]['neighbors']:
                node_id = neighbor['datapoint']['datapointId']
                if G.has_node(node_id):
                    results.append({
                        'id': node_id,
                        'text': get_node_display_text(G, node_id),
                        'score': round(float(neighbor['distance']), 4)
                    })
                else:
                    print(f"⚠️ Vector Search returned ID '{node_id}' which is not in the loaded graph. Skipping.")
        
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Error during HTTP Vector Search call or processing: {e}")
        return jsonify({'error': f'Vector Search failed: {e}'}), 500
# =====================================================================
# === END OF CORRECTION ===============================================
# =====================================================================

if __name__ == '__main__':
    # When running locally with `python app.py`, debug should be True
    # For production on Render, Gunicorn sets this, and debug should be False
    is_debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=is_debug, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))