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

# --- NEW: Import for Vertex AI Vector Search ---
# Note: We only import the top-level package here. Specific classes
# will be imported inside the lazy-loading function to reduce startup overhead.
from google.cloud import aiplatform_v1
import vertexai
from vertexai.language_models import TextEmbeddingInput # Keep this one for the worker

# --- Environment and Vertex AI Initialization ---
# This section remains unchanged. It correctly sets up credentials.
CREDENTIALS_FILENAME = "commanding-fact-441820-j9-0e1712201ab6.json"
RENDER_CREDENTIALS_PATH = f"/etc/secrets/{CREDENTIALS_FILENAME}"

if os.path.exists(RENDER_CREDENTIALS_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = RENDER_CREDENTIALS_PATH
    print(f"✅ Credentials found on Render at: {RENDER_CREDENTIALS_PATH}")
else:
    if os.path.exists(CREDENTIALS_FILENAME):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILENAME
        print(f"✅ Credentials found locally for development: {CREDENTIALS_FILENAME}")
    else:
        print(f"❌ FATAL ERROR: Credentials file not found at {RENDER_CREDENTIALS_PATH} or locally.")
        exit()

# --- Configuration Constants ---
# Moved all client configuration here, but NOT the client objects themselves.
PROJECT_ID = "commanding-fact-441820-j9"
MODEL_ID = "text-multilingual-embedding-002"
API_ENDPOINT = "1028163771.us-central1-856708660097.vdb.vertexai.goog"
INDEX_ENDPOINT = "projects/856708660097/locations/us-central1/indexEndpoints/6254692840882831360"
DEPLOYED_INDEX_ID = "my_narrs_emb_index_display_1751478778959"


# =====================================================================
# === LAZY INITIALIZATION FOR GOOGLE CLOUD CLIENTS ====================
# =====================================================================
# These global variables will hold the client objects once they are created.
# They start as None.
_model_vertex = None
_vector_search_client = None
# A lock to prevent race conditions if two requests try to initialize at the same time.
_init_lock = threading.Lock()

def get_vertex_clients():
    """
    Initializes and returns the Vertex AI clients using a thread-safe,
    lazy-loading pattern. This prevents high memory usage on app startup.
    """
    global _model_vertex, _vector_search_client

    # "Double-Checked Locking" pattern. First check is fast and avoids locking.
    if _model_vertex and _vector_search_client:
        return _model_vertex, _vector_search_client

    # If clients are not initialized, acquire a lock to ensure only one
    # thread initializes them.
    with _init_lock:
        # Check again inside the lock in case another thread finished initialization
        # while this thread was waiting.
        if _model_vertex and _vector_search_client:
            return _model_vertex, _vector_search_client

        print("🚀 First-time initialization of Vertex AI clients (this happens only once)...")
        try:
            # Import the specific model class here
            from vertexai.language_models import TextEmbeddingModel

            # Initialize Embedding Model
            vertexai.init(project=PROJECT_ID)
            _model_vertex = TextEmbeddingModel.from_pretrained(MODEL_ID)
            print("✅ Vertex AI Embedding Model initialized successfully.")

            # Initialize Vector Search Client
            client_options = {"api_endpoint": API_ENDPOINT}
            _vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)
            print("✅ Vertex AI Vector Search client initialized successfully.")

            return _model_vertex, _vector_search_client

        except Exception as e:
            print(f"❌ FATAL ERROR during lazy initialization of Vertex AI clients: {e}")
            # Re-raise the exception so the request that triggered this fails cleanly.
            raise
# =====================================================================
# === END OF LAZY INITIALIZATION SETUP ================================
# =====================================================================


# --- Thread-Safe Async Embedding Setup ---
embedding_queue = queue.Queue()

def embedding_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Get the model client for this thread. This will trigger the first-time
    # initialization if the app has just started.
    print("Embedding worker thread is acquiring Vertex AI client...")
    model_vertex, _ = get_vertex_clients()
    print("Embedding worker thread has client.")

    async def aembed_text(texts: list, task: str = "RETRIEVAL_QUERY"):
        try:
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            embeddings = await model_vertex.get_embeddings_async(inputs)
            return np.array([embedding.values for embedding in embeddings])
        except Exception as e:
            print(f"Error during async embedding call: {e}")
            raise

    async def main_loop():
        while True:
            future, texts = embedding_queue.get()
            print(f"Worker received job to embed {len(texts)} text(s).")
            try:
                result = await aembed_text(texts)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    loop.run_until_complete(main_loop())

worker_thread = threading.Thread(target=embedding_worker, daemon=True)
worker_thread.start()
print("✅ Embedding worker thread started.")

# --- App Initialization and Data Loading ---
app = Flask(__name__)

# IMPORTANT: Ensure you are using a "light" version of your pickle file
# that does NOT contain the embeddings. This is the other key to low memory usage.
GRAPH_FILE = "saved_components_tree_processed_light.pkl" # Assumes you created this file

try:
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    print(f"✅ Successfully loaded lightweight graph data from '{GRAPH_FILE}'.")
except FileNotFoundError:
    print(f"❌ ERROR: '{GRAPH_FILE}' not found. Creating dummy graph.")
    G = nx.DiGraph()
    G.add_node("outter")
    G.add_node("branch_A", text="Sample Branch A", uploaded_at_avg="2023-01-01")
    G.add_node("leaf_1", text="Sample Leaf 1", uploaded_at_avg="2023-01-01")
    G.add_edge("outter", "branch_A")
    G.add_edge("branch_A", "leaf_1")

print(f"✅ Graph loaded with {len(G.nodes())} nodes.")

def get_node_display_text(graph, node_id):
    if not graph.has_node(node_id):
        return f"Unknown Node: {node_id}"

    node_data = graph.nodes[node_id]
    if graph.out_degree(node_id) > 0 and 'text' in node_data:
        base_text = node_data['text']
    else:
        base_text = node_id
        
    if 'uploaded_at_avg' in node_data and node_data['uploaded_at_avg']:
        date_str = str(node_data['uploaded_at_avg']).split(' ')[0]
        return f"{base_text} ({date_str})"
    return base_text

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

def create_node_dict(graph, node_id):
    is_branch = graph.out_degree(node_id) > 0
    node_data = graph.nodes[node_id]
    # 'embeddings' key should not exist in the light graph, but this is safe
    payload_data = {k: str(v) if v is not None else '' for k, v in node_data.items() if k != 'embeddings'}

    return {
        'id': node_id,
        'text': get_node_display_text(graph, node_id),
        'has_children': is_branch,
        'data': payload_data
    }

@app.route('/api/nodes/')
def get_root_nodes():
    children_data = []
    if G.has_node('outter'):
        for child_id in G.successors('outter'):
            children_data.append(create_node_dict(G, child_id))
    return jsonify(children_data)

@app.route('/api/nodes/<path:node_id>')
def get_children(node_id):
    children_data = []
    if G.has_node(node_id):
        sorted_children = sorted(list(G.successors(node_id)), key=lambda id: get_node_display_text(G, id))
        for child_id in sorted_children:
            children_data.append(create_node_dict(G, child_id))
    return jsonify(children_data)

@app.route('/api/path_to_node/<path:node_id>')
def get_path_to_node(node_id):
    if not G.has_node(node_id):
        return jsonify({'error': 'Node not found'}), 404
    try:
        path = nx.shortest_path(G, source='outter', target=node_id)
        return jsonify(path)
    except nx.NetworkXNoPath:
        return jsonify({'error': f'No path found from outter to {node_id}'}), 404

@app.route('/api/search', methods=['POST'])
def search_nodes():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is empty'}), 400

    try:
        # Get the clients. This will be instant after the first call.
        _, vector_search_client = get_vertex_clients()

        # Step 1: Get embedding for the query using the background worker
        future = Future()
        embedding_queue.put((future, [query]))
        query_embedding_vector = future.result(timeout=30)[0]

        # Step 2: Use Vertex AI Vector Search
        datapoint = aiplatform_v1.IndexDatapoint(
            feature_vector=query_embedding_vector.tolist()
        )
        query_obj = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=20
        )
        search_request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=INDEX_ENDPOINT,
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_obj],
            return_full_datapoint=True,
        )

        response = vector_search_client.find_neighbors(search_request)

        # Step 3: Process the response
        results = []
        if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                node_id = neighbor.datapoint.datapoint_id
                if G.has_node(node_id):
                    results.append({
                        'id': node_id,
                        'text': get_node_display_text(G, node_id),
                        'score': round(float(neighbor.distance), 4)
                    })
                else:
                    print(f"⚠️ Vector Search returned ID '{node_id}' which is not in the loaded graph. Skipping.")
        
        return jsonify(results)
    
    except Exception as e:
        # This single block now catches errors from client initialization,
        # embedding, or the vector search call.
        print(f"❌ Error during search process: {e}")
        return jsonify({'error': f'An internal error occurred during the search: {e}'}), 500

if __name__ == '__main__':
    # For local development, Render will use its own Gunicorn command
    app.run(debug=True, host='0.0.0.0', port=5000)