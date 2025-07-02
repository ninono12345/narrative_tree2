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
from google.cloud import aiplatform_v1

# --- Environment and Vertex AI Initialization ---
# CREDENTIALS_FILE = "commanding-fact-441820-j9-0e1712201ab6.json"
# if not os.path.exists(CREDENTIALS_FILE):
#     print(f"FATAL ERROR: Credentials file '{CREDENTIALS_FILE}' not found.")
#     # exit()

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
# print(f"Using credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

CREDENTIALS_FILENAME = "commanding-fact-441820-j9-0e1712201ab6.json"

# Render places secret files in /etc/secrets/
# Construct the full path to the credentials file
RENDER_CREDENTIALS_PATH = f"/etc/secrets/{CREDENTIALS_FILENAME}"

# Check if the file exists at the expected path
if os.path.exists(RENDER_CREDENTIALS_PATH):
    # Set the environment variable that the Google Cloud library expects
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = RENDER_CREDENTIALS_PATH
    print(f"✅ Credentials found on Render at: {RENDER_CREDENTIALS_PATH}")
else:
    # This is a fallback for local development (optional but good practice)
    # if os.path.exists(CREDENTIALS_FILENAME):
    #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILENAME
    #     print(f"✅ Credentials found locally: {CREDENTIALS_FILENAME}")
    # else:
    print(f"❌ FATAL ERROR: Credentials file not found at {RENDER_CREDENTIALS_PATH} or locally.")
    exit()


PROJECT_ID = "commanding-fact-441820-j9"
MODEL_ID = "text-multilingual-embedding-002"

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    vertexai.init(project=PROJECT_ID)
    model_vertex = TextEmbeddingModel.from_pretrained(MODEL_ID)
    print("✅ Vertex AI Embedding Model initialized successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not initialize Vertex AI Embedding Model. Error: {e}")
    # exit()

# --- NEW: Vertex AI Vector Search Configuration ---
API_ENDPOINT = "1028163771.us-central1-856708660097.vdb.vertexai.goog"
INDEX_ENDPOINT = "projects/856708660097/locations/us-central1/indexEndpoints/6254692840882831360"
DEPLOYED_INDEX_ID = "my_narrs_emb_index_display_1751478778959"

try:
    client_options = {"api_endpoint": API_ENDPOINT}
    vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)
    print("✅ Vertex AI Vector Search client initialized successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not initialize Vertex AI Vector Search client. Error: {e}")
    # exit()


# --- Thread-Safe Async Embedding Setup (Unchanged) ---
embedding_queue = queue.Queue()

def embedding_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def aembed_text(texts: list, task: str = "CLUSTERING"):
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

try:
    with open("saved_components_tree_processed_full.pkl", "rb") as f:
        G = pickle.load(f)
    print("✅ Successfully loaded the graph data.")
except FileNotFoundError:
    print("❌ ERROR: 'saved_components_tree_processed_full_emb.pkl' not found. Creating dummy graph.")
    G = nx.DiGraph()
    G.add_node("outter")
    G.add_node("branch_A", text="Sample Branch A", uploaded_at_avg="2023-01-01")
    G.add_node("leaf_1", text="Sample Leaf 1", uploaded_at_avg="2023-01-01")
    G.add_edge("outter", "branch_A")
    G.add_edge("branch_A", "leaf_1")

print(f"✅ Graph loaded with {len(G.nodes())} nodes.")

def get_node_display_text(graph, node_id):
    if not graph.has_node(node_id):
        print(f"Warning: Attempted to get display text for non-existent node '{node_id}'")
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

# =====================================================================
# === CORRECTED SEARCH ENDPOINT =======================================
# =====================================================================
@app.route('/api/search', methods=['POST'])
def search_nodes():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is empty'}), 400

    # Step 1: Get embedding for the query using the background worker
    try:
        future = Future()
        embedding_queue.put((future, [query]))
        query_embedding_vector = future.result(timeout=30)[0]
    except Exception as e:
        print(f"❌ Error getting embedding from worker thread: {e}")
        return jsonify({'error': f'Failed to embed query: {e}'}), 500

    # Step 2: Use Vertex AI Vector Search to find nearest neighbors
    try:
        datapoint = aiplatform_v1.IndexDatapoint(
            feature_vector=query_embedding_vector.tolist()
        )
        query_obj = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=20 # Fetch top 20 results
        )
        search_request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=INDEX_ENDPOINT,
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_obj],
            return_full_datapoint=True, # Set to True to get the datapoint object
        )

        response = vector_search_client.find_neighbors(search_request)

        # Step 3: Process the response and format for the frontend
        results = []
        if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                # === THE FIX IS HERE ===
                # Access the ID via neighbor.datapoint.datapoint_id
                node_id = neighbor.datapoint.datapoint_id

                if G.has_node(node_id):
                    results.append({
                        'id': node_id,
                        'text': get_node_display_text(G, node_id),
                        # The score from the API is a distance metric (lower is better).
                        'score': round(float(neighbor.distance), 4)
                    })
                else:
                    print(f"⚠️ Vector Search returned ID '{node_id}' which is not in the loaded graph. Skipping.")
        
        # NOTE: The frontend JS will display distance as 'score'. Lower is more similar.
        # To make it more intuitive (higher = better), you could sort by distance ascending
        # or convert distance to similarity (e.g., 1 - distance, if distance is normalized).
        # For now, we'll keep it as is.
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Error during Vertex AI Vector Search call: {e}")
        return jsonify({'error': f'Vector Search failed: {e}'}), 500
# =====================================================================
# === END OF CORRECTION ===============================================
# =====================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)