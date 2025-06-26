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

# --- Environment and Vertex AI Initialization ---
CREDENTIALS_FILE = ""
if not os.path.exists(CREDENTIALS_FILE):
    print(f"FATAL ERROR: Credentials file '{CREDENTIALS_FILE}' not found.")
    exit()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
print(f"Using credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

PROJECT_ID = "commanding-fact-441820-j9"
MODEL_ID = "text-multilingual-embedding-002"

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    vertexai.init(project=PROJECT_ID)
    model_vertex = TextEmbeddingModel.from_pretrained(MODEL_ID)
    print("✅ Vertex AI initialized successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not initialize Vertex AI. Error: {e}")
    exit()

# --- Thread-Safe Async Embedding Setup ---
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
    print("❌ ERROR: 'saved_components_tree_processed_full.pkl' not found. Creating dummy graph.")
    G = nx.DiGraph()
    G.add_node("outter")
    G.add_node("branch_A", text="Sample Branch A", uploaded_at_avg="2023-01-01", embeddings=np.random.rand(768))
    G.add_node("leaf_1", text="Sample Leaf 1", uploaded_at_avg="2023-01-01", embeddings=np.random.rand(768))
    G.add_edge("outter", "branch_A")
    G.add_edge("branch_A", "leaf_1")

all_node_ids = list(G.nodes())
valid_nodes_for_search = [
    nid for nid in all_node_ids
    if 'embeddings' in G.nodes[nid] and isinstance(G.nodes[nid]['embeddings'], np.ndarray) and G.nodes[nid]['embeddings'].size > 0
]

if valid_nodes_for_search:
    all_embeddings = np.array([G.nodes[nid]['embeddings'] for nid in valid_nodes_for_search])
    all_embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    print(f"✅ Pre-processed and normalized {len(valid_nodes_for_search)} node embeddings.")
else:
    all_embeddings_norm = np.array([])
    print("⚠️ No nodes with valid embeddings found.")


def get_node_display_text(graph, node_id):
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
    """
    Helper function to create a node dictionary for our custom tree.
    """
    is_branch = graph.out_degree(node_id) > 0
    node_data = graph.nodes[node_id]
    payload_data = {k: str(v) for k, v in node_data.items() if k != 'embeddings' and v is not None}

    return {
        'id': node_id,
        'text': get_node_display_text(graph, node_id),
        'has_children': is_branch,
        'data': payload_data
    }

@app.route('/api/nodes/')
def get_root_nodes():
    """API for custom tree root nodes."""
    children_data = []
    if G.has_node('outter'):
        for child_id in G.successors('outter'):
            children_data.append(create_node_dict(G, child_id))
    return jsonify(children_data)

@app.route('/api/nodes/<path:node_id>')
def get_children(node_id):
    """API for custom tree child nodes."""
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
    if not query: return jsonify({'error': 'Query is empty'}), 400
    if all_embeddings_norm.size == 0: return jsonify({'error': 'No searchable embeddings loaded.'}), 500

    try:
        future = Future()
        embedding_queue.put((future, [query]))
        query_embedding = future.result(timeout=30)
    except Exception as e:
        print(f"❌ Error getting embedding from worker thread: {e}")
        return jsonify({'error': f'Failed to embed query: {e}'}), 500

    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    similarities = np.dot(all_embeddings_norm, query_embedding_norm.T).flatten()
    top_indices = np.argsort(similarities)[-20:][::-1]
    results = [{'id': valid_nodes_for_search[idx], 'text': get_node_display_text(G, valid_nodes_for_search[idx]), 'score': round(float(similarities[idx]), 4)} for idx in top_indices]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)