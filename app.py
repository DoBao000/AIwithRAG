import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from rag.chunks import chunk_text
from rag.embedder import model, embed_document
from rag.generator import generate

# --- CONFIG ---
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
VECTOR_DB = []
# --------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --- APP ---
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Receive a file, chunk it, embed it, and store the FAISS index."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .txt and .pdf files are supported'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            content_chunks = chunk_text(content)

        chunk_embeddings = model.encode(content_chunks)
        for chunk, embedding in zip(content_chunks, chunk_embeddings):
            VECTOR_DB.append({
                'text': chunk,
                'embedding': embedding
            })

        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {filename}',
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided in request body'}), 400

    user_question = data['question']
    if user_question.strip() == '':
        return jsonify({'error': 'Question cannot be empty'}), 400

    # 2. Check if the database actually has documents uploaded yet
    if not VECTOR_DB:
        return jsonify({'error': 'No documents have been uploaded yet. Please upload a file first.'}), 400

    try:
        ai_response = generate(user_question, VECTOR_DB)

        return jsonify({
            "status": "success",
            "question": user_question,
            "answer": ai_response
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# -----------