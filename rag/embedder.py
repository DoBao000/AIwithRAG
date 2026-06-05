from sentence_transformers import SentenceTransformer
from chunks import chunk_text

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_document(document_text: str, embedding_model: SentenceTransformer):
    chunks = chunk_text(document_text, chunk_size=500, overlap=50)
    embeddings = embedding_model.encode(chunks)
    payload = []
    for chunk, embedding in zip(chunks, embeddings):
        payload.append({
            'text': chunk,
            'embedding': embedding
        })
    return payload