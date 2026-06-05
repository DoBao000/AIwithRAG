from sentence_transformers import SentenceTransformer
from chunks import chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)