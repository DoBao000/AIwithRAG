import os
import torch
from google import genai
from dotenv import load_dotenv
load_dotenv()
from rag.embedder import model

client = genai.Client(api_key=os.getenv('API_KEY'))

def retrieve(query, processed_chunks, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = torch.tensor([item['embedding'] for item in processed_chunks])
    scores = model.similarity(query_embedding, doc_embeddings)[0]

    actual_k = min(top_k, len(processed_chunks))
    top_indices = torch.topk(scores, k=actual_k).indices.tolist()

    return [processed_chunks[i]['text'] for i in top_indices]

def generate(query, processed_chunks):
    # 1. Fetch relevant context from your embedder
    relevant_chunks = retrieve(query, processed_chunks)
    context = "\n\n".join(relevant_chunks)

    # 2. Construct the prompt
    prompt = f'''Answer the question based on the context below.
                Context: {context}
                Question: {query}
            '''

    # 3. Call the Gemini API
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text