import os
import torch
from google import genai
from dotenv import load_dotenv
load_dotenv()
from embedder import model, embeddings, chunks

client = genai.Client(api_key=os.getenv('API_KEY'))

def retrieve(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = model.similarity(query_embedding, embeddings)[0]
    top_indices = torch.topk(scores, k=top_k).indices
    return [chunks[i] for i in top_indices]

def generate(query):
    # 1. Fetch relevant context from your embedder
    relevant_chunks = retrieve(query)
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

# --- Execution ---
if __name__ == '__main__':
    # Replace this with a question relevant to your 'chunks' data
    user_question = 'What is bed of roses?'

    print('Thinking...')
    answer = generate(user_question)

    print('\n--- Answer ---')
    print(answer)