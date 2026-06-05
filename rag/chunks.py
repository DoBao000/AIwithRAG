def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    if chunk_size <= 0 or overlap < 0:
        raise ValueError('chunk_size must be > 0 and overlap >= 0')

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

document = '''
    Many people say that life isn't like a bed of roses. I beg to differ. 
    I think that life is quite like a bed of roses. 
    Just like life, a bed of roses looks pretty on the outside, but when you're in it, you find that it is nothing but thorns and pain. 
    I myself have been pricked quite badly.
'''

chunks = chunk_text(document, chunk_size=30, overlap=10)