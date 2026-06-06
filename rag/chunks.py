def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    if chunk_size <= 0 or overlap < 0:
        raise ValueError('chunk_size must be > 0 and overlap >= 0')
    if overlap >= chunk_size:
        raise ValueError('overlap cannot be greater than chunk_size')

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks