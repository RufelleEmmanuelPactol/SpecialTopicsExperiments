from resume_scorer.keyword_processing.tokenizers import tokenizer


def chunk_text(text, max_chunk_size=5, overlap=2):
    """
    Chunk the input text into overlapping segments of max_chunk_size tokens.

    Args:
    text (str): The input text to be chunked.
    max_chunk_size (int): The maximum number of tokens per chunk. Default is 512.
    overlap (int): The number of overlapping tokens between chunks. Default is 255.

    Returns:
    list: A list of strings, where each string represents a chunk of text.
    """
    # Tokenize the entire text
    encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']

    # Calculate the step size
    step = max_chunk_size - overlap

    # Split the tokens into overlapping chunks
    chunks = []
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_chunk_size]
        chunk_offsets = offset_mapping[i:i + max_chunk_size]

        # Get the text for this chunk
        start = chunk_offsets[0][0]
        end = chunk_offsets[-1][1]
        chunk_text = text[start:end]

        chunks.append(chunk_text)

        # If this is the last chunk (potentially shorter than max_chunk_size), break the loop
        if i + max_chunk_size >= len(tokens):
            break

    return chunks

chunks = chunk_text("""

A pure function is a concept in functional programming which utilizes two key concepts: first, is the concept of "no side-effects". No side effects means that the function does not modify any external state, or any state outside of its scope. The second concept is when a function, given the same input, will always produce the same output. This makes it such that pure functions are a reliable, maintainable way to code, making it easier to spot and identify bugs.
""")

print(chunks)