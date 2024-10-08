import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import re

nltk.download('stopwords', quiet=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Get English stopwords
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Clean special characters, remove additional spaces, and remove stop words.

    Args:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stop words
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]

    return ' '.join(cleaned_words)


def chunk_text(text, max_chunk_size=512, overlap=255):
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