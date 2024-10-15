import nltk
import openai
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


def chunk_text(text, max_chunk_size=64, overlap=32):
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


def parity_chunking(text1, text2, max_chunk_size=64, overlap=32):
    """
    Chunk two input texts into an equal number of overlapping segments, based on the text with the smaller chunk count.

    Args:
    text1 (str): The first input text to be chunked.
    text2 (str): The second input text to be chunked.
    max_chunk_size (int): The maximum number of tokens per chunk. Default is 64.
    overlap (int): The number of overlapping tokens between chunks. Default is 32.

    Returns:
    tuple: Two lists of strings, where each string represents a chunk of text for text1 and text2 respectively.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_and_map(text):
        encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoding['input_ids'], encoding['offset_mapping']

    def chunk_text(tokens, offset_mapping, text, chunk_count):
        chunk_size = len(tokens) // chunk_count
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_offsets = offset_mapping[i:i + chunk_size]
            start = chunk_offsets[0][0]
            end = chunk_offsets[-1][1]
            chunk_text = text[start:end]
            chunks.append(chunk_text)
        return chunks

    # Tokenize both texts
    tokens1, offset_mapping1 = tokenize_and_map(text1)
    tokens2, offset_mapping2 = tokenize_and_map(text2)

    # Calculate the number of chunks for each text
    step = max_chunk_size - overlap
    chunk_count1 = max(1, (len(tokens1) - overlap) // step)
    chunk_count2 = max(1, (len(tokens2) - overlap) // step)

    # Use the smaller chunk count for both texts
    min_chunk_count = min(chunk_count1, chunk_count2)

    # Chunk both texts using the minimum chunk count
    chunks1 = chunk_text(tokens1, offset_mapping1, text1, min_chunk_count)
    chunks2 = chunk_text(tokens2, offset_mapping2, text2, min_chunk_count)

    return chunks1, chunks2

def chunk_sentences_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that chunks text into sentences. Return the result as a JSON array of sentences."},
                {"role": "user",
                 "content": f"Please chunk the following text into sentences and return the result as a JSON array: {text}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Extract the content from the API response
        result = response.choices[0].message['content']

        # Parse the JSON string into a Python object
        sentences = json.loads(result)

        return sentences
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []