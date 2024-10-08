import io

import PyPDF2
import re


def pdf_to_text(pdf_bytes):
    # Create a BytesIO object from the input bytes
    pdf_file = pdf_bytes

    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Initialize an empty string to store the text
    text = ""

    # Iterate through all pages and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines with space
    text = re.sub(r'\n{2,}', '\n\n', text)  # Replace multiple newlines with double newline

    # Additional cleaning specific to the observed issues
    text = text.replace('●', '\n●')  # Ensure bullet points start on new lines
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words incorrectly joined

    return text.strip()
