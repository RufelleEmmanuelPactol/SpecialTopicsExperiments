from pdfminer.high_level import extract_text
import re


def pdf_to_text(pdf_path):
    # Extract text using pdfminer
    text = extract_text(pdf_path)

    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines with space
    text = re.sub(r'\n{2,}', '\n\n', text)  # Replace multiple newlines with double newline

    # Additional cleaning specific to the observed issues
    text = text.replace('●', '\n●')  # Ensure bullet points start on new lines
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words incorrectly joined

    return text.strip()


# Example usage
