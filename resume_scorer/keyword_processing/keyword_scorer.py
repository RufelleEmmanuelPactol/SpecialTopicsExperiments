import re

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize


class TextProcessor:
    """
    Processes and parses the text through the removal of stopwords
    """

    def __init__(self, text):
        self.text = text

    def _get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def _preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_tokens = [word for word in word_tokens if word not in stop_words]

        lemmatized_tokens = [lemmatizer.lemmatize(w, self._get_wordnet_pos(w)) for w in filtered_tokens]

        return lemmatized_tokens

    def process_and_get_text(self):
        return self._preprocess_text(self.text)


