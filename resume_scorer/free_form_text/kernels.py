import numpy as np
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def extreme_activator(scores):
    """
    Given a list of scores in the range [0, 1], return the score that is farthest from 0.5.

    Args:
    scores (list): A list of float values between 0 and 1.

    Returns:
    float: The score that is farthest from 0.5.
    """
    if not scores:
        return None

    # Calculate the distance of each score from 0.5
    distances = [abs(score - 0.5) for score in scores]

    # Find the index of the maximum distance
    max_distance_index = np.argmax(distances)

    # Return the score corresponding to the maximum distance
    return scores[max_distance_index]


class CrossProductSimilarity:
    def __init__(self, transformer='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(transformer)

    def get_embedding(self, text):
        return self.model.encode([text])[0]

    def cosine_sim(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    def penalize_non_similar(self, score):
        if score < 0.5:
            return score ** 2
        return score

    def evaluate_similarity(self, x, y):
        # Tokenize x and y into sentences
        x_sentences = sent_tokenize(x)
        y_sentences = sent_tokenize(y)

        # Get embeddings for all sentences
        x_embeddings = [self.get_embedding(sent) for sent in x_sentences]
        y_embeddings = [self.get_embedding(sent) for sent in y_sentences]

        # Calculate cross-product similarity with extreme transformation
        similarity_scores = []
        for x_emb in x_embeddings:
            maxes = []
            for y_emb in y_embeddings:
                similarity = self.cosine_sim(x_emb, y_emb)
                similarity = self.penalize_non_similar(similarity)
                maxes.append(similarity)
            similarity_scores.append(max(maxes))
        print('First Engine', similarity_scores)
        # Aggregate similarity using the original strategy
        return self.aggregate_similarity(similarity_scores)

    def aggregate_similarity(self, scores):
        # Use sigmoid function as in the original code
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        return sigmoid(sum(scores) / math.sqrt(len(scores)))


class NgramCrossProductSimilarity:
    def __init__(self, transformer='paraphrase-MiniLM-L6-v2', ngram_range=(2, 5)):
        self.model = SentenceTransformer(transformer)
        self.ngram_range = ngram_range
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        word_tokens = word_tokenize(text)
        filtered_tokens = [word for word in word_tokens if word not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(w, self._get_wordnet_pos(w)) for w in filtered_tokens]
        return ' '.join(lemmatized_tokens)

    def generate_ngrams(self, text, n):
        words = text.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def get_embedding(self, text):
        return self.model.encode([text])[0]

    def cosine_sim(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]

    def transform_score(self, x):
        return np.sign(x) * abs(x ** 2)

    def evaluate_similarity(self, text1, text2):
        preprocessed_text1 = self.preprocess_text(text1)
        preprocessed_text2 = self.preprocess_text(text2)

        ngrams1 = []
        ngrams2 = []

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams1.extend(self.generate_ngrams(preprocessed_text1, n))
            ngrams2.extend(self.generate_ngrams(preprocessed_text2, n))

        embeddings1 = [self.get_embedding(ngram) for ngram in ngrams1]
        embeddings2 = [self.get_embedding(ngram) for ngram in ngrams2]

        similarity_scores = []
        for emb1 in embeddings1:
            max_score = []
            for emb2 in embeddings2:
                similarity = self.cosine_sim(emb1, emb2)
                if np.isnan(similarity) or similarity is None:
                    print("Shouldn't be NaN at all!")
                    continue
                transformed_score = self.penalize_non_similar(similarity)
                max_score.append(transformed_score)
            similarity_scores.append(max(max_score))
        print(similarity_scores)
        return self.aggregate_similarity(similarity_scores)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def aggregate_similarity(self, scores):
        return (sum(scores) / len(scores))

    def penalize_non_similar(self, score):
        if score < 0.5:
            return score ** 2
        return score
