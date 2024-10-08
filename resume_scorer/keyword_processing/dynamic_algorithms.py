import math
import re

import nltk
import numpy as np
from functools import lru_cache

import streamlit.logger
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SimilarityScorer:
    """
    This is a multi-modal class that utilizes different similarity engines to calculate
    the similarity scores between keywords and a decently-sized text corpus.

    Currently, the scorer supports the following similarity engines:
    `ngram-product`: uses ngram-based similarity, where an average score is generated using ngrams where n in {1, 2, 3}.
                    this may be computationally intensive.


    """

    def __init__(self, engine='ngram-product', transformer='paraphrase-MiniLM-L6-v2', verbose=True):
        self.engine = engine
        self.verbose = verbose
        self.model = SentenceTransformer(transformer)
        self._write_log("Initializing SimilarityScorer")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _write_log(self, log):
        if self.verbose:
            print(log)

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

    def cosine_sim(self, text1, text2):
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        return cosine_similarity(embedding1, embedding2)[0][0]

    @lru_cache(maxsize=1280)
    def get_embedding(self, text):
        return self.model.encode([text])

    def transform_score(self, x):
        return np.sign(x) * abs(x ** 2)

    def calculate_relevance_scores(self, text, keywords):
        if self.engine == 'ngram-product':
            return self._calculate_relevance_scores_ngram(text, keywords)
        elif self.engine == 'sentence-chunk':
            return self._calculate_relevance_scores_sentence(text, keywords)
        else:
            raise ValueError("Invalid engine specified. Use 'ngram-product' or 'sentence-chunk'.")

    def _calculate_relevance_scores_ngram(self, text, keywords):
        preprocessed_text = self.preprocess_text(text)
        relevance_scores = {}

        for keyword in keywords:
            preprocessed_keyword = self.preprocess_text(keyword)

            all_scores = []

            for n in [2, 3]:
                max_score = 0
                best_ngram = ''
                text_ngrams = self.generate_ngrams(preprocessed_text, n)

                for ngram in text_ngrams:
                    similarity = self.cosine_sim(ngram, preprocessed_keyword)
                    if np.isnan(similarity) or similarity is None:
                        continue
                    all_scores.append(similarity)

                    if similarity > max_score:
                        max_score = similarity
                        best_ngram = ngram

            transformed_scores = [self.transform_score(score) for score in all_scores]
            final_score = np.tanh((sum(transformed_scores) / np.sqrt(len(all_scores))) / 2) if len(
                all_scores) > 0 else 0

            relevance_scores[keyword] = {
                'similarity_score': final_score,
                'best_matching_ngram': best_ngram
            }

        streamlit.logger.get_logger(__name__).info(relevance_scores)

        return relevance_scores

    def _calculate_relevance_scores_sentence(self, text, keywords):
        sentences = sent_tokenize(text)
        relevance_scores = {}

        for keyword in keywords:
            relevance_unit = self.RelevanceUnit(self, keyword)
            for sentence in sentences:
                relevance_unit.transform_calculate_sentence(sentence)

            relevance_scores[relevance_unit.get_keyword()] = {
                'similarity_score': relevance_unit.aggregate_similarity(),
            }

        return relevance_scores

    class RelevanceUnit:

        def __init__(self, parent, keyword):
            self.parent = parent
            self.keyword = keyword
            self.scores = []
            self.sentences = []

        def transform_score(self, x):
            return np.sign(x) * abs(x ** 2)

        def transform_calculate_sentence(self, sentence):
            self.sentences.append(sentence)
            score = self.parent.cosine_sim(sentence, self.keyword)
            score = self.transform_score(score)
            self.scores.append(score)


        def bind_scores(self, x):
            if x > 0.8:
                return 1
            if x < 0.45:
                return 0
            var  = x - 0.45
            ov_args = var / (0.8-0.45)
            return ov_args


        def aggregate_similarity(self):
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            return np.tanh(sum(self.scores) / math.sqrt(len(self.scores)) )

        def get_keyword(self):
            return self.keyword







def internet_available():
    try:
        # Try to connect to a reliable server
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False



def format_resume(resume_text):
    if not internet_available():
        print("Internet is not available. Returning original text.")
        return resume_text

    try:
        client = OpenAI()

        prompt = f"""Please format the following text into a professional resume:

        {resume_text}

        Ensure the resume is well-structured, includes all relevant information, and follows standard resume 
        formatting practices. Please do not add any of your personal comments in the response. Just give me the 
        resume. This is imperative. Do not change the resume. Only format it at a readable state. Do not format in markdown. Format in plain text."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume writer."},
                {"role": "user", "content": prompt}
            ]
        )

        formatted_resume = response.choices[0].message.content
        return formatted_resume
    except Exception as e:
        print(f"An error occurred while querying ChatGPT: {e}")
        return resume_text



