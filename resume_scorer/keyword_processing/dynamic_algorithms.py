import math
import re

import nltk
import numpy as np
from functools import lru_cache
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from resume_scorer.keyword_processing.tokenizers import preprocess_text, chunk_text

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
            max_score = 0
            best_ngram = ''
            for n in [3, 4, 5]:

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

        print(relevance_scores)

        return relevance_scores

    def _calculate_relevance_scores_sentence(self, text, keywords):
        sentences = preprocess_text(text)
        sentences = chunk_text(sentences, max_chunk_size=24, overlap=8)
        print('[ilo]', sentences)



        print('TOKENIZED SENTENCES', sentences)
        print('LENGTH OF IT', len(sentences))
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

        def pairwise_max(self, scores, sentences):
            i = 0
            for x in range(len(scores)):
                if scores[x] > scores[i]:
                    i = x
            return sentences[i]

        def augmented_cosine_loss(self, x, y, k):
            import torch



            cosine_sim =cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

            # Compute regular cosine distance
            cosine_dist = 1 - cosine_sim

            # Create a mask for pairs with different signs
            sign_mask = (np.sign(x) != np.sign(y))
            # Double the distance for pairs with different signs
            modified_dist = np.where(sign_mask, 4 * cosine_dist, cosine_dist)

            # Compute the loss as 1 - modified distance
            loss = 1 - modified_dist

            # Compute the mean loss
            mean_loss = loss.mean()

            return mean_loss



        def augmented_cosine_similarity(self, corpus_x, corpus_y, k):
            x_embed = self.parent.get_embedding(corpus_x)[0]
            y_embed = self.parent.get_embedding(corpus_y)[0]
            return self.relu(1 - self.augmented_cosine_loss(x_embed, y_embed, k))

        def relu(self, value):
            return max(0, value)


        def transform_calculate_sentence(self, sentence):
            self.sentences.append(sentence)
            score = self.augmented_cosine_similarity(sentence, self.keyword, 3)
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
            print(f'MAX RESULT[{self.keyword}]', self.pairwise_max(self.scores, self.sentences))
            return sigmoid(sum(self.scores) / np.sqrt(len(self.scores)) ** 2)

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


class CrossProductSimilarity:
    def __init__(self, transformer='paraphrase-MiniLM-L6-v2', verbose=True, strictness_policy=3):
        self.verbose = verbose
        self.model = SentenceTransformer(transformer)
        self.strictness_policy = strictness_policy

    def _write_log(self, log):
        if self.verbose:
            print(log)

    @lru_cache(maxsize=1280)
    def get_embedding(self, text):
        return self.model.encode([text])

    def augmented_cosine_loss(self, x, y, k):
        cosine_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
        cosine_dist = 1 - cosine_sim
        sign_mask = (np.sign(x) != np.sign(y))
        modified_dist = np.where(sign_mask, k * cosine_dist, cosine_dist)
        loss = 1 - modified_dist
        return loss.mean()

    def augmented_cosine_similarity(self, corpus_x, corpus_y, k):
        x_embed = self.get_embedding(corpus_x)[0]
        y_embed = self.get_embedding(corpus_y)[0]
        return max(0, 1 - self.augmented_cosine_loss(x_embed, y_embed, k))

    def cosine_similarity(self, corpus_x, corpus_y):
        x_embed = self.get_embedding(corpus_x)[0]
        y_embed = self.get_embedding(corpus_y)[0]
        return 1 - cosine_similarity(x_embed.reshape(1, -1), y_embed.reshape(1, -1))

    def calculate_similarity(self, text1, text2):
        """

        :param text1: Correct Answer
        :param text2: Querying Answer
        :return:
        """
        sentences1 = preprocess_text(text1, remove_stopwords=True)
        sentences2 = preprocess_text(text2, remove_stopwords=True)
        chunks1 = chunk_text(sentences1, max_chunk_size=320, overlap=8)
        chunks2 = chunk_text(sentences2, max_chunk_size=320, overlap=8)
        print('len', len(chunks1), chunks1)
        print('len', len(chunks2), chunks2)
        self.len_diff = abs(len(chunks1) - len(chunks2))

        self.is_complete = len(chunks1) > len(chunks2)
        print(self.is_complete, self.len_diff)

        self._write_log(f"Number of chunks in text1: {len(chunks1)}")
        self._write_log(f"Number of chunks in text2: {len(chunks2)}")

        similarity_scores = []

        for chunk2 in chunks2:
            # Initialize max score and best matching chunk2
            max_score = float('-inf')
            best_chunk2 = None

            # First find the most similar chunk2 using regular cosine similarity
            for chunk1 in chunks1:
                score = self.cosine_similarity(chunk2, chunk1)
                if score > max_score:
                    max_score = score
                    best_chunk2 = chunk2

            # Calculate and append the augmented cosine similarity only for the best match
            if best_chunk2 is not None:
                augmented_score = self.augmented_cosine_similarity(chunk1, best_chunk2, 5)
                similarity_scores.append(augmented_score)

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        final_similarity = max(similarity_scores)
        final_similarity = 1 - final_similarity

        # self._write_log(f"Final similarity score: {final_similarity}")
        return self.augmented_loss_normalization(final_similarity)

        DENOMINATOR = 0.5

        # base denominator

        factor = (DENOMINATOR + (0.075 * self.strictness_policy)) * 100
        raw_score = (final_similarity / factor) * 100

        return self.augmented_loss_normalization(raw_score)

    import numpy as np

    def augmented_loss_normalization(self, score):
        # Adjusted maximum based on strictness policy
        base_max = 0.6
        max_increase_per_policy = 0.05
        target_max = base_max + (max_increase_per_policy * self.strictness_policy)

        # Define the breakpoints
        len_diff_factor = self.len_diff / (9 + self.len_diff ** 2)
        strictness_factor = (5 - self.strictness_policy) / 10

        # completeness is when the answer is shorter.
        print('completeness', self.is_complete)
        if self.is_complete:
            lower_bound = (-0.3 - len_diff_factor) - strictness_factor
            middle_point = (0.1 - len_diff_factor) - strictness_factor
        else:
            lower_bound = (-0.3 + len_diff_factor) - strictness_factor
            middle_point = (0.1 + strictness_factor)

        upper_bound = 0.25

        # Clip the score to ensure it's within the expected range
        # score = np.clip(score, lower_bound, upper_bound)

        if score >= middle_point:
            # Map to the target range 0.85 - target_max
            target_high = 0.80

            r_val = target_high + (target_max - target_high) * (score - middle_point) / (
                    upper_bound - middle_point)
            print('score is greater than mid', score, r_val)
            return r_val
        else:
            # Map to the range 0 - 0.85 with a steeper curve
            print('score is less than mid')
            target_min = 0
            target_high = 0.80
            return target_min + target_high * ((score - lower_bound) / (middle_point - lower_bound)) ** 2

    def sigmoid_adjusted_score(score, strictness_policy):
        # Compute the adjustment factor
        adjustment_factor = 0.6 + (0.05 * strictness_policy)

        # Apply sigmoid transformation
        sigmoid_input = score / adjustment_factor
        sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))

        # Clip scores above 1
        sigmoid_output = np.clip(sigmoid_output, 0, 1)

        # Zero out scores below -10
        if score < -10:
            return 0
        return sigmoid_output