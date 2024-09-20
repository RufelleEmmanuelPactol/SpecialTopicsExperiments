import numpy as np
import gensim.downloader as api
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams


word_vectors = api.load("glove-wiki-gigaword-100")
print("Word vectors loaded.")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]


def get_ngrams(text, n=1):
    tokens = preprocess(text)
    return [' '.join(gram) for gram in ngrams(tokens, n)]


def get_phrase_embedding(phrase, model):
    tokens = preprocess(phrase)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else None


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def chunk_text(text, chunk_size=3):
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks


def keyword_relevance(keywords, text, model=word_vectors, chunk_size=3):
    chunks = chunk_text(text, chunk_size)
    relevance_scores = {keyword: {'exact_count': 0, 'max_similarity': 0} for keyword in keywords}

    for chunk in chunks:
        chunk_lower = chunk.lower()
        chunk_embedding = get_phrase_embedding(chunk, model)

        for keyword in keywords:
            # Exact match count (case-insensitive)
            exact_count = chunk_lower.count(keyword.lower())
            relevance_scores[keyword]['exact_count'] += exact_count

            # Semantic similarity
            keyword_embedding = get_phrase_embedding(keyword, model)
            if keyword_embedding is not None and chunk_embedding is not None:
                similarity = cosine_similarity(keyword_embedding, chunk_embedding)
                relevance_scores[keyword]['max_similarity'] = max(relevance_scores[keyword]['max_similarity'], similarity)

    # Calculate final scores
    final_scores = {}
    for keyword, scores in relevance_scores.items():
        exact_score = min(scores['exact_count'], 1)  # Cap exact matches at 1
        similarity_score = scores['max_similarity']
        final_scores[keyword] = 0.7 * exact_score + 0.3 * similarity_score

    return final_scores


# Example usage
keywords = ["sql", "python", "java", "scala", "data science", "Amazon Web Services", "sklearn experience", "data"]
text = """
Education Details \r\nMay 2013 to May 2017 B.E   UIT-RGPV\r\nData Scientist \r\n\r\nData Scientist - Matelabs\r\nSkill Details \r\nPython- Experience - Less than 1 year months\r\nStatsmodels- Experience - 12 months\r\nAWS- Experience - Less than 1 year months\r\nMachine learning- Experience - Less than 1 year months\r\nSklearn- Experience - Less than 1 year months\r\nScipy- Experience - Less than 1 year months\r\nKeras- Experience - Less than 1 year monthsCompany Details \r\ncompany - Matelabs\r\ndescription - ML Platform for business professionals, dummies and enthusiasts.\r\n60/A Koramangala 5th block,\r\nAchievements/Tasks behind sukh sagar, Bengaluru,\r\nIndia                               Developed and deployed auto preprocessing steps of machine learning mainly missing value\r\ntreatment, outlier detection, encoding, scaling, feature selection and dimensionality reduction.\r\nDeployed automated classification and regression model.\r\nlinkedin.com/in/aditya-rathore-\r\nb4600b146                           Research and deployed the time series forecasting model ARIMA, SARIMAX, Holt-winter and\r\nProphet.\r\nWorked on meta-feature extracting problem.\r\ngithub.com/rathorology\r\nImplemented a state of the art research paper on outlier detection for mixed attributes.\r\ncompany - Matelabs\r\ndescription -
"""

relevance_scores = keyword_relevance(keywords, text, word_vectors)

