from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(texts):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X


def vectorize_query(vec, query):
    return vec.transform([query])