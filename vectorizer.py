from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(text_data):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(text_data)
    return X, tfidf