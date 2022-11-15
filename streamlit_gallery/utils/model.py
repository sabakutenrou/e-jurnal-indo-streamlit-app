import pandas as pd
from streamlit_gallery.utils.preprocess import preprocess
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from streamlit_gallery.utils.relabel import relabel


df = pd.read_csv("dataset.csv")
df = df[['judul-jurnal','abstrak-jurnal','kategori']]
df['kategori'] = df['kategori'].factorize()[0]
df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(preprocess)

X = df['abstrak-jurnal']
y = df['kategori']

def load_data():
    return df

def predict(text):
    df = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y , random_state= 0)

    pipeline    = Pipeline([("vect",CountVectorizer()), ("tfidf",TfidfTransformer())])
    X_train_vect = pipeline.fit_transform(X_train)
    clf = LinearSVC().fit(X_train_vect, y_train)
    X_test_vect = pipeline.transform([text])
    result = clf.predict(X_test_vect)
    decision_func = clf.decision_function(X_test_vect)
    label = relabel(result[0])

    return {"text": text, "label":label, "decision":decision_func}

