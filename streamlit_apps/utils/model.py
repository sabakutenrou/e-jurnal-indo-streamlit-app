import pandas as pd
from streamlit_apps.utils.preprocess import preprocess
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from streamlit_apps.utils.relabel import relabel
# import chardet

# with open('dataset.csv','rb') as f:
#     result=chardet.detect(f.read())

# print(result['encoding'])

__df = pd.read_csv("dataset.csv", encoding='utf-8')
__df_clean = pd.read_csv("dataset-clean.csv", encoding='utf-8')

def __format_data(df):
    df = df[['judul-jurnal','abstrak-jurnal','kategori']]
    # df['kategori'] = df['kategori'].factorize()[0]
    df['abstrak-jurnal'] = df['abstrak-jurnal']
    df['judul-abstrak'] = df['judul-jurnal'] + " " + df["abstrak-jurnal"]
    return df[['judul-abstrak','kategori']]

def load_data(clean_data=False):
    df = __df if not clean_data else __df_clean
    return __format_data(df)
    #  if format else df[['judul-abstrak','kategori']]
    
def predict(text):
    """
        this predict uses the cleaned dataset
    """
    df = load_data(clean_data=True)
    X = df['judul-abstrak']
    y = df['kategori']
    
    df['kategori'] = df['kategori'].factorize()[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y , random_state= 0)

    pipeline    = Pipeline([("vect",CountVectorizer()), ("tfidf",TfidfTransformer())])
    X_train_vect = pipeline.fit_transform(X_train)
    clf = LinearSVC().fit(X_train_vect, y_train)
    X_test_vect = pipeline.transform([text])
    result = clf.predict(X_test_vect)
    decision_func = clf.decision_function(X_test_vect)
    label = result[0]

    return {"text": text, "label":label, "decision":decision_func}

def con():
    from sklearn.calibration import CalibratedClassifierCV
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm) 
    # clf.fit(X_test_vect, y_train)
    # y_proba = clf.predict_proba(X_test)









    return [261852, 341913, 478837, 181540, 299514, 461686, 200163, 366967, 253917, 372347, 4576188]