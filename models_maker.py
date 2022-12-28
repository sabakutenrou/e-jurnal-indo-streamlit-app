import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import matplotlib as plt
import joblib
from streamlit_apps.utils.model import load_data

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #

from joblib import dump, load

# df = load_data()
df = pd.read_csv("dataset.csv")

st.write(df)

from streamlit_apps.utils.preprocess import preprocess, remove_stopwordss
from streamlit_apps.utils.lang_detection import mod_lang_detect_match
df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(preprocess)
df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(lambda text: remove_stopwordss(text))
df['lang'] = df['abstrak-jurnal'].apply(mod_lang_detect_match)

df.to_csv('dataset-lang.csv', index=False)

X = df['abstrak-jurnal']
y = df['kategori']


if st.button('download csv'):
    st.write()

test_size = 0.25
train_size = 1 - test_size
train_set, test_set = train_test_split(df, test_size=test_size, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=test_size, random_state= 0)

prog_col1, prog_col2, prog_col3 = st.columns([1,8,1])
prog_col1.write(str(train_size*100) + '%')
prog_col2.progress(train_size)
prog_col3.write(str(test_size*100) + '%')




# analyzer_vectorizer = TfidfVectorizer(analyzer=stemming)
# analyzer_vectorizer = load('analyzer_vectorizer.bin')
# X_train_vect = analyzer_vectorizer.fit_transform(X_train)
# X_train_vect = analyzer_vectorizer.transform(X_train)
# X_test_vect = analyzer_vectorizer.transform(X_test)
# feature_names = analyzer_vectorizer.get_feature_names_out()

# dump(analyzer_vectorizer, 'analyzer_vectorizer.bin', compress=True)
# dump(X_train_vect, 'X_train_vect.bin', compress=True)
# dump(X_test_vect, 'X_test_vect.bin', compress=True)
# dump(feature_names, 'feature_names.bin', compress=True)

# sc=load('X_train_vect.bin')