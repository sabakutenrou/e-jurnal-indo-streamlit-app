from os import write
from langdetect.detector_factory import detect_langs
from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import detect_id as di
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

import joblib

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

raw = pd.read_csv("dataset.csv")

raw = raw[['kategori','judul-jurnal','abstrak-jurnal']]

raw['cat_id'] = raw['kategori'].factorize()[0]    # category_id column auto inserted? 

langdf = []
for line in raw['abstrak-jurnal']:
    result = detect(line)
    langdf.append(result)

raw['lang'] = langdf

raw = raw.loc[raw['lang'] == 'id']

df = raw[['judul-jurnal', 'abstrak-jurnal', 'cat_id']]

st.table(df.head(2))


tfidf = TfidfVectorizer(sublinear_tf= True, #use a logarithmic form for frequency
                       min_df = 5, #minimum numbers of documents a word must be present in to be kept
                       norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
                       ngram_range= (1,2), #to indicate that we want to consider both unigrams and bigrams.
                       stop_words ='english') #to remove all common pronouns to reduce the number of noisy features

# features = tfidf.fit_transform(df.abstract).toarray()
features = tfidf.fit_transform(df['abstrak-jurnal']).toarray()
st.write(features)

labels = df.cat_id
features.shape

X_train, X_test, y_train, y_test = train_test_split(df['abstrak-jurnal'], df['cat_id'], random_state= 0)
count_vect = CountVectorizer()
CV = count_vect.fit(X_train)
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
TF = tfidf_transformer.fit(X_train_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_CV = count_vect.transform(X_test)
X_test_CV = tfidf_transformer.transform(X_test_CV)

clf = LinearSVC().fit(X_train_tfidf, y_train)

abstract = 'ini ibu budi membeli sebuah mainan baru citra'

try:
    if detect(abstract) == 'id':
        st.write(clf.predict(count_vect.transform([abstract])))
    else : st.write('Dokumen tidak berbahasa Indonesia')
except LangDetectException:
    st.write('Dokumen tidak berbahasa Indonesia')

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

class_names=['pengolahan','data mining','microcontroller','multimedia']
mapping = {'pengolahan': 0, ' data mining': 1, 'microcontroller': 2, 'multimedia': 3}

st.subheader('Confusion Matrix')
# plot_confusion_matrix(clf,X_test_CV,y_test,display_labels=mapping)
# st.pyplot()

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

def kfold_cross_validation_scikit(X_train, y_train, model):
	result = cross_validation_functions(model, X_train, y_train)
	st.subheader("Validation Result - KFold")
	st.write("Accuracy: %.2f" % (result['mean']*100), "%")
	st.write("Standard Deviation: %.2f" % (result['std']*100))
	st.write("Confusion Matrix:\n", result['conf_mat'])
	return

def cross_validation_functions(model, input, output):
   kfold = StratifiedKFold(n_splits=10)
   cv_results = cross_val_score(model, input, output, cv=kfold, scoring='accuracy')
   y_pred = cross_val_predict(model, input, output, cv=10)
   conf_mat = confusion_matrix(output, y_pred)
   mean = cv_results.mean()
   std = cv_results.std()
   return ({
      'cv_results': cv_results,
      'conf_mat': conf_mat,
      'mean': mean,
      'std': std
   })
kfold_cross_validation_scikit(X_train_tfidf, y_train, clf)


from yellowbrick.classifier import ROCAUC
X_test_counts = CV.transform(X_test)
X_test_tfidf = TF.transform(X_test_counts)
visualizer = ROCAUC(clf, classes=labels)
visualizer.fit(X_train_tfidf, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test_tfidf, y_test)        # Evaluate the model on the test data
visualizer.show()
# st.pyplot()

# st.subheader('ROC Curve')
# plot_roc_curve(clf,X_test_CV,y_test)
# st.pyplot()
# st.subheader('Precision-Recall Curve')
# plot_precision_recall_curve(clf,X_test_CV,y_test)
# st.pyplot()
#------------------------------------------------
# split
# split persebaran

df_text_genre = df[['abstrak-jurnal', 'cat_id']]
train_df, test_df = train_test_split(df_text_genre, test_size=0.2, random_state=42, shuffle=True)
train_cts = train_df.groupby("cat_id").size()
test_cts  = test_df.groupby("cat_id").size()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), sharey=True)
f.patch.set_facecolor('xkcd:mint green')
train_cts.plot(kind='bar',ax= ax1,rot=0)
test_cts.plot(kind='bar',ax= ax2,rot=0)
ax1.set_title('Train Set')
ax2.set_title('Test Set')
ax1.set_ylabel("Counts")

st.pyplot()

percents = 100 * train_df.groupby("cat_id").size() / train_df.shape[0]

percents.plot(kind='bar', title='Target Class Distributions', rot=0)
plt.ylabel("%")
# plt.show()
st.pyplot()

#------------------------------------------------
# split wordcloud
from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()
train_df["cat_id"] = labeler.fit_transform(train_df["cat_id"])
test_df["cat_id"]  = labeler.transform(test_df["cat_id"])

mapping = dict(zip(labeler.classes_, range(len(labeler.classes_))))
mapping = {'pengolahan': 0, ' data mining': 1, 'microcontroller': 2, 'multimedia': 3}

def plot_wordcloud(df: pd.DataFrame, category: str, target: int) -> None:
    words = " ".join(train_df[train_df["cat_id"] == target]["abstrak-jurnal"].values)

    plt.rcParams['figure.figsize'] = 10, 20
    wordcloud = WordCloud(stopwords=STOPWORDS, 
                          background_color="white",
                          max_words=1000).generate(words)

    plt.title("WordCloud For {}".format(category))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    st.pyplot()

for category, target in mapping.items():
    plot_wordcloud(train_df, category, target)
#------------------------------------------------
# feature extraction
# feature extraction BOW\
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_tf = count_vect.fit_transform(train_df["abstrak-jurnal"])

st.write("Shape of term-frequency matrix:", X_train_tf.shape)

st.write("Number of training documents: ", train_df.shape[0])

# feature extraction tfidf
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape

from feature_plots import plot_tfidf
from imblearn.pipeline import Pipeline

plot_tfidf(pipe    = Pipeline([("vect",count_vect), ("tfidf",tfidf_transformer)]),
           labeler = labeler,
           X       = train_df["abstrak-jurnal"],
           y       = train_df["cat_id"],
           vect    = "vect",
           tfidf   = "tfidf",
           top_n   = 25)

# svm modelling
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

def evaluate_model(
    train_df : pd.DataFrame,
    test_df  : pd.DataFrame,
    mapping  : dict,
    pipe     : Pipeline,
) -> None:

    model = pipe.fit(train_df["abstract"], 
                     train_df["cat_id"])


    pred  = model.predict(test_df["abstract"])

    st.write(classification_report(test_df["cat_id"],
                                pred
                                , target_names=mapping
                                ))
    
    st.write()
    st.write("balanced_accuracy", balanced_accuracy_score(test_df["cat_id"],pred))

from functools import partial

evaluate_pipeline = partial(evaluate_model,
                            train_df,
                            test_df,
                            mapping)

from sklearn.svm import LinearSVC

svm_pipe1 = Pipeline([('vect',    CountVectorizer()),
                      ('tfidf',   TfidfTransformer()),
                      ('model',   LinearSVC(random_state=50))])

evaluate_pipeline(svm_pipe1)

# oversample
from imblearn.over_sampling  import RandomOverSampler

svm_pipe2 = Pipeline([('vect',   CountVectorizer()),
                     ('tfidf',   TfidfTransformer()),
                     ('sampler', RandomOverSampler('minority',random_state=42)),
                     ('model',   LinearSVC(random_state=50))])

evaluate_pipeline(svm_pipe2)

# weight
svm_pipe3 = Pipeline([('vect',    CountVectorizer()),
                     ('tfidf',   TfidfTransformer()),
                     ('model',   LinearSVC(class_weight='balanced',
                                           random_state=50))])

evaluate_pipeline(svm_pipe3)

# plot

from feature_plots import plot_coefficients

plot_coefficients(
    pipe       = svm_pipe1,
    tf_name    = 'vect',
    model_name = 'model',
    ovr_num    = mapping["multimedia"],
    title      = "SVM",
    top_n      = 10
)

plot_coefficients(
    pipe       = svm_pipe2,
    tf_name    = 'vect',
    model_name = 'model',
    ovr_num    = mapping["multimedia"],
    title      = "Oversampled SVM",
    top_n      = 10
)

plot_coefficients(
    pipe       = svm_pipe3,
    tf_name    = 'vect',
    model_name = 'model',
    ovr_num    = mapping["multimedia"],
    title      = "Weighted SVM",
    top_n      = 10
)

# end test section
#------------------------------------------------


""" yang bisa store nilai banyak2 cuma object cuma constructor
    fungsi biasa gak bisa cuma bisa store 1 (atau lebih dari 1 di python)
    apalagi fungsi void fungsi ujung"""

""" playing separating oop<-pattern:
    Do partial to partial Right after goal start goal
    track, no need? dont track (don't make the function for that or value return of it)
    A function is always blackbox, no variable in it can escape except from the return or 
    outer reassign. Outer reassign itself has limitation, it has to be global. the variable 
    is only its own and the parameters which is again cannot escape. When you trying to
    escape or track something inside a function you better ... which is need to run once
    more time and not pure
    """

"""
discovered key:
    fit a support vector machine
    consistent
    Without a modeling pipeline, the data preparation steps may be performed manually twice: 
    once for evaluating the model and once for making predictions. Any changes to the sequence 
    must be kept consistent in both cases, otherwise differences will impact the capability 
    and skill of the model.
        mainstream case mungkin dengan pemanggilan fungsi (defined) dg return value banyak tapi 
        tetap, pemanggilan fungsi akan menjalankan proses lg yg mungkin lama dan pasti proses 
        berbeda dari proses pemanggilan fungsi pertama
    A pipeline ensures that the sequence of operations is defined once and is consistent when 
    used for model evaluation or making predictions.
        Jadi pipeline itu versi upgrade pemanggilan fungsi (def) biasa yang tidak menjalankan 
        proses jika sudah pernah dipanggil (sekali saja proses). Ini diperlukan untuk proses
        data processing/ ML aja karena emang mereka perlu gitu biar datanya kalau di evaluasi
        jadi konsisten. Kalau aplikasi mainstream ya fungsi def biasa aja  
"""