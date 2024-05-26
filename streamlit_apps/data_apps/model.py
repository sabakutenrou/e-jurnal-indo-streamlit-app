import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from streamlit_apps.utils.feature_plots import get_top_features, plot_tfidf, plot_coefficients, top_n_tfidfs
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import matplotlib as plt
from joblib import dump, load
from streamlit_apps.utils.st_components import st_header, st_title, get_accent_color
from streamlit_apps.utils.preprocess import remove_stopwordss, preprocess
from streamlit_apps.utils.model import load_data

def main():
    accent_color = get_accent_color()
    
    st_header("Model Klasifikasi 📑")

    # st.write(load_data(clean_data=True))

    process_filter = st.multiselect('Implementasi', ['Preprocessing','Tfidf'], default=['Preprocessing','Tfidf'])

    if('Preprocessing' in process_filter):
        df = load_data(clean_data=True)
        # features = X_train_vect.toarray() #
    else:
        df = load_data(clean_data=False)
    
    # df['kategori'] = df['kategori'].factorize()[0]    # category_id column auto inserted?
    
    X = df['judul-abstrak']
    y = df['kategori']

    X_train, X_test, y_train, y_test = train_test_split(X, y , random_state= 0)
    
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    if 'Tfidf' in process_filter:
        pipeline    = Pipeline([("vect",count_vect), ("tfidf",tfidf_transformer)])
        vect = ['vect','tfidf']
        vectorizer_text = "TFIDF"
    else:
        pipeline    = Pipeline([("vect",count_vect)])
        vect = ['vect']
        vectorizer_text = "BOW"

    X_train_vect = pipeline.fit_transform(X_train)
    features = X_train_vect.toarray()
    X_test_vect = pipeline.transform(X_test)

    # feature_n = TfidfVectorizer().fit(X_train)
    feature_names = pipeline.get_feature_names_out()

    nodes = ("D{0}".format(i) for i in range(1,features.shape[0]+1))
    # corpus_index = [n for n in X_train]]
    tfidf_df = pd.DataFrame(X_train_vect.T.todense(), index=feature_names, columns=nodes)

    st.markdown('---')
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st_title('MATRIKS '+ vectorizer_text, font_size=24, margin_top=10)
        st.markdown(' ')
        st.markdown(' ')
        # st.markdown('##### Jumlah dokumen :')
        st.write('Jumlah dokumen latih:')
        st_title(features.shape[0])
        # st.markdown('##### Jumlah feature :')
        st.write('Jumlah fitur:')
        st_title(features.shape[1])

    labels = df['kategori']  ### needs check

    with col2:
        with st.expander('tabel {}'.format(vectorizer_text), expanded=True):
            st.dataframe(tfidf_df)
    
    st.markdown('---')

    mapping = {'data mining': 0, 'hardware programming': 1, 'jaringan': 2, 'multimedia': 3, 'pengolahan citra': 4}
    # from streamlit_gallery.utils.labels import get_labels
    # mapping = get_labels()

    st_title('Fitur pada setiap kelas', font_size=24)

    labeler = LabelEncoder()
    y_train = labeler.fit_transform(y_train)
    y_test  = labeler.transform(y_test)
    y_tes = labeler.inverse_transform(y_test)

    # st.write(y_train, y_test, y_tes)

    # top_features = plot_tfidf(pipe    = pipeline,
    #                 labeler = labeler,  # coba mapping
    #                 X       = X_train,
    #                 y       = y_train,
    #                 features= feature_names,
    #                 vect    = vect)
    
    top_features = plot_tfidf(
                    labeler = labeler,  # coba mapping
                    X       = X_train_vect,
                    y       = y_train,
                    features= feature_names)

    # list_kata = top_features.loc
    # st.write(top_features)
    # kata_tertinggi = [top_features.loc[top_features['class_name'] == value] for key, value in mapping]
    # st.write(kata_tertinggi)

    # list_kata = []
    # for kata in kata_tertinggi:
    #     kata = kata.loc[kata['tfidf'].idxmax()]
    # #     list_kata.append(kata.feature)

    # # st.write(list_kata)

    # st.info("""Panjang dari setiap bar menunjukkan rata-rata skor fitur pada 
    # semua dokumen berdasarkan kelas targetnya. Berdasarkan yang dapat dilihat 
    # pada grafik diatas kata "{}", "{}", dll adalah kata yang memiliki skor 
    # yang tinggi. Beberapa kata lain seperti "{}", "{}", "{}" memiliki 
    # skor dan frekuensi yang tinggi pada kelas yang lain. Diharapkan suatu model 
    # classifier yang sensitif pada kata-kata tersebut untuk mempertimbangkan kelas 
    # target dengan tepat.
    # """.format(list_kata[0].upper(), list_kata[1].upper(), list_kata[2].upper(), list_kata[3].upper(), list_kata[4].upper()))
    # # format(top_features['feature'].iloc[0].upper(), top_features['feature'].iloc[1].upper(), top_features['feature'].iloc[2].upper(), top_features['feature'].iloc[3].upper(), top_features['feature'].iloc[4].upper()))
    # #  sementara kata-kata dengan sensitivitas rendah menjadi faktor penentu juga pada kelas yang lain
    st.markdown('---')

    # svm modelling
    st_header('Support Vector Machine')

    col21, empty, col22 = st.columns([2,0.1,1.5])
    with(col21):
        # st.write('accuracy : ""')
        st.write("""Support Vector Machine adalah sebuah model prediksi 
        dengan menggunakan analisa pola data untuk melakukan klasifikasi. 
        SVM bertujuan untuk mencari hyperplane terbaik yang berfungsi sebagai 
        pemisah data antar kelas. SVM memiliki prinsip dasar linear classifier 
        yaitu klasifikasi yang memisahkan dua kelas secara linier, namun SVM 
        dalam pengembangannya dapat bekerja pada masalah non-linier dengan 
        memanfaatkan kernel. 
        """)
    with(col22):
        image = Image.open('e-jurnal_logo.png')
        st.image(image)
    # pcm = plot_confusion_matrix(clf,X_test_vect,y_test,display_labels=mapping)
    
    clf = LinearSVC().fit(X_train_vect, y_train)
    # mapping = {'pengolahan citra': 0, 'jaringan': 1, 'hardware programming': 2, 'data mining': 3, 'multimedia': 4}
    mapping = {'data mining': 0, 'hardware programming': 1, 'jaringan': 2, 'multimedia': 3, 'pengolahan citra': 4}

    from sklearn.metrics import ConfusionMatrixDisplay
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(False)
    confusion_matrix = ConfusionMatrixDisplay.from_estimator(clf,X_test_vect,y_test,display_labels=mapping,cmap=plt.cm.GnBu_r)
    # confusion_matrix.ax_.grid(False)
    # confusion_matrix.ax_.tick_params(colors='#6dd3b6', which='both')
    # confusion_matrix.ax_.xaxis.label.set_color('#6dd3b6')
    # confusion_matrix.ax_.yaxis.label.set_color('#6dd3b6')
    # confusion_matrix.figure_.set_alpha(.5)
    # confusion_matrix.figure_.patch.set_alpha(.0)
    # st.pyplot(confusion_matrix.figure_)

    # st.text(confusion_matrix.text_)

    import plotly.express as px

    z = confusion_matrix.text_

    x = list(mapping.keys())
    y = x

    z_text = [[y.get_text() for y in x] for x in z]
    # st.text(z_text)

    z = [[int(y) for y in x] for x in z_text]

    # set up figure 
    fig = px.imshow(z, x=x, y=y, 
        text_auto=True,
        color_continuous_scale='GnBu_r', 
        aspect="auto")

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color=accent_color,size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="#6dd3b6",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


    from sklearn.metrics import classification_report
    from sklearn.metrics import balanced_accuracy_score

    
    train_df = pd.DataFrame({'abstract': X_train, 'cat_id': y_train})
    test_df = pd.DataFrame({'abstract': X_test, 'cat_id': y_test})
    
    def evaluate_model(
        train_df : pd.DataFrame,
        test_df  : pd.DataFrame,
        mapping  : dict,
        pipe     : Pipeline,
    ) -> None:

        model = pipe.fit(train_df['abstract'], 
                        train_df['cat_id'])

        pred  = model.predict(test_df['abstract'])

        st_title('Classification report', font_size=26, margin_top=20, margin_bottom=10)
        report = classification_report(test_df['cat_id'],
                                    pred, 
                                    target_names=mapping,
                                    # output_dict=True
                                    )
        st.code("++" + report)
        # st.write(report)
        # st.write("balanced_accuracy", balanced_accuracy_score(test_df['cat_id'],pred))

    from functools import partial

    train_df = pd.DataFrame({'abstract': X_train, 'cat_id': y_train})
    test_df = pd.DataFrame({'abstract': X_test, 'cat_id': y_test})
    evaluate_pipeline = partial(evaluate_model,
                                train_df,
                                test_df,
                                mapping)

    svm_pipe1 = Pipeline([('vect',    CountVectorizer()),
                        ('tfidf',   TfidfTransformer()),
                        # ('model',   LinearSVC(random_state=50)),
                        ('model',   LinearSVC())
                        ])

    evaluate_pipeline(svm_pipe1)

    # # oversample
    # from imblearn.over_sampling  import RandomOverSampler

    # svm_pipe2 = Pipeline([('vect',   CountVectorizer()),
    #                     ('tfidf',   TfidfTransformer()),
    #                     ('sampler', RandomOverSampler('minority',random_state=42)),
    #                     ('model',   LinearSVC(random_state=50))])

    # evaluate_pipeline(svm_pipe2)

    # # weight
    # svm_pipe3 = Pipeline([('vect',    CountVectorizer()),
    #                     ('tfidf',   TfidfTransformer()),
    #                     ('model',   LinearSVC(class_weight='balanced',
    #                                         random_state=50))])

    # evaluate_pipeline(svm_pipe3)

    # plot
    st_title('Most predictive words', font_size=26, margin_top=20,  margin_bottom=10)

    col1, emp, col2 = st.columns([2,0.1, 4])
    with(col1):
        model = st.radio('model:', list(mapping.keys())) # model = st.radio('model:', ('pengolahan citra','data mining','microcontroller','multimedia'))
    with(col2):
        if model == 'pengolahan citra':
            plot_coefficients(
                    pipe       = svm_pipe1,
                    tf_name    = 'vect',
                    model_name = 'model',
                    ovr_num    = mapping["pengolahan citra"],
                    title      = "SVM",
                    top_n      = 10)
        elif model == 'data mining':
            plot_coefficients(
                    pipe       = svm_pipe1,
                    tf_name    = 'vect',
                    model_name = 'model',
                    ovr_num    = mapping["data mining"],
                    title      = "SVM",
                    top_n      = 10)
        elif model == 'jaringan':
            plot_coefficients(
                    pipe       = svm_pipe1,
                    tf_name    = 'vect',
                    model_name = 'model',
                    ovr_num    = mapping["jaringan"],
                    title      = "SVM",
                    top_n      = 10)
        elif model == 'multimedia':
            plot_coefficients(
                    pipe       = svm_pipe1,
                    tf_name    = 'vect',
                    model_name = 'model',
                    ovr_num    = mapping["multimedia"],
                    title      = "SVM",
                    top_n      = 10) 
        elif model == 'hardware programming':
            plot_coefficients(
                    pipe       = svm_pipe1,
                    tf_name    = 'vect',
                    model_name = 'model',
                    ovr_num    = mapping["hardware programming"],
                    title      = "SVM",
                    top_n      = 10)                       

    # for key, value in mapping.items():
    #     st.subheader(key)
    #     plot_coefficients(
    #         pipe       = svm_pipe1,
    #         tf_name    = 'vect',
    #         model_name = 'model',
    #         ovr_num    = value,
    #         title      = "SVM",
    #         top_n      = 10
    #     )

    # plot_coefficients(
    #     pipe       = svm_pipe2,
    #     tf_name    = 'vect',
    #     model_name = 'model',
    #     ovr_num    = mapping["multimedia"],
    #     title      = "Oversampled SVM",
    #     top_n      = 10
    # )

    # plot_coefficients(
    #     pipe       = svm_pipe3,
    #     tf_name    = 'vect',
    #     model_name = 'model',
    #     ovr_num    = mapping["multimedia"],
    #     title      = "Weighted SVM",
    #     top_n      = 10
    # )

    # end test section
    #------------------------------------------------

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

    # kfold_cross_validation_scikit(X_train_tfidf, y_train, clf)

    from yellowbrick.classifier import ROCAUC
    # X_test_counts = count_vect.transform(X_test)
    # X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    visualizer = ROCAUC(clf, classes=labels)
    # visualizer.fit(X_train_tfidf, y_train)        # Fit the training data to the visualizer
    # visualizer.score(X_test_tfidf, y_test)        # Evaluate the model on the test data
    # visualizer.show()
    # st.pyplot()

    # st.subheader('ROC Curve')
    # plot_roc_curve(clf,X_test_CV,y_test)
    # st.pyplot()
    # st.subheader('Precision-Recall Curve')
    # plot_precision_recall_curve(clf,X_test_CV,y_test)
    # st.pyplot()

if __name__ == "__main__":
    main()
