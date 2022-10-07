import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from streamlit_gallery.utils.feature_plots import get_top_features, plot_tfidf, plot_coefficients, top_n_tfidfs
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

from streamlit_gallery.utils.preprocess import remove_tweet_special

def main():
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    accent_color = st.get_option('theme.primaryColor')
    st.header("Model Klasifikasi 📑")

    df = pd.read_csv("dataset.csv")
    df = df[['judul-jurnal','abstrak-jurnal','kategori']]
    df['kategori'] = df['kategori'].factorize()[0]    # category_id column auto inserted? 
    ## filter by id
    # st.table(df.head(2))
    # AgGrid(df)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    grid_options = gb.build()
    # AgGrid(df, gridOptions=grid_options)


    impl = ['Preprocessing','Tfidf']
    multi_sel = st.multiselect('Implementasi', impl, default=impl)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    if(impl[0] in multi_sel):
            df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(remove_tweet_special)
    if(impl[1] in multi_sel):
        pipeline    = Pipeline([("vect",count_vect), ("tfidf",tfidf_transformer)])
        vect = ['vect','tfidf']
        vectorizer_text = "TFIDF"
    else:
        pipeline = Pipeline([("vect",count_vect)])
        vect = ['vect']
        vectorizer_text = "BOW"

    X = df['abstrak-jurnal']
    y = df['kategori']

    X_train, X_test, y_train, y_test = train_test_split(X, y , random_state= 0)

    features = pipeline.fit_transform(X_train).toarray()

    st.markdown('---')

    st.subheader('MATRIKS {}'.format(vectorizer_text))
    
    def markdown(text:str, color:str=accent_color, font_size:int=40):
        st.markdown('<div style="color:{}; font-size: {}px; font-weight:600; margin-top:-20px">{}</div>'
            .format(color, font_size, text), unsafe_allow_html=True)
    
    st.markdown('##### Jumlah dokumen :')
    markdown(features.shape[0])
    st.markdown('##### Jumlah feature :')
    markdown(features.shape[1])

    labels = df.kategori  ### needs check

    X_train_vect = pipeline.fit_transform(X_train)
    clf = LinearSVC().fit(X_train_vect, y_train)
    X_test_vect = pipeline.transform(X_test)

    nodes = ("D{0}".format(i) for i in range(1,features.shape[0]+1))
    feature_n = TfidfVectorizer().fit(X_train)
    feature_names = feature_n.get_feature_names_out()
    # corpus_index = [n for n in X_train]
    df = pd.DataFrame(X_train_vect.T.todense(), index=feature_names, columns=nodes)
    with st.expander('tabel {}'.format(vectorizer_text)):
        st.dataframe(df)

    mapping = {'pengolahan citra': 0, 'jaringan': 1, 'hardware programming': 2, 'data mining': 3, 'multimedia': 4}

    # ----

    # st.write("Shape of term-frequency matrix:", X_train_tf.shape)
    # st.write("Number of training documents: ", train_df.shape[0])
    # st.write(' ')
    st.markdown('---')
    st.subheader('Fitur pada setiap kelas')

    labeler = LabelEncoder()
    y_train = labeler.fit_transform(y_train)
    y_test  = labeler.transform(y_test)

    top_features = plot_tfidf(pipe    = pipeline,
                    labeler = labeler,  # coba mapping
                    X       = X_train,
                    y       = y_train,
                    vect    = vect)

    st.info("""Panjang dari setiap bar menunjukkan rata-rata score fitur pada 
    semua dokumen berdasarkan kelas targetnya. Berdasarkan yang dapat kita lihat 
    dari grafik diatas kata "{}", "{}", dll adalah kata umum yang memiliki score 
    yang tinggi. Kita juga dapat lihat beberapa kata seperti "{}", "{}", "{}" memiliki 
    skor dan frekuensi yang tinggi pada kelas yang lain. Kita ingin model 
    classifier sensitif pada kata-kata tersebut untuk mempertimbangkan kelas 
    target dengan tepat ketika kata-kata dengan sensitivitas rendah diharapkan 
    dapat menjadi faktor penentu juga pada kelas yang lain.
    """.format(top_features['feature'].iloc[0], top_features['feature'].iloc[1], top_features['feature'].iloc[1], top_features['feature'].iloc[1], top_features['feature'].iloc[1]))

    # svm modelling

    st.markdown('---')
    st.subheader('Support Vector Machine')

    col21, emp, col22 = st.columns([2,0.1,1.5])
    with(col21):
        st.write('accuracy : ""')
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

    def evaluate_model(
        train_df : pd.DataFrame,
        test_df  : pd.DataFrame,
        mapping  : dict,
        pipe     : Pipeline,
    ) -> None:

        model = pipe.fit(X_train, 
                        y_train)

        pred  = model.predict(X_test)

        st.subheader('Classification report')
        st.code(classification_report(y_test,
                                    pred, 
                                    target_names=mapping
                                    ))
        st.write("balanced_accuracy", balanced_accuracy_score(y_test,pred))

    from functools import partial

    train_df = pd.DataFrame({'abstract': X_train, 'cat_id': y_train})
    test_df = pd.DataFrame({'abstract': X_test, 'cat_id': y_test})
    evaluate_pipeline = partial(evaluate_model,
                                train_df,
                                test_df,
                                mapping)


    svm_pipe1 = Pipeline([('vect',    CountVectorizer()),
                        ('tfidf',   TfidfTransformer()),
                        ('model',   LinearSVC(random_state=50))])

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
    st.subheader('Most predictive words')

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
