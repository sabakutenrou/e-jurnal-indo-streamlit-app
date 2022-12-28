import streamlit as st
import plotly.graph_objects as go

from streamlit_apps.utils.labels import get_labels
from streamlit_apps.utils.st_components import st_title
from streamlit_apps.utils.model import load_data
import pandas as pd

def st_header(text, font_size=30, color=st.get_option('theme.primaryColor')):
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=color, font_size=font_size)
    st.markdown("<h1{}>{}</h1>".format(text_style, text), unsafe_allow_html=True)

def show_data(df):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    # fill_color='#262730',
                    align='left'),
        cells=dict(values=[df["judul-abstrak"], df["kategori"]],
                # fill_color='#262730',
                    align='left',
                    height=10
                    ))
    ])

    fig.update_layout(
        margin=dict(l=0, r=20, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def main():
    df = load_data()

    st_title("Total data : " + str(len(df)), font_size=20)
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(df , random_state= 0)

    train_set['split'] = 'train'
    test_set['split'] = 'test'

    train_set['count'] = 1
    test_set['count'] = 1

    new_train_set = train_set[['kategori','count','split']]
    new_test_set = test_set[['kategori','count','split']]

    # result_df = pd.append([new_train_set, new_test_set], axis=1, join='inner')
    result_df = new_train_set.append(new_test_set)
    result_df = pd.concat([new_train_set,new_test_set], axis=0, ignore_index=True)

    # result_df['kategori'] = result_df['kategori'].map({'hardware programming': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4})
    # result_df['kategori'] = result_df['kategori'].cat.rename_categories({'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4})

    mapping = get_labels()

    result_df['kategori'] = result_df['kategori'].replace(mapping, regex=True)
    # st.write(df)

    # st.write(result_df)

    import plotly.express as px
    # df = px.data.tips()
    # st.write(df[['sex', 'total_bill','smoker']])

    # st.write(pd.get_dummies(df['sex']))

    fig = px.histogram(result_df, x="kategori", y="count",
                color='split', barmode='group',
                height=400,
                labels={'count':'journal'},
                #  color_discrete_sequence=['indianred','aqua']
                )
    st.plotly_chart(fig, use_container_width=True, theme='streamlit')

    
    col1, col2= st.columns(2)
    
    with col1:
        st_header('Train set')
        # with st.expander('info'):
        #     st.write('diagram')
        # show_data(train_set[['judul-abstrak', 'kategori']])
        st.write(train_set[['judul-abstrak', 'kategori']])

    with col2:
        st_header('Test set')
        # with st.expander('info'):
        #     st.write('diagram')
        # show_data(test_set[['judul-abstrak', 'kategori']])
        st.write(test_set[['judul-abstrak', 'kategori']])

if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()