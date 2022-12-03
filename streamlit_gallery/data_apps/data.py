import streamlit as st
import plotly.graph_objects as go

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
    df = st.session_state.df
    col1, col2= st.columns(2)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df , random_state= 0)

    with col1:
        st_header('Train set')
        # with st.expander('info'):
        #     st.write('diagram')
        show_data(train)

    with col2:
        st_header('Test set')
        # with st.expander('info'):
        #     st.write('diagram')
        show_data(test)

if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()