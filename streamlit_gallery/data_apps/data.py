import streamlit as st

def st_header(text, font_size=30, color=st.get_option('theme.primaryColor')):
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=color, font_size=font_size)
    st.markdown("<h1{}>{}</h1>".format(text_style, text), unsafe_allow_html=True)

def main():
    df = st.session_state.df
    # st.header('Data klasifikasi')
    col1, col2= st.columns(2)

    with col1:
        st_header('Train set')
        with st.expander('info'):
            st.write('diagram')
        st.write(df)
    
    with col2:
        st_header('Test set')
        with st.expander('info'):
            st.write('diagram')
        st.dataframe(df)

if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()