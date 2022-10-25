import streamlit as st

def main():
    df = st.session_state.df
    # st.header('Data klasifikasi')
    col1, col2= st.columns(2)

    with col1:
        st.subheader('Train set')
        with st.expander('info'):
            st.write('diagram')
        st.write(df)
    
    with col2:
        st.subheader('Test set')
        with st.expander('info'):
            st.write('diagram')
        st.dataframe(df)

if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()