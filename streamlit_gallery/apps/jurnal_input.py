from contextlib import suppress
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_gallery.utils.lottie import lottie_show

def main():
    # reload data
    # st.header("Input data")
    accent_color = st.get_option('theme.primaryColor')
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=accent_color, font_size=35)
    st.markdown("<h1{}>Input data</h1>".format(text_style), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    file = col1.file_uploader('upload file yang akan diklasifikasikan')
    if file is None:
        col1.empty()
    else: col1.button('clasify')

    with col2:
        lottie_url_download = "https://assets10.lottiefiles.com/packages/lf20_voi0gxts.json"
        lottie_show(lottie_url_download, key="empty")

    # st_lottie(lottie_download, key="hello")


if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()