import streamlit as st

from streamlit_gallery import apps, data_apps
from streamlit_gallery.utils.page import page_group

import pandas as pd
from PIL import Image

from streamlit_gallery.utils.model import load_data

def main():
    image = Image.open('e-jurnal_logo.png')
    page = page_group("p")
    
    if 'df' not in st.session_state:
        df = load_data()
        st.session_state.df = df

    with st.sidebar:
        st.image(image)

        with st.expander("✨ KLASIFIKASI", True):
            page.item("Home", apps.home, default=True)
            page.item("Input Jurnal", apps.jurnal_input)
            page.item("Data Jurnal", apps.jurnal_data)

        with st.expander("🧩 DATA", True):
            page.item("Data klasifikasi", data_apps.data)
            page.item("Model klasifikasi", data_apps.model)

        with st.expander("⭐ LAINNYA", False):
            st.write('Tentang Aplikasi')
            st.write('aplikasi ini di buat oleh Dian Mahesa')
        st.write(st.session_state)
    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()
