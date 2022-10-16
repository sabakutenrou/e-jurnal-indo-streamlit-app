from contextlib import suppress
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_gallery.utils.lottie import lottie_show

from abstrak import pdfparser, abstractExtraction, between
from pathlib import Path
import tempfile
import re

import subprocess

def main():
    # reload data
    # st.header("Input data")
    accent_color = st.get_option('theme.primaryColor')
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=accent_color, font_size=35)
    st.markdown("<h1{}>Input data</h1>".format(text_style), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    files = col1.file_uploader('upload file (.pdf)', type="pdf", accept_multiple_files=True)

    if files is not None:
        for uploaded in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                fp = Path(tmp_file.name)
                fp.write_bytes(uploaded.getvalue())
                st.markdown("## Original PDF file")
                # st.write(show_pdf(tmp_file.name))
                file = tmp_file.name
                # text = pdfparser('132747-ID-klasifikasi-berita-online-menggunakan-me.pdf')
                text = pdfparser(file)

                desired = between(text,"abstrak","kata kunci")
                # st.write('The abstract of the document is :' + desired)

                text = desired.encode('ascii','ignore').lower() # It returns an utf-8 encoded version of the string & Lowercasing each word
                text = text.decode('ISO-8859-1')
                keywords = re.findall(r'[a-zA-Z]\w+',text)
                # st.write(keywords)

                # text = pdfparser(file)
                # parag = abstractExtraction(text, 'abstrak')
                # st.write(parag)
                # st.markdown('---')
                # st.write(text)

                # java = subprocess.call(["java","-cp",r'"D:\PythonProjects\New folder\KEJBIMS streamlit-gallery-main\cermine-impl-1.13-jar-with-dependencies.jar"',"pl/edu/icm/cermine/ContentExtractor","-path",r'"D:\PythonProjects\New folder\KEJBIMS streamlit-gallery-main\1603-3668-1-SM.pdf"'], shell=True)
                java = subprocess.call('java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path cermine-pdfs')
                st.write(java)
                # $ java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path cermine-pdfs
        with col2:
            if not files:
                lottie_url_download = "https://assets10.lottiefiles.com/packages/lf20_voi0gxts.json"
                lottie_show(lottie_url_download)
    
    # st_lottie(lottie_download, key="hello")
    


if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()