import streamlit as st
import base64

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

show_pdf('contents.pdf')