import streamlit as st

file_uploader = st.file_uploader("file uploader", accept_multiple_files=True)
st.write(file_uploader.__dir__())
z = []
st.write(125 in z)
for file in file_uploader:
    st.write(file.size)
    z.append(file.size)
st.text(str(z))