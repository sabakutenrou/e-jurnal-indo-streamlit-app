import streamlit as st

import PyPDF2
import data_func
import csv

# reader = PyPDF2.PdfFileReader(
#     './data/original/Complete_Works_Lovecraft.pdf')

reader = PyPDF2.PdfFileReader(
    '7155-1-50405-1-10-20200630.pdf')

# st.write(reader.documentInfo)

num_of_pages = reader.numPages
# st.write('Number of pages: ' + str(num_of_pages))

writer = PyPDF2.PdfFileWriter()

writer.addPage(reader.getPage(1))
    
output_filename = 'contents.pdf'

with open(output_filename, 'wb') as output:
    writer.write(output)

text = data_func.convert_pdf_to_string(
    '132747-ID-klasifikasi-berita-online-menggunakan-me.pdf')

# text = text.replace('.','')
# text = text.replace('\x0c','')
# table_of_contents_raw = text.split('\n')
# st.write(text)

import abstrak
text = abstrak.pdfparser('132747-ID-klasifikasi-berita-online-menggunakan-me.pdf')

para = abstrak.abstractExtraction(text, 'abstrak')
st.write(para)
st.markdown('---')  
st.write(text)