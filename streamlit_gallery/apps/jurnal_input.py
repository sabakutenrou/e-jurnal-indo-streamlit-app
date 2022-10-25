from ast import pattern
from contextlib import suppress
from turtle import bgcolor
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_gallery.utils.lottie import lottie_show, lottie_json, show_lottie_json

from abstrak import pdfparser, abstractExtraction, between
from pathlib import Path
import tempfile
import re

import subprocess

import os
import xml.etree.ElementTree as ET
import glob
from streamlit_lottie import st_lottie_spinner

from streamlit_gallery.utils.streamlit_prop import get_theme_colors

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    # return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def parse_hasil_cermine(file_path):
    # tree = ET.parse('tempDir/132747-ID-klasifikasi-berita-online-menggunakan-me.cermxml')
    tree = ET.parse(file_path)

    # get the parent tag 
    root = tree.getroot()

    # print the root (parent) tag along with its memory location 
    # st.write(root) 

    # print the attributes of the first tag  
    # st.write(root[0].attrib) 

    # print the text contained within first subtag of the 5th tag from the parent 
    # st.write(root[0][0][0][0].text)
    # st.write(root[0][1][0][0].text)
    
    # for child in root:
    #     st.write(child.tag, child.attrib)

    # beda List Comprehension dan Standart Comprehension
    # LC : as list
    texts = [elem.text for elem in root[1].iter()]

    # SC : to single (instance) each
    # text = ""
    # for elem in root[1].iter():
    #     add = elem.text
    #     if add == None: add = ""
    #     text = text + add

    # st.write(text)

    judul = "".join([judul.text for judul in root.iter('article-title')])
    # judul = "".join(judul)
    abstract = "".join([abstract[0].text for abstract in root.iter('abstract')])
    # for abstract in root.iter('abstract'):
    #     #1
    #     # abstract>p tag
    #     abstract = abstract[0].text 
    #     #2
    #     # for text in abstract.itertext():
    #     #     abstract = text

    # if abstract == "":
    #     # root>body tag
    #     for elem in root[1].iter():
    #         add = elem.text
    #         if add == None: add = ""
    #         abstract = abstract + add
    # # st.write(abstract)

    if abstract == "":
        from abstrak import pdfparser
        file_path = file_path.replace('cermxml', 'pdf')
        parsed = pdfparser(file_path)
        pattern = re.compile('(?<=abstrak)(.*)(?=kata kunci)', re.IGNORECASE)
        abstract = re.findall(pattern, parsed)
    
    return judul, abstract

def delete_files():
    import os, shutil
    folder = 'tempDir'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                # st.write(file_path)
        except Exception as e:
            st.write('Failed to delete %s. Reason: %s' % (file_path, e))

def st_progress(progress, percent, empty, message):
    progress.progress(percent)
    empty.write(message)

def main():
    # reload data
    # st.header("Input data")
    delete_files()
    colors = get_theme_colors()
    accent_color = colors['primaryColor']
    background_color = colors['backgroundColor']
    scnd_background_color = colors['secondaryBackgroundColor']
    text_color = colors['textColor']
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=accent_color, font_size=35)
    st.markdown("<h1{}>Input data</h1>".format(text_style), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([4,1,2,1])
    files = col1.file_uploader('upload file jurnal (.pdf)', type="pdf", accept_multiple_files=True)

    total_files = len(files)
    area_hasil = st.container()

    if total_files != 0:
        # message = st.empty()
        lottie = lottie_json("lf30_editor_ilf1k19x.json")
        with col3:
            message = st.empty()
            
            with st_lottie_spinner(lottie):
                message_md = """
                    <p style="text-align: center; color: {}; font-size:10px"></p>
                    """.format(accent_color)
                message.markdown(message_md, unsafe_allow_html=True)
                for pdf_file in files:
                    # file_details = {"FileName":pdf_file.name,"FileType":pdf_file.type}
                    # with open(os.path.join("tempDir",pdf_file.name),"wb") as f: 
                    #     f.write(pdf_file.getbuffer())
                    save_uploadedfile(pdf_file)

                
                # for file in os.listdir("/tempDir"):
                #     if file.endswith(".cermxml"):
                #         print(os.path.join("/tempDir", file))

                # getting pdf files list, just for fun. not used
                # file_list = [os.path.join("tempDir", file) for file in os.listdir("tempDir") if file.endswith(".pdf")]
                # st.write(file_list)

                # for file_pdf in file_list:
                java = subprocess.call('java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path tempDir -outputs jats')
                xml_list = [os.path.join("tempDir", file) for file in os.listdir("tempDir") if file.endswith(".cermxml")]
                
                hasil_abstract = [parse_hasil_cermine(xml) for xml in xml_list]
                
                message.empty()

            # show_lottie_json("square-loading.json")
        with area_hasil:
            # st.write(hasil_abstract)
            st.button('download hasil (.csv)')
            hasil_md = """
            <div class="column_1" style="background: {bgcolor}; padding: 20px; border-radius: 5px; margin-top: 20px; width: 75%; float: left;">
                <h1 style="color:{color}; font-size:20px">{}</h1>
                <p>author</p>
                <p>kata kunci</p>
                <p>pub date</p>
                <p>jurnal</p>
                <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                    <p style="color:{color}; font-size:15px">{}</p>
                </div>
            </div>
            <div class="column_2" style="padding: 20px; border-radius: 5px; width: 23%; float: left;">
                <h1 style="color:{textcolor}; text-align: center;">pengolahan citra</h1>
            </div>"""
            
            for hasil in hasil_abstract:
                st.markdown(hasil_md.format(hasil[0], hasil[1][0], color=text_color, bgcolor=scnd_background_color, textcolor=accent_color), unsafe_allow_html=True)


        # for uploaded in files:
        #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        #         fp = Path(tmp_file.name)
        #         fp.write_bytes(uploaded.getvalue())
        #         st.markdown("## Original PDF file")
        #         # st.write(show_pdf(tmp_file.name))
        #         file = tmp_file.name
        #         # text = pdfparser('132747-ID-klasifikasi-berita-online-menggunakan-me.pdf')
        #         text = pdfparser(file)

        #         desired = between(text,"abstrak","kata kunci")
        #         # st.write('The abstract of the document is :' + desired)

        #         text = desired.encode('ascii','ignore').lower() # It returns an utf-8 encoded version of the string & Lowercasing each word
        #         text = text.decode('ISO-8859-1')
        #         keywords = re.findall(r'[a-zA-Z]\w+',text)
        #         # st.write(keywords)

        #         # text = pdfparser(file)
        #         # parag = abstractExtraction(text, 'abstrak')
        #         # st.write(parag)
        #         # st.markdown('---')
        #         # st.write(text)

        #         # java = subprocess.call(["java","-cp",r'"D:\PythonProjects\New folder\KEJBIMS streamlit-gallery-main\cermine-impl-1.13-jar-with-dependencies.jar"',"pl/edu/icm/cermine/ContentExtractor","-path",r'"D:\PythonProjects\New folder\KEJBIMS streamlit-gallery-main\1603-3668-1-SM.pdf"'], shell=True)
        #         java = subprocess.call('java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path cermine-pdfs')
        #         st.write(java)
        #         # $ java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path cermine-pdfs
    else:
        with col3:
            lottie_url_download = "https://assets10.lottiefiles.com/packages/lf20_voi0gxts.json"
            lottie_show(lottie_url_download)
    
    # st_lottie(lottie_download, key="hello")


if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()