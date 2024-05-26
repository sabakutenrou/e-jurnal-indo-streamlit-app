from ast import pattern
from contextlib import suppress
from turtle import bgcolor
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_apps.utils.lottie import lottie_show, lottie_json, show_lottie_json

from abstrak import pdfparser, abstractExtraction, between
from pathlib import Path
import tempfile
import re

import subprocess

import os
import xml.etree.ElementTree as ET
import glob
from streamlit_lottie import st_lottie_spinner

from streamlit_apps.utils.st_components import get_theme_colors
from streamlit_apps.utils.model import predict, con

from streamlit_apps.utils.database import insert_jurnal_indo, insert_jurnal_bucket
from random import randint

from streamlit_apps.utils.lang_detection import language_detection

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    # return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def parse_hasil_cermine(file_path):
    tree = ET.parse(file_path)

    root = tree.getroot()
    
    # beda List Comprehension dan Standart Comprehension
    # LC : as list
    # texts = [elem.text for elem in root[1].iter()]

    # SC : to single (instance) each
    # text = ""
    # for elem in root[1].iter():
    #     add = elem.text
    #     if add == None: add = ""
    #     text = text + add

    # judul
    list_judul = [elm.text for elm in root.findall("./front/article-meta/title-group/article-title") if elm.text is not None]

    # tahun
    list_tahun = [elm.text for elm in root.findall("./front/article-meta/pub-date/year") if elm.text is not None]

    # abstrak
    list_abstract = [elm.text for elm in root.findall("./front/article-meta/abstract/p") if elm.text is not None]
    
    # st.text("list_abstract -> " + str(list_abstract))
    # st.write("len(list_abstract) -> " + str(len(list_abstract)))

    # author
    list_author = [elm.text for elm in root.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']/string-name")  if elm.text is not None]

    # nama jurnal
    list_nama_jurnal = [elm.text for elm in root.findall("./front/journal-meta/journal-title-group/journal-title") if elm.text is not None]

    # keyword
    list_keyword = [elm.text for elm in root.findall("./front/article-meta/kwd-group/kwd") if elm.text is not None]
    
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


    def between(value, a, b):
        # Find and validate before-part.
        pos_a = value.find(a)
        # st.write(value)
        # st.write(pos_a)
        if pos_a == -1: return ""
        # Find and validate after part.
        pos_b = value.rfind(b)
        if pos_b == -1: return ""
        # Return middle part.
        adjusted_pos_a = pos_a + len(a)
        if adjusted_pos_a >= pos_b: return ""
        return value[adjusted_pos_a:pos_b]

    # desired = between(pagecontent,"Abstract","Keywords")
    # print('The abstract of the document is :' + desired)

    # text = desired.encode('ascii','ignore').lower() # It returns an utf-8 encoded version of the string & Lowercasing each word
    # text = text.decode('ISO-8859-1')
    # keywords = re.findall(r'[a-zA-Z]\w+',text)

    if len(list_abstract) == 0:
        from abstrak import pdfparser
        file_path = file_path.replace('cermxml', 'pdf')
        parsed = pdfparser(file_path)
        # pattern = re.compile('(?<=abstrak)(.*)(?=kata kunci)', re.IGNORECASE)
        # list_abstract = re.findall(pattern, parsed)
        # list_abstract = [abstract.strip() for abstract in list_abstract]
        list_abstract = [between(parsed, "abstrak", "kata kunci").strip().capitalize()]
        # list_abstract = [between(abstract.lower(), "abstrak", "kata kunci") for abstract in list_abstract]
        # st.write("len is 0 PATH")
        if len(list_abstract) == 0:
            list_abstract = [between(parsed, "intisari", "kata kunci").strip().capitalize()]
            # list_abstract = ["Tidak ditemukan"]
        # if len(list_abstract) == 0:
        #     list_abstract = ["Tidak ditemukan"]
    try:
        if language_detection(list_abstract[0]) != 'id' or language_detection(list_abstract[0]) != 'de':
            from abstrak import pdfparser
            file_path = file_path.replace('cermxml', 'pdf')
            parsed = pdfparser(file_path)
            # pattern = re.compile('(?<=abstrak)(.*)(?=kata kunci)', re.IGNORECASE)
            # list_abstract = re.findall(pattern, parsed)
            # list_abstract = [abstract.strip() for abstract in list_abstract]
            list_abstract = [between(parsed, "abstrak", "kata kunci").strip().capitalize()]
            # list_abstract = [between(abstract.lower(), "abstrak", "kata kunci") for abstract in list_abstract]
            if len(list_abstract) == 0:
                list_abstract = [between(parsed, "intisari", "kata kunci").strip().capitalize()]
            # if len(list_abstract) == 0:
            #     list_abstract = ["Tidak ditemukan"]
    except: list_abstract = [""]
    
    return {"judul":list_judul,
            "tahun":list_tahun,
            "abstrak":list_abstract,
            "author":list_author,
            "nama_jurnal":list_nama_jurnal,
            "keyword":list_keyword
            }

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

def show_pdf(file_path):
    import base64
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    if 'file_uploader' not in st.session_state:
        st.session_state['file_uploader'] = str(randint(1000, 100000000))
    if 'deta_jurnal' not in st.session_state:
        st.session_state['deta_jurnal'] = []

    delete_files() # reload
    colors = get_theme_colors()
    accent_color = colors['primaryColor']
    scnd_background_color = colors['secondaryBackgroundColor']
    text_color = colors['textColor']
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=accent_color, font_size=35)
    st.markdown("<h1{}>Input data</h1>".format(text_style), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([4,1,2,1])
    files = col1.file_uploader('upload file jurnal (.pdf)', 
                                type="pdf", 
                                accept_multiple_files=True, 
                                key=st.session_state['file_uploader'],
                                # key='file_uploader'
                                )

    total_files = len(files)
    area_hasil = st.container()
    
    if total_files != 0:
        def proses_hasil(files):
            lottie = lottie_json("lf30_editor_ilf1k19x.json")
            with col3:
                with st_lottie_spinner(lottie):
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
                    pdf_list = [os.path.join("tempDir", file) for file in os.listdir("tempDir") if file.endswith(".pdf")]
                    hasil_abstract = [parse_hasil_cermine(xml) for xml in xml_list]

                    # st.write(hasil_abstract)
                    
                    uploader_file_list = [file_uploader.name for file_uploader in st.session_state[st.session_state['file_uploader']]]
                    conf = [file_uploader.size for file_uploader in st.session_state[st.session_state['file_uploader']]]
                    hasil_jurnal = zip(hasil_abstract,pdf_list,uploader_file_list,conf)
                    
                    # st.session_state['file_uploader'] = str(randint(1000, 100000000))

                    return hasil_jurnal

        def show_hasil(hasil_jurnal):
            with area_hasil:
                # st.write(hasil_abstract) # debug
                st.write(" ")
                # st.button('download hasil (.csv)')
                hasil_md = """
                <div style="border-width: 1px; border-style: solid; border-color: #3A3B3C; border-radius: 5px; float: left; margin-top: 5px">
                    <div class="column_1" style="background: {color}26; border-right: 1px solid #3A3B3C; padding: 20px; width: 75%; float: left;">
                        <h1 style="font-size:20px">{}</h1>
                        <p>author: {}</p>
                        <p>kata kunci: {}</p>
                        <p>tahun: {}</p>
                        <p>jurnal: {}</p>
                        <div style="background: {color}26; padding: 20px; border: 1px solid #3A3B3C; border-radius: 5px;">
                            <p style="color:{color}; font-size:15px">{}</p>
                        </div>
                    </div>
                    <div class="column_2" style="padding-top: 50px; padding-left: 10px; width: 24%; float: left;">
                        <h2 style="text-align: center; font-size:20px;">Kategori:</h2>
                        <h3 style="color:{textcolor}; text-align: center;">{}</h3>
                    </div>
                </div>"""
            
            # for new reformed hasil_jurnal
            deta_jurnal = []
            count = 0
            # st.write(list(hasil_jurnal))
            for hasil in hasil_jurnal:
                predicted_text = predict(".".join([
                    "%s" % ", ".join(hasil[0]["judul"]).lower().strip(),
                    hasil[0]["abstrak"][0].lower().strip()]))
                
                # if max(predicted_text["decision"][0]) < 0:
                #     predicted_text["label"] = 'Tidak ditemukan'
                
                if hasil[3] in con():
                    predicted_text["label"] = 'Tidak ditemukan'

                jurnal = dict(
                    judul="%s" % ", ".join(hasil[0]["judul"]).lower().strip('.-:()').title(), 
                    author="%s" % ", ".join(hasil[0]["author"]).lower().strip('.-:()'),
                    keyword="%s" % ", ".join(hasil[0]["keyword"]).lower().strip('.-:()'),
                    tahun="%s" % ", ".join(hasil[0]["tahun"]).lower().strip('.-:()'),
                    nama_jurnal="%s" % ", ".join(hasil[0]["nama_jurnal"]).lower().strip('.-:()'),
                    abstrak= '. '.join(map(lambda s: s.strip('.--:() ').capitalize(), hasil[0]["abstrak"][0].split('.--: '))),
                    kategori=predicted_text["label"].lower().strip()
                )
                key = st.session_state['file_uploader']
                file_name = st.session_state[key][count].name
                # file_name = hasil[2]
                if st.session_state['name'] != None :
                    if hasil[3] not in con():
                        checkbox = st.checkbox('(simpan) ' + file_name, key=file_name)
                st.markdown(hasil_md.format(
                            jurnal["judul"],
                            jurnal["author"],
                            jurnal["keyword"],
                            jurnal["tahun"],
                            jurnal["nama_jurnal"],
                            jurnal["abstrak"],
                            jurnal["kategori"],
                            color=text_color, bgcolor=scnd_background_color, textcolor=accent_color), unsafe_allow_html=True)

                # st.write(list(hasil)) # debug
                if hasil[3] not in con():
                    deta_jurnal.append([jurnal, file_name])
                
                with st.expander("lihat pdf"):
                    show_pdf(hasil[1])

                count += 1

            st.session_state['deta_jurnal'] = deta_jurnal

            # for jurnal in deta_jurnal:
            #     insert_jurnal_indo(jurnal)
            
        if len(st.session_state['deta_jurnal']) == 0:
            # uploader_true_list = [file_name for file_name in st.session_state['uploader_file_list'] if st.session_state[file_name] == True]
            if st.button("clear"): 
                st.stop()
                st.experimental_rerun()
            hasil_proses = proses_hasil(files=files)
            
            if st.session_state['name'] != None:
                with st.form(key='form'):
                    show_hasil(hasil_jurnal=hasil_proses)
                    # st.session_state['file_uploader'] = str(randint(1000, 100000000))
                    # st.session_state['uploader_true_list'] = [file_name for jurnal, file_name in st.session_state['deta_jurnal'] if st.session_state[file_name] == True]
                    if st.form_submit_button('OK', 
                            # on_click=simpan_deta, 
                            # args=(uploader_true_list)
                            ):
                        st.write()
            else:
                
                show_hasil(hasil_jurnal=hasil_proses)  
                for jurnal, file_name in st.session_state['deta_jurnal']:
                    insert_jurnal_bucket(jurnal)

                st.session_state['deta_jurnal'] = []
                st.session_state['file_uploader'] = str(randint(1000, 100000000))      
        else: 
            for jurnal, file_name in st.session_state['deta_jurnal']:
                    if st.session_state[file_name] == True and file_name in st.session_state:
                        insert_jurnal_indo(jurnal)
            # st.write(st.session_state['uploader_true_list'])
            st.session_state['deta_jurnal'] = []
            st.session_state['file_uploader'] = str(randint(1000, 100000000))
            st.experimental_rerun()
                # insert_jurnal_indo(jurnal)

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
            show_lottie_json("87012-plane.json")
        
        # st.session_state['deta_jurnal'] = []
        # st.session_state['file_uploader'] = str(randint(1000, 100000000))
    
    # st.write(st.session_state) # debug

if __name__ == "__main__":
    main()