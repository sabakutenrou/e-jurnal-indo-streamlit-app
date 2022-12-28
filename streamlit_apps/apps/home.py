from email import charset
import streamlit as st

from pathlib import Path

import pandas as pd

from streamlit_option_menu import option_menu
import datetime

from streamlit_apps.utils.model import predict
from streamlit_apps.utils.decision_plot import plot_radar

from streamlit_apps.utils.lottie import lottie_show

import streamlit_apps.utils.database as db

from streamlit_apps.utils.lang_detection import language_detection
import re

from streamlit_apps.utils.preprocess import preprocess, remove_stopwordss, stemming

def main():
    def st_header(text, font_size=30, color=st.get_option('theme.primaryColor')):
        text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=color, font_size=font_size)
        st.markdown("<h1{}>{}</h1>".format(text_style, text), unsafe_allow_html=True)

    def get_accent_color():
        return st.get_option('theme.primaryColor')

    accent_color = get_accent_color()
    input_valid = None
    col1, col2  = st.columns(2)
    
    def onchange(key):
        st.session_state[key] = True

    def get_matching_pattern(list):            ## Function name changed
        for item in list:
            match = re.search('id', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match:                              ## Correct the indentation of if condition
                return match                       ## You also don't need an else statement
            match = re.search('de', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match: 
                return match 

    def validasi_kata(text, min, key):
        if not key in st.session_state: st.session_state[key] = None
        if st.session_state[key]:
            text_length = len(text.split())
            if text_length > 0:
                indo_text = language_detection(text)
                # st.write(indo_text) # debug
                indo = get_matching_pattern(indo_text)
                # st.write(indo) # debug
            else: indo = True
            # indo = re.compile("^id")
            if text_length >= min and indo:
                return True
            else:
                # if text_length == 0:
                #     st.error('Error: field kosong')
                if text_length in range(1,min):
                    st.warning("Peringatan: teks terlalu singkat")
                if not indo:
                    st.warning("Peringatan: gunakan bahasa indonesia!")
                return False

    def show_card():
        """buat welcome card"""
        text1 = 'Selamat datang di,'
        text2 = 'Aplikasi Klasifikasi E-Jurnal berbahasa Indonesia 🎉'
        text3 = """Aplikasi klasifikasi jurnal melakukan klafikasi pada dokumen 
                berbahasa indonesia. Dokumen input merupakan dokumen elektronik 
                dengan format pdf dan diklasifikasikan menggunakan beberapa tahapan 
                pemrosesan teks dan metode analisis SVM yang menghasilkan kelas prediksi dokumen"""
        welcome_card_md = """
            <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                <p style="color:{color}; font-size:25px">{}<br></p>
                <strong style="color:{color}; font-size:28px">{}<br><br></strong>
                <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                    <p style="color:{color}; font-size:17px">{}</p>
                </div>
            </div>"""
        st.markdown(welcome_card_md.format(text1, text2, text3, color=accent_color), unsafe_allow_html=True)

    with col1:
        st_header("Klasifikasi cepat 📑")
        judul = ''
        abstrak = ''
        valid = {}
        if not 'classify' in st.session_state: st.session_state['classify'] = False
        if not 'checkbox' in st.session_state or not st.session_state['checkbox']:
            st.info("Tool digunakan untuk melihat hasil klasifikasi atau evaluasi cepat pada teks sampel")
        if st.checkbox("judul dan abstrak", key='checkbox'):
            # judul = st.text_input('judul jurnal',help="input judul (opsional)", placeholder="masukkan judul jurnal", on_change=onchange, key="judul")
            judul = st.text_input('judul jurnal',help="input judul (opsional)", placeholder="masukkan judul jurnal", on_change=onchange, args=["judul"])
            valid['judul'] = validasi_kata(judul, 5, 'judul')
        else:
            valid['judul'] = True
        abstrak = st.text_area("abstrak jurnal", height=200, help="masukkan atau copy abstrak jurnal disini..", placeholder="masukkan teks abstrak disini", on_change=onchange, args=["abstrak"])
        valid["abstrak"] = validasi_kata(abstrak, 10, 'abstrak')
        abstrak = ".".join([judul,abstrak]) if judul != '' else abstrak
        st.session_state['teks_abstrak'] = abstrak
        # st.write(language_detection(abstrak, "multiple")) if len(abstrak) != 0 else st.write() # debug
        classify = st.button(label='Klasifikasi')

    with col2:
        if classify: 
            st.session_state['classify'] = True
        if st.session_state['classify'] and valid["abstrak"] and valid['judul']:
            selected = option_menu('HASIL', ["kelas", "preprocess"], 
            icons=['house', 'cloud-upload', "list-task", 'gear'], 
            # menu_icon="cast", 
            default_index=0, 
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "0px"},
                "nav-link": {"font-size": "15px", "text-align": "center", "margin":"2px"},
                "nav-link-selected": {"font-size": "12px"}
                }
            )
            if selected == 'kelas':
                predicted_text = predict(st.session_state['teks_abstrak'])
                plot_radar(predicted_text["decision"][0])
                st.success('Kelas Prediksi : ' + predicted_text['label'])
                # st.write(predicted_text) # debug
            elif selected == 'preprocess':
                st.write("Teks:")
                cleaned = preprocess(st.session_state['teks_abstrak'])
                with st.expander('hasil hapus karakter spesial'):
                    st.warning(cleaned)
                st.write("Hapus stopwords:")

                stopword_removed = remove_stopwordss(cleaned)
                with st.expander('hasil hapus stopwords'):
                    st.info(stopword_removed)

                st.write("Stemming:")
                stem_wait_field = st.empty()
                stem_wait_field.text('tunggu sebentar... (stemming)')
                stem = stemming(stopword_removed)
                
                    
                from streamlit_apps.utils.preprocess import readkamus, penemuan_kata_dasar
                kamus = readkamus('kata-dasar.txt')
                kalimat = stopword_removed
                listkalimat = [str(i) for i in kalimat.split()]
                kalimatdasar = []
                strkalimatdasar = ''
                for i in listkalimat:
                    i = penemuan_kata_dasar(i,kamus)
                    kalimatdasar.append(i)
                    strkalimatdasar = strkalimatdasar + ' ' + i
                
                with st.expander('hasil stemming'):
                    st.success(stem)
                stem_wait_field.empty()
                
        else: show_card()
    
if __name__ == "__main__":
    main()