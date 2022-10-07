from email import charset
import streamlit as st

from pathlib import Path

import pandas as pd

from streamlit_option_menu import option_menu
import datetime

from streamlit_gallery.utils.model import predict
from streamlit_gallery.utils.decision_plot import plot_radar

from streamlit_gallery.utils.lottie import lottie_show

import streamlit_gallery.utils.database as db

def main():
    df = st.session_state.df
    df.to_csv(r'.\my_data.csv', index=False)
    # df.to_sql('dataset', con=conn, if_exists='replace', index = False)
    # df = pd.read_sql('SELECT * FROM e_jurnal_db.dataset', con=conn)
    # st.dataframe(df)

    col1, col2  = st.columns(2)
    accent_color = st.get_option('theme.primaryColor')

    container_style = ' style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;"'.format(color=accent_color)
    
    with col1:
        text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=accent_color, font_size=30)
        st.markdown("<h1{}>Klasifikasi cepat 📑</h1>".format(text_style), unsafe_allow_html=True)
        
        if not 'checkbox' in st.session_state or not st.session_state['checkbox']:
            st.info("""Tool digunakan untuk melihat hasil klasifikasi atau evaluasi cepat pada teks sampel""")
        checkbox = st.checkbox("judul dan abstrak", key='checkbox')
        judul = ''
        if checkbox:
            judul = st.text_input('judul jurnal',help="input judul (opsional)", placeholder="masukkan judul jurnal")
        abstrak = st.text_area('abstrak jurnal', height=200, key='abstrak', help="masukkan atau copy abstrak jurnal disini..", placeholder='masukkan teks abstrak disini')
        st.button(label='Klasifikasi')

    with col2:
        if len(judul) == 0 and len(abstrak) == 0:
            text1 = 'Selamat datang di,'
            text2 = 'Aplikasi Klasifikasi E-Jurnal berbahasa Indonesia 🎉'
            text3 = """Aplikasi klasifikasi jurnal melakukan klafikasi pada dokumen 
                    berbahasa indonesia. Dokumen input merupakan dokumen elektronik 
                    dengan format pdf dan diklasifikasikan menggunakan beberapa tahapan 
                    pemrosesan teks dan metode analisis SVM yang menghasilkan kelas prediksi dokumen"""
            ht_welcome = """
                <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                    <p style="color:{color}; font-size:25px">{}<br></p>
                    <strong style="color:{color}; font-size:28px">{}<br><br></strong>
                    <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                        <p style="color:{color}; font-size:15px">{}</p>
                    </div>
                </div>"""
            st.markdown(ht_welcome.format(text1, text2, text3, color=accent_color), unsafe_allow_html=True)
        elif len(judul) > 0 and len(abstrak) == 0:
            st.write('null')
        else:
            # in range(100,500)
            teks_length = len(abstrak.split())
            # if teks_length >= 100 and teks_length <= 500:
            if teks_length < 100 or teks_length > 500:
                container = st.empty()
                with container:
                    if not st.button('Teks yang diinputkan dideteksi bukan abstrak. Tetap ingin melanjutkan? (ya)'):
                        return
                container.empty()
            
            # def prediksi():
            #     st.session_state['hasil'] = 

            selected = option_menu('hasil', ["kelas", "preprocess", "tfidf"], 
            icons=['house', 'cloud-upload', "list-task", 'gear'], 
            menu_icon="cast", default_index=0, orientation="horizontal",
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "0px"},
                "nav-link": {"font-size": "15px", "text-align": "center", "margin":"2px"},
                "nav-link-selected": {"font-size": "12px"}
                }
            )
            if selected == 'kelas':
                st.success('Prediksi : Data Mining')
                # elif "text" in st.session_state:
                predicted_text = predict(st.session_state['abstrak'])
                plot_radar(predicted_text["decision"][0])
            
            st.write(predicted_text)

if __name__ == "__main__":
    main()
