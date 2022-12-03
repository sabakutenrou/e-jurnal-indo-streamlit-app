import streamlit as st

from streamlit_gallery import apps, data_apps
from streamlit_gallery.utils.page import page_group

import pandas as pd
from PIL import Image

from streamlit_gallery.utils.model import load_data
import streamlit_authenticator as stauth


st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
# st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

usernames = {"usernames" : 
                {"dianas" : 
                    {"email" : "jsmith@gmail.com",
                    "name" : "John Smith",
                    "password" : "$2b$12$EXEEdYF63IxFG/VAa4xMwO88d4hQZdpdzo8767cNtOqHI8koYpa8K"
                    },
                    
                }
            }

# hashed_password = stauth.Hasher(["123"]).generate()
# st.write(hashed_password)

authenticator = stauth.Authenticate(
                usernames,
                "cookie",
                "signature",
                30
            )    

def main():
    image = Image.open('e-jurnal_logo.png')
    page = page_group("p")
    
    if 'df' not in st.session_state:
        df = load_data()
        st.session_state.df = df

    with st.sidebar:
        st.image(image)
        if st.session_state["authentication_status"]:
            user = "Welcome, " + st.session_state["name"] + " - (profil)"
        else: user = "Login"
        with st.expander(user, False):
            if st.session_state["authentication_status"]:
                st.button('edit profile')
                authenticator.logout('Logout', 'main')
            else:
                st.session_state["name"], st.session_state["authentication_status"], st.session_state["username"] = authenticator.login('Login',
                # "main"
                )
                if st.session_state["authentication_status"] == False:
                    st.error('Username/password is incorrect')
                elif st.session_state["authentication_status"] == None:
                    st.write()

        # this dummy
        import csv
        csvFilePath = r'dataset.csv'

        data = []
     
        with open(csvFilePath, encoding='utf-8-sig') as csvf:
            csvReader = csv.DictReader(csvf)
            
            for rows in csvReader:
                
                # key = rows["judul-jurnal"]
                # data[key] = rows
                data.append(rows)
                # data.append(list(rows.values()))

        # judul, nama, abstrak, kategori = zip(*data)
        # kategori = list(kategori)
        # st.write(type(kategori))
        
        step = 25
        output = [data[i:i + step] for i in range(0, len(data), step)]
        
        
        from streamlit_gallery.utils import database
        
        #FUNGSI BATCH DETA
        # st.write(len(data)) # debug
        # import time
        # for trip in output:
        #     database.data_set.put_many(trip)
        #     time.sleep(5)



        import json
        jsonFilePath = r'dataset.json'

        with open(jsonFilePath, 'w', encoding='utf-8-sig') as jsonf:
            jsonf.write(json.dumps(output, indent=4))

        
        # database.data_set.put_many(data)

        # xyz = database.insert_user("usern", "usr", "test")


        # end dummy

        def filter_csv():
            from streamlit_gallery.utils.lang_detection import language_detection
            lines = list()
            # memberName = input("Please enter a member's name to be deleted.")
            with open('dataset.csv', 'r') as readFile:
                reader = csv.reader(readFile)
                for row in reader:
                    lines.append(row)
                    for field in row:
                        if language_detection(field) != 'id':
                            lines.remove(row)
            with open('mycsv.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(lines)
        
        # filter_csv()

        with st.expander("✨ KLASIFIKASI", True):
            page.item("Home", apps.home, default=True)
            page.item("Input Jurnal", apps.jurnal_input)
            page.item("Data Jurnal", apps.jurnal_data)

        with st.expander("🧩 DATA", True):
            page.item("Data klasifikasi", data_apps.data)
            page.item("Model klasifikasi", data_apps.model)

        with st.expander("⭐ LAINNYA", False):
            st.write('developed by -Dian Mahesa-')
            st.write("NPM - 065116239")
            st.write("ILKOM UNPAK")
        
    page.show()

if __name__ == "__main__":
    
    main()
