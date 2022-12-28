import streamlit as st

from streamlit_apps import apps, data_apps
from streamlit_apps.utils.page import page_group

from PIL import Image

import streamlit_authenticator as stauth

from streamlit_apps.utils.database import fetch_all_users

st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
# st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# usernames = {"usernames" : 
#                 {"dianas" : 
#                     {"email" : "jsmith@gmail.com",
#                     "name" : "Dianas",
#                     "password" : "$2b$12$EXEEdYF63IxFG/VAa4xMwO88d4hQZdpdzo8767cNtOqHI8koYpa8K"
#                     },
#                 # "dianas" : 
#                 #     {"email" : "jsmith@gmail.com",
#                 #     "name" : "Dianas",
#                 #     "password" : "$2b$12$EXEEdYF63IxFG/VAa4xMwO88d4hQZdpdzo8767cNtOqHI8koYpa8K"
#                 #     },
#                 }
#             }

# hashed_password = stauth.Hasher(["123"]).generate()
# st.write(hashed_password)

# hashed_password = stauth.Hasher(["admin"]).generate()
# insert_user("admin", "Dian Mahesa", "dianmahes@gmail.com", hashed_password[0])

usernames = fetch_all_users()

authenticator = stauth.Authenticate(
                usernames,
                "cookie",
                "signature",
                30
            )    

def main():
    image = Image.open('e-jurnal_logo.png')
    page = page_group("p")

    with st.sidebar:
        st.image(image)
        if st.session_state["authentication_status"]:
            user = "Welcome, " + st.session_state["name"]
        else: user = "Login"
        with st.expander(user, False):
            if st.session_state["authentication_status"]:
                # st.button('edit profile')
                authenticator.logout('Logout', 'main')
            else:
                st.session_state["name"], st.session_state["authentication_status"], st.session_state["username"] = authenticator.login('Login',
                # "main"
                )
                if st.session_state["authentication_status"] == False:
                    st.error('Username/password is incorrect')
                elif st.session_state["authentication_status"] == None:
                    st.write()

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

        # st.write(st.session_state) # debug
        
    page.show()

if __name__ == "__main__":
    
    main()
