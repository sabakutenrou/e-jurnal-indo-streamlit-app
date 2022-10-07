import streamlit as st
from deta import Deta

DETA_KEY = "c0fv9k2i_T8ujMgk2Xv5yiMhBrhYVfLzfNEuZLUYd"

# 2) initialize with a project key
deta = Deta(DETA_KEY)

# 3) create and use as many DBs as you want!
users = deta.Base("jurnal")

users.insert({
    "name": "Geordi klo",
    "title": "Chief Engineer",
    "desc": "desc"
})

fetch_res = users.fetch({"name": "Geordi"})

for item in fetch_res.items:
    users.delete(item["key"])