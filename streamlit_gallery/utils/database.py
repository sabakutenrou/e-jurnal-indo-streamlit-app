import streamlit as st
from deta import Deta

from dotenv import load_dotenv
import os

load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

# 2) initialize with a project key
deta = Deta(DETA_KEY)

# 3) create and use as many DBs as you want!
db = deta.Base("jurnal")

def insert_user(username, name, password):
    """returns the user of the creation"""
    return db.put({"key": username, "name": name, "password": password})

def fetch_all_users():
    """returns a dict of all users"""
    res = db.fetch()
    return res.items

def get_user(username):
    """returns user dict, if no returns None"""
    return db.get(username)

def update_user(username, updates):
    """returns None if updated, if not raised exception"""
    return db.update(updates, username)

def delete_user(username):
    """always returns None (even if the key doesn't exist)"""
    return db.delete(username)

# older codes

# users.insert({
#     "name": "Geordi klo",
#     "title": "Chief Engineer",
#     "desc": "desc"
# })

# fetch_res = users.fetch({"name": "Geordi"})

# for item in fetch_res.items:
#     users.delete(item["key"])