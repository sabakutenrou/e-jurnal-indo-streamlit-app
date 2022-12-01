import streamlit as st
from deta import Deta

from dotenv import load_dotenv
import os

import streamlit as st

load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

# 2) initialize with a project key
deta = Deta(DETA_KEY)

# 3) create and use as many DBs as you want!
admin = deta.Base("admin")
stopword = deta.Base("stopword")
kata_dasar = deta.Base("kata_dasar")
jurnal_indo = deta.Base("jurnal_indo")
jurnal_bucket = deta.Base("jurnal_bucket")
data_set = deta.Base("data_set")

def fetch_stopwords():
    res = stopword.fetch()
    return res.items

def fetch_kata_dasar():
    res = kata_dasar.fetch()
    return res.items

def fetch_data_set():
    res = data_set.fetch()
    return res.items

def fetch_jurnal_indo():
    res = jurnal_indo.fetch()
    return res.items

def fetch_jurnal_bucket():
    res = jurnal_bucket.fetch()
    return res.items

def insert_jurnal_indo(username, name, password):
    return jurnal_indo.put({"key": username, "name": name, "password": password})

def insert_jurnal_bucket(username, name, password):
    return jurnal_bucket.put({"key": username, "name": name, "password": password})

def update_jurnal_indo(username, updates):
    """returns None if updated, if not raised exception"""
    return admin.update(updates, username)

def delete_jurnal_indo(username):
    """always returns None (even if the key doesn't exist)"""
    return admin.delete(username)

def update_jurnal_bucket(username, updates):
    """returns None if updated, if not raised exception"""
    return admin.update(updates, username)

def delete_jurnal_bucket(username):
    """always returns None (even if the key doesn't exist)"""
    return admin.delete(username)

def insert_user(username, name, password):
    """returns the user of the creation"""
    return admin.put([{"key": username, "name": name, "password": password},
                    {"key": "username", "name": "name", "password": password}])

def fetch_all_users():
    """returns a dict of all users"""
    res = admin.fetch()
    return res.items

def get_user(username):
    """returns user dict, if no returns None"""
    return admin.get(username)

def update_user(username, updates):
    """returns None if updated, if not raised exception"""
    return admin.update(updates, username)

def delete_user(username):
    """always returns None (even if the key doesn't exist)"""
    return admin.delete(username)

# older codes

# users.insert({
#     "name": "Geordi klo",
#     "title": "Chief Engineer",
#     "desc": "desc"
# })

# fetch_res = users.fetch({"name": "Geordi"})

# for item in fetch_res.items:
#     users.delete(item["key"])