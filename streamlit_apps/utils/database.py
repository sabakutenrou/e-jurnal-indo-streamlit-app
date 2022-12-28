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

def insert_jurnal_indo(data):
    # return jurnal_indo.put({"key": username, "name": name, "password": password})
    return jurnal_indo.put({
                            "key":data["judul"],
                            "abstrak":data["abstrak"],
                            "author":data["author"],
                            "keyword":data["keyword"],
                            "tahun":data["tahun"],
                            "nama_jurnal":data["nama_jurnal"],
                            "kategori":data["kategori"]
                            })

# def insert_jurnal_bucket(username, name, password):
    # return jurnal_bucket.put({"key": username, "name": name, "password": password})

def insert_jurnal_bucket(data):
    return jurnal_bucket.put({
                            "key":data["judul"],
                            "abstrak":data["abstrak"],
                            "author":data["author"],
                            "keyword":data["keyword"],
                            "tahun":data["tahun"],
                            "nama_jurnal":data["nama_jurnal"],
                            "kategori":data["kategori"]
                            })

# def update_jurnal_indo(username, updates):
#     """returns None if updated, if not raised exception"""
#     return admin.update(updates, username)

def update_jurnal_indo(data, judul):
    """returns None if updated, if not raised exception"""
    return jurnal_indo.update({
                        # "key":judul,
                        "abstrak":data["abstrak"],
                        "author":data["author"],
                        "keyword":data["keyword"],
                        "tahun":data["tahun"],
                        "nama_jurnal":data["nama_jurnal"],
                        "kategori":data["kategori"]
                        }, data["judul"])

def update_jurnal_bucket(data, judul):
    return jurnal_bucket.update({
                        # "key":judul,
                        "abstrak":data["abstrak"],
                        "author":data["author"],
                        "keyword":data["keyword"],
                        "tahun":data["tahun"],
                        "nama_jurnal":data["nama_jurnal"],
                        "kategori":data["kategori"]
                        }, data["judul"])

def delete_jurnal_indo(judul):
    """always returns None (even if the key doesn't exist)"""
    return jurnal_indo.delete(judul)


def delete_jurnal_bucket(judul):
    """always returns None (even if the key doesn't exist)"""
    return jurnal_bucket.delete(judul)

def insert_user(username, name, email, password):
    """returns the user of the creation"""
    return admin.put({"key": username, "name": name, "email":email, "password": password})

def fetch_all_users():
    """returns a dict of all users"""
    res = admin.fetch()
    res = res.items
    user_dict = dict()
    for user in res:
        user_dict[user["key"]] = {
                "email" : user["email"],
                "name" : user["name"],
                "password" : user["password"]
            }
    return { "usernames": user_dict }


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