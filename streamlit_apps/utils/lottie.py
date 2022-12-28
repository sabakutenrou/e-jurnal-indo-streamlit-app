import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def lottie_show(url: str, key=None):
    lottie = load_lottieurl(url)
    st_lottie(lottie, key)

def lottie_json(path):
    with open(path, "r",errors='ignore') as f:
        data = json.load(f)
    return data

def show_lottie_json(path):
    st_lottie(lottie_json(path))