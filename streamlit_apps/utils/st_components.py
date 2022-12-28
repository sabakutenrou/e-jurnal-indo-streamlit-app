import streamlit as st

def get_theme_colors():
    return {'primaryColor': st.get_option('theme.primaryColor'), 
    'backgroundColor': st.get_option('theme.backgroundColor'),
    'secondaryBackgroundColor': st.get_option('theme.secondaryBackgroundColor'),
    'textColor': st.get_option('theme.textColor') }
    
def get_accent_color():
    return st.get_option('theme.primaryColor')

def get_scndry_color():
    return st.get_option('theme.textColor')

def st_header(text, font_size=30, color=get_accent_color()):
    text_style = ' style="color:{color}; font-size:{font_size}px"'.format(color=color, font_size=font_size)
    st.markdown("<h1{}>{}</h1>".format(text_style, text), unsafe_allow_html=True)

def st_title(text:str, color:str=get_accent_color(), font_size:int=40, margin_top:int=-20, margin_bottom:int=0):
        st.markdown('<div style="color:{}; font-size: {}px; font-weight:600; margin-top:{}px; margin-bottom:{}px">{}</div>'
            .format(color, font_size, margin_top, margin_bottom, text), unsafe_allow_html=True)