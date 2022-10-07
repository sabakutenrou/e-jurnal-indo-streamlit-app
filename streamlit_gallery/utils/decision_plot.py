import streamlit as st
import plotly.graph_objects as go
from streamlit_gallery.utils.labels import get_labels

def plot_radar(decision):
    labels_list = list(get_labels().values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=decision,
        theta=labels_list,
        fill='toself'
        )
    )
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(
            visible=True
            ),
        ),
    showlegend=False
    )
    # get color
    color = st.get_option('theme.primaryColor')
    fig.update_traces(marker_color=color)   

    st.plotly_chart(fig,use_container_width=True,key="plotly_chart")