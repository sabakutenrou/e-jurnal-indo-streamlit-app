import streamlit as st
import plotly.graph_objects as go
from streamlit_gallery.utils.labels import get_labels

def plot_radar(decision):
    labels_list = list(get_labels().values())
    
    layout = go.Layout(
        autosize=False,
        width=1000,
        height=300,

        xaxis= go.layout.XAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True),

        yaxis= go.layout.YAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=25,
            t=25,
            pad = 4
        )
    )

    fig = go.Figure(data=go.Scatterpolar(
        r=decision,
        theta=labels_list,
        fill='toself'
        ),
        layout=layout
    )
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(
            visible=True
            )
        ),
    showlegend=False
    )
    # get color
    color = st.get_option('theme.primaryColor')
    fig.update_traces(marker_color=color)   

    st.plotly_chart(fig,use_container_width=True,key="plotly_chart")