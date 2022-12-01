from contextlib import suppress
from turtle import width
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_option_menu import option_menu

from streamlit_gallery.utils.streamlit_prop import get_theme_colors

import pandas as pd
import plotly.express as px

def main():

    df = st.session_state.df
    freq = df['kategori'].value_counts()
    total = len(df.index)
    # st.write(total)

    colors = get_theme_colors()
    accent_color = colors['primaryColor']
    background_color = colors['backgroundColor']
    scnd_background_color = colors['secondaryBackgroundColor']
    
    col1, col2, col3= st.columns([1,1.5,0.5])
    with col1:
        selected = option_menu(None, ["semua dokumen","---", "hardware programming", "data mining", 'jaringan', 'pengolahan citra', 'multimedia'], 
            icons=['border-all', '', 'cpu', 'clipboard', 'hdd-network', 'file-earmark-image-fill', 'camera-video'], 
            # icons=None, # menu_icon="cast", # orientation="horizontal",
            default_index=0, 
            styles={
                "container": {"padding": "0"},
                "icon": {"font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"2px"},
                "nav-link-selected": {"font-size": "12px"}
            }
        )

    with col2:
        ht_welcome = """
            <div>
                <h1 style="text-align: center; color:{color}; font-size:24px">{}</h1>
            </div>"""

                # <strong style="color:{color}; font-size:28px">{}<br><br></strong>
                # <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                #     <p style="color:{color}; font-size:15px">{}</p>
                # </div>

        from streamlit_gallery.utils.labels import get_labels

        # st.markdown(ht_welcome.format(selected, color=accent_color), unsafe_allow_html=True)
        data_df = {'labels': get_labels().values(), 'values': freq}
        
        data_df = pd.DataFrame(data_df)
        # st.write(get_labels().values())
        
        default_color = background_color
        colors = {selected: accent_color}

        if selected == "semua dokumen":
            default_color = accent_color

        color_discrete_map = {
            c: colors.get(c, default_color) 
            for c in data_df.labels}
        
        fig = px.bar(data_df, x='labels', y='values', color='labels',
                    color_discrete_map=color_discrete_map,
                    width=500, height=320,
                    title='<b>' + selected.upper() + '</b>'
                    )
        fig.update_xaxes(title= '', visible=True, showticklabels=False)
        fig.update_yaxes(title= 'total data', visible=True, showticklabels=True)
        fig.update_traces(showlegend=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=60, b=0),
            title_font_color=accent_color,
            # paper_bgcolor="LightSteelBlue",
            # title_font_family="Times New Roman",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        val = 0
        if selected == "semua dokumen":
            val = total
        else: 
            val = data_df.loc[data_df.labels==selected, 'values'].values[0]
        progress = val/total * 100
        prog_df = pd.DataFrame({'names' : ['progress','-'],
                        'values' :  [progress, 100 - progress]})
                        
        # plotly
        colowr = {'progress': accent_color,'-': scnd_background_color}
        
        fig = px.pie(prog_df, values ='values', names = 'names', hole = 0.6,
                    # color_discrete_sequence = [accent_color, scnd_background_color],
                    color='names',
                    color_discrete_map = colowr,
                    height=200,
                    # title='<b><i>Persentase :</i></b>'
                    )

        fig.data[0].textfont.color = 'white'
        fig.update_traces(showlegend=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            title_font_color=accent_color,
            # paper_bgcolor="LightSteelBlue",
        )
        # fig.show()
        st.info('jumlah data: ' + str(val))
        st.write("persentase:")
        st.plotly_chart(fig, use_container_width=True)

        # f = fig.full_figure_for_development(warn=False)
        # fig.show()
        
    st.subheader('Data Jurnal')
    col1, col2, col3 = st.columns([3, 0.1, 1])
    
    data = ''

    # Test data is copy from www.ag-grid.com
    # In order to minimize the coding time, please allow me to show it in such a weird format.
    
    defaultColDef = {
        "filter": True,
        "resizable": True,
        "sortable": True,
        "floatingFilter": True
    }

    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column('judul-jurnal', maxWidth=250)
    gb.configure_column('abstrak-jurnal', maxWidth=500)
    gb.configure_column('kategori', maxWidth=150)
    gb.configure_default_column(**defaultColDef)
    # gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_grid_options(rowHeight=50, floatingFilter=True, pagination=True)
    gb.configure_selection(selection_mode="single")
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()
    grid_options = gb.build()
    data = AgGrid(df, 
        gridOptions=grid_options,
        theme='streamlit', # streamlit, alpine, balham, material
        # enable_enterprise_modules=True, 
        allow_unsafe_jscode=True, 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        # update_mode=GridUpdateMode.VALUE_CHANGED,
        height=800,
        fit_columns_on_grid_load=True,
        # try_to_convert_back_to_original_types=True,
        enable_enterprise_modules=False,
        # custom_css=custom_css,
        # width='500%'
        )

    data_length = len(data['selected_rows'])

    if data_length != 0:
        with col1:
            selected_data = st.container()
        with col3:
            # atau html
            # st.markdown('---')
            st.success("kategori: KAMEHAMEHA")
            st.info("author: dianasAC")
            # st.write("jurnal: Teknologi")
            # st.write("kata kunci: keywords")
            # st.write("tahun: 100 SM")
            welcome_card_md = """
                <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                    <p style="color:{color}; font-size:15px">jurnal: {}<br></p>
                    <p style="color:{color}; font-size:15px">kata kunci: {}<br></p>
                    <p style="color:{color}; font-size:15px">tahun: {}</p>
                </div>"""
            st.markdown(welcome_card_md.format("Teknologi", "keywords", "100 SM", color=accent_color), unsafe_allow_html=True)

            st.markdown('---')
            simpan = st.button('simpan')
            hapus = st.button('hapus')
            selected_data.text_area('judul',value=str(data['selected_rows'][0]['judul-jurnal']),disabled=False)
            # selected_data.text(data['selected_rows'][0]['abstrak-jurnal'])
            selected_data.text_area('abstrak',value=str(data['selected_rows'][0]['abstrak-jurnal']),disabled=False, height=250)

if __name__ == "__main__":
    # st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()