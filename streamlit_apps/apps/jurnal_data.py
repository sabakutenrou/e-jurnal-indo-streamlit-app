from contextlib import suppress
from turtle import width
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_option_menu import option_menu

import pandas as pd
import plotly.express as px
from streamlit_apps.utils.database import fetch_jurnal_indo, fetch_jurnal_bucket, update_jurnal_indo, update_jurnal_bucket, delete_jurnal_indo, delete_jurnal_bucket, insert_jurnal_indo
from streamlit_apps.utils.st_components import get_theme_colors, st_header

def main():
    if 'data_choice' not in st.session_state:
        st.session_state['data_choice'] = 'Data sistem'

    if 'fetched_jurnal' not in st.session_state:
        st.session_state['fetched_jurnal'] = dict(jurnal_indo=fetch_jurnal_indo(), jurnal_bucket=fetch_jurnal_bucket())
    
    if st.session_state['data_choice'] == 'Data sistem':
        data = st.session_state['fetched_jurnal']['jurnal_indo']
    elif st.session_state['data_choice'] == 'Data input user':
        data = st.session_state['fetched_jurnal']['jurnal_bucket']

    df = pd.DataFrame.from_dict(data)
    df = df[['key','abstrak','author','tahun','kategori','keyword','nama_jurnal']]

    df.rename(columns = {'key':'judul'}, inplace = True) # rename back to key again in the end!
    freq = df['kategori'].value_counts()
    total = len(df.index)

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
        from streamlit_apps.utils.labels import get_labels

        data_df = {'labels': get_labels().values(), 'values': freq}
        data_df = pd.DataFrame(data_df)
        
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
                    height=150,
                    # title='<b><i>Persentase :</i></b>'
                    )

        fig.data[0].textfont.color = 'white'
        fig.update_traces(showlegend=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            title_font_color=accent_color,
            # paper_bgcolor="LightSteelBlue",
        )

        st.write("jumlah data:")
        st.info(str(val) + " jurnal")
        st.write("persentase:")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1: st_header('Data Jurnal')
    with col2:
        data_choice = ['Data input user', 'Data sistem']
        radio = st.radio(label = 'data_choice', options = data_choice, index=1, key='data_choice', label_visibility="hidden")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row-reverse;}</style>', unsafe_allow_html=True)
    
    if radio == 'Data sistem':
        st.info('Data jurnal berbahasa indonesia dalam database aplikasi E-Jurnal Indonesia')
    if radio == 'Data input user':
        st.warning('Data jurnal berbahasa indonesia riwayat input pengguna aplikasi E-Jurnal Indonesia')
    
    col1, col2, col3 = st.columns([3, 0.1, 1])

    # df_all = df
    # if selected == 'semua dokumen':
    #     df = df_all
    # elif selected == 'hardware programming':
    #     df = df.loc[df['kategori'] == 'hardware programming']
    # elif selected == 'pengolahan citra':
    #     df = df.loc[df['kategori'] == 'pengolahan citra']
    # elif selected == 'jaringan':
    #     df = df.loc[df['kategori'] == 'jaringan']
    # elif selected == 'data mining':
    #     df = df.loc[df['kategori'] == 'data mining']
    # elif selected == 'multimedia':
    #     df = df.loc[df['kategori'] == 'multimedia']

    if selected == 'semua dokumen':
        selected = ''
    
    data = ''
    
    defaultColDef = {
        "filter": True,
        "resizable": True,
        "sortable": True,
        "floatingFilter": True
    }

    jscode = """
        function onFirstDataRendered(parmas) {{
            var filterComponent = parmas.api.getFilterInstance('kategori');
            filterComponent.setModel({
                type: 'contains',
                filter: '%s',
                filterTo: null,
            });

            parmas.api.onFilterChanged();
        }}
        """ % selected

    onFirstDataRendered = JsCode(jscode)
    
    options = {
        # "rowSelection": "multiple",
        # "rowMultiSelectWithClick": True,
        # "sideBar": ["columns", 'filters'],
        # "enableRangeSelection": True,
        "onFirstDataRendered": onFirstDataRendered
    }


    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column('judul', maxWidth=250) # key
    gb.configure_column('abstrak', maxWidth=500)
    gb.configure_column('kategori', maxWidth=150)
    gb.configure_default_column(**defaultColDef)
    gb.configure_grid_options(**options)
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
        # fit_columns_on_grid_load=True,
        # try_to_convert_back_to_original_types=True,
        enable_enterprise_modules=False,
        # custom_css=custom_css,
        # width='500%'
        )

    data_length = len(data['selected_rows'])
    # st.write(data['selected_rows']) #debug

    if data_length != 0:
        with col1:
            selected_data = st.container()
        with col3:
            from streamlit_apps.utils.labels import get_labels

            labels = list(get_labels().values())
            data['selected_rows'][0]['kategori'] = st.selectbox('kategori', labels, index=labels.index(data['selected_rows'][0]['kategori']))
            data['selected_rows'][0]['author'] = st.text_area('author', data['selected_rows'][0]['author'])
            data['selected_rows'][0]['nama_jurnal'] = st.text_input('nama jurnal', data['selected_rows'][0]['nama_jurnal'])
            data['selected_rows'][0]['keyword'] = st.text_input('kata kunci', data['selected_rows'][0]['keyword'])
            years = [""] + list(range(2010, 2022))
            tahun = "" if data['selected_rows'][0]['tahun'] == "" else int(data['selected_rows'][0]['tahun'])
            data['selected_rows'][0]['tahun'] = st.selectbox('Year', years, index=years.index(tahun))
            judul = selected_data.text_area('judul',value=str(data['selected_rows'][0]['judul']),disabled=False)
            data['selected_rows'][0]['abstrak'] = selected_data.text_area('abstrak',value=str(data['selected_rows'][0]['abstrak']),disabled=False, height=250)

            if st.session_state['name'] != None:
                # selected_data.write(data['selected_rows'][0])
                if st.session_state['data_choice'] == 'Data sistem':
                    if selected_data.button('simpan'):
                        if judul != data['selected_rows'][0]['judul']:
                            insert_jurnal_indo({
                                "judul":judul,
                                "abstrak":data['selected_rows'][0]['abstrak'],
                                "author":data['selected_rows'][0]['author'],
                                "keyword":data['selected_rows'][0]['keyword'],
                                "tahun":data['selected_rows'][0]['tahun'],
                                "nama_jurnal":data['selected_rows'][0]['nama_jurnal'],
                                "kategori":data['selected_rows'][0]['kategori']
                            })
                            delete_jurnal_indo(data['selected_rows'][0]['judul'])

                        else:
                            update_jurnal_indo(data['selected_rows'][0], judul)
                        del st.session_state['fetched_jurnal']
                        st.experimental_rerun()
                    if selected_data.button('hapus'):
                        delete_jurnal_indo(judul)
                        del st.session_state['fetched_jurnal']
                        st.experimental_rerun()
                elif st.session_state['data_choice'] == 'Data input user':
                    if selected_data.button('pindah'):
                        # update_jurnal_bucket(data['selected_rows'][0], judul)
                        insert_jurnal_indo(data['selected_rows'][0])
                        del st.session_state['fetched_jurnal']
                        st.experimental_rerun()
                    if selected_data.button('hapus'):
                        delete_jurnal_bucket(judul)
                        del st.session_state['fetched_jurnal']
                        st.experimental_rerun()
    # st.write(st.session_state) # debug

if __name__ == "__main__":
    main()