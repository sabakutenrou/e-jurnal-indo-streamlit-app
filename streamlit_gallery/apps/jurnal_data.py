from contextlib import suppress
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from streamlit_option_menu import option_menu

def main():
    # reload data
    col1, col2= st.columns([1,2])
    with col1:
        selected2 = option_menu(None, ["semua","---", "hardware programming", "data mining", 'jaringan', 'pengolahan citra', 'multimedia'], 
            icons=['border-all', '', 'cpu', 'clipboard', 'hdd-network', 'file-earmark-image-fill', 'camera-video'], 
            # icons=None, 
            # menu_icon="cast", 
            default_index=0, 
            # orientation="horizontal",
            styles={
                "container": {"padding": "0"},
                "icon": {"font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"2px"},
                "nav-link-selected": {"font-size": "12px"}
            }
        )
        selected2
    with col2:
        ht_welcome = """
            <div>
                <p style="color:{color}; font-size:25px">{}<br></p>
                <strong style="color:{color}; font-size:28px">{}<br><br></strong>
                <div style="background: {color}26; padding: 20px; border: 1px solid {color}33; border-radius: 5px;">
                    <p style="color:{color}; font-size:15px">{}</p>
                </div>
            </div>"""
        
        
        # st.markdown(ht_welcome.format(text1, text2, text3, color=accent_color), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        selected_data = st.container()
    with col2:
        st.header('Image')
        st.button('edit')
        st.button('simpan')
        st.button('batal')
        st.button('delete')
    data = ''

    st.write('aggrid pencarian')
    df = st.session_state.df
    gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_grid_options(rowHeight=50)
    # gb.configure_auto_height(autoHeight=True)
    gb.configure_column('judul-jurnal',maxWidth=250)
    gb.configure_column('abstrak-jurnal', maxWidth=500)
    gb.configure_default_column(editable=True, groupable=True)
    # gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_selection(selection_mode="single")
    gb.configure_pagination(enabled=True, paginationPageSize=2000)
    gb.configure_grid_options(pagination=True)
    grid_options = gb.build()
    data = AgGrid(df, 
        gridOptions=grid_options, 
        theme='streamlit',
        # enable_enterprise_modules=True, 
        allow_unsafe_jscode=True, 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=500)

    if data=='':
        selected_data.write(data)
    else :
        selected_data.write(data['selected_rows'][0]['judul-jurnal'])
        selected_data.text_area('judul',value=str(data['selected_rows'][0]['judul-jurnal']),disabled=True)
        # selected_data.text(data['selected_rows'][0]['abstrak-jurnal'])
        selected_data.text_area('abstrak',value=str(data['selected_rows'][0]['abstrak-jurnal']),disabled=True, height=200)


if __name__ == "__main__":
    st.set_page_config(page_title="E-Jurnal Indonesia", page_icon="🎈", layout="wide")
    main()