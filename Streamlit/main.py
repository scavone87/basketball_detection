import streamlit as st
import pre_elaborazione
import homography


PAGES = {
    "Pre Elaborazione": pre_elaborazione,
    "Omografie": homography
}
st.sidebar.title('Navigazione')
selection = st.sidebar.radio("vai", list(PAGES.keys()))
page = PAGES[selection]
page.app()
