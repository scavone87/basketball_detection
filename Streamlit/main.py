import streamlit as st
import homography
import stitching


PAGES = {
    "Stitching": stitching,
    "Omografia": homography
}
st.sidebar.title('Navigazione')
selection = st.sidebar.radio("vai", list(PAGES.keys()))
page = PAGES[selection]
page.app()
