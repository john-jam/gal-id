import streamlit as st
from giwebapp.config import CONFIG
from giwebapp.assets import get_favicon_image
from giwebapp.pages.home import render as home_page
from giwebapp.pages.preprocessing import render as preprocessing_page
from giwebapp.pages.training import render as training_page
from giwebapp.pages.evaluation import render as evaluation_page
from giwebapp.pages.prediction import render as prediction_page


def main():
    # Page configuration
    st.set_page_config(
        page_title='GAL-iD',
        page_icon=get_favicon_image(),
        layout="wide"
    )

    # Debug session state
    if CONFIG.debug:
        print(st.session_state)

    # Sub pages definition
    pages = {
        'Home': home_page,
        'Preprocessing': preprocessing_page,
        'Training': training_page,
        'Evaluation': evaluation_page,
        'Prediction': prediction_page,
    }

    # Sidebar navigation
    page_id = st.sidebar.selectbox('Go to', pages.keys())

    # Refresh button
    st.sidebar.button('Refresh')

    # Show the selected page
    pages[page_id]()


if __name__ == '__main__':
    main()
