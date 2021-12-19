import streamlit as st
from giwebapp.assets import get_home_image, get_favicon_image


def render():
    col1, col2 = st.columns(2)

    with col1:
        st.image(get_favicon_image())
        st.title('Welcome to GAL-iD webapp !')
        st.markdown('Use the power of AI to detect the shape of galaxies !')
        st.markdown("## Navigation")
        st.markdown("Select a page through the navigation select box on the left sidebar.")
        st.markdown("### Available pages")
        st.markdown("- **Preprocessing**: Show/Create a Galaxy10 DECals Dataset")
        st.markdown("- **Training**: Create a CNN model and train it")
        st.markdown("- **Evaluation**: Evaluate a trained model and show its info")
        st.markdown("- **Prediction**: Select an image of a galaxy and predict its shape")

    with col2:
        st.image(get_home_image())
