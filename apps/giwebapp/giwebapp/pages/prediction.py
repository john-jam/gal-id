import streamlit as st
import pandas as pd
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import urlsplit, parse_qsl
from gicommon.models.learning import PredictionIn
from giwebapp.config import CONFIG
from giwebapp.utils import get_item_from_option, get_dl_models_options, format_dl_model_option
from giwebapp.api import ApiClient
from giwebapp.assets import DECALS_LABELS, get_decals_sample_image

api_client = ApiClient(CONFIG.api_url)


def render():
    st.title('Prediction')

    st.markdown('### Select an image')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "You can test your model and do predictions from the [Legacy Survey Viewer](https://www.legacysurvey.org/viewer?layer=ls-dr9) by following the previous steps:")
        st.markdown('- Click the center of the galaxy you want to predict')
        st.markdown('- On the opened popup, click **Link Here**')
        st.markdown('- Copy/Paste the full url from your browser into the form below :point_down:')

        lsv_url = st.text_input('Legacy Survey Viewer url', placeholder='Paste the url here')
        parsed_params = dict(parse_qsl(urlsplit(lsv_url).query))

        if lsv_url == '':
            st.stop()
        if not all(param in parsed_params for param in
                   (CONFIG.lsv_ra_param, CONFIG.lsv_dec_param, CONFIG.lsv_layer_param)):
            st.error('Please enter a valid **Legacy Survey Viewer url**')
            st.stop()

    with col2:
        lsv_zoom = st.slider('Zoom', min_value=12, max_value=15, value=13)
        lsv_ra = st.number_input('Right Ascension', value=float(parsed_params[CONFIG.lsv_ra_param]), format="%.4f")
        lsv_dec = st.number_input('Declination', value=float(parsed_params[CONFIG.lsv_dec_param]), format="%.4f")
        lsv_layer = parsed_params[CONFIG.lsv_layer_param]
        st.markdown(f'**Layer**: {lsv_layer}')

    st.markdown('### Let the model predict')

    col1, col2 = st.columns(2)

    with col1:
        params = {
            CONFIG.lsv_ra_param: lsv_ra,
            CONFIG.lsv_dec_param: lsv_dec,
            CONFIG.lsv_zoom_param: lsv_zoom,
            CONFIG.lsv_layer_param: lsv_layer
        }
        response = requests.get(CONFIG.lsv_cutout_url, params=params)

        st.markdown('#### Original image')
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)

        dl_models = api_client.get_all_dl_models()
        dl_model_option = st.selectbox(
            'Model', options=get_dl_models_options(dl_models, trained_only=True), format_func=format_dl_model_option
        )
        dl_model = get_item_from_option(dl_models, dl_model_option)

        predict_button = None
        if dl_model:
            predict_button = st.button('Predict')

    with col2:
        if predict_button:
            try:
                with st.spinner('Predicting...'):
                    prediction = api_client.post_prediction(dl_model.id, PredictionIn(image=image.tolist()))

                st.markdown('#### Predicted image')
                st.image(get_decals_sample_image(prediction.category))
                st.markdown(f'#### CATEGORY {prediction.category}: {DECALS_LABELS[prediction.category]}')
                df = pd.DataFrame({
                    'Category': DECALS_LABELS.values(),
                    'Probability': prediction.probabilities
                })
                st.dataframe(df)
            except RuntimeError as err:
                st.error(err)
