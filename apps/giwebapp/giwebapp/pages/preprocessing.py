import streamlit as st
from gicommon.models.preprocessing import DatasetIn
from giwebapp.api import ApiClient
from giwebapp.config import CONFIG
from giwebapp.utils import get_item_from_option, get_dataset_options, format_dataset_option

api_client = ApiClient(CONFIG.api_url)


def render():
    st.title('Preprocessing')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### Show the dataset info')

        datasets = api_client.get_all_datasets()
        dataset_option = st.selectbox('Dataset', options=get_dataset_options(datasets),
                                      format_func=format_dataset_option)
        dataset = get_item_from_option(datasets, dataset_option)

        if dataset:
            st.json(dataset.json())

    with col2:
        st.markdown('### Create a new dataset')

        with st.form('dataset_form'):
            test_size = st.slider('Test size', min_value=0.01, max_value=0.99, value=0.2, step=0.01, format="%.2f")
            random_state = st.number_input('Random state', min_value=0, value=42)
            image_size = st.slider('Image size', min_value=16, max_value=256, value=256)
            compressed = st.checkbox('Compressed')

            create_button = st.form_submit_button('Create Dataset')
        if create_button:
            try:
                dataset = api_client.post_dataset(DatasetIn(
                    test_size=test_size,
                    random_state=random_state,
                    image_size=image_size,
                    compressed=compressed,
                ))
                st.success(f'Dataset scheduled: {dataset.id}')
            except RuntimeError as err:
                st.error(err)
