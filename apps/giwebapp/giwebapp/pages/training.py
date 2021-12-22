import streamlit as st
from gicommon.models.learning import DLModelIn
from giwebapp.components.tfboard import st_tensorboard
from giwebapp.config import CONFIG
from giwebapp.api import ApiClient
from giwebapp.utils import get_item_from_option, get_dataset_options, format_dataset_option

api_client = ApiClient(CONFIG.api_url)


def render():
    st.title('Training')

    st.markdown('### Create and train a new model')

    with st.form('dl_model_form'):
        col1, col2 = st.columns(2)

        with col1:
            datasets = api_client.get_datasets()
            dataset_option = st.selectbox(
                'Dataset', options=get_dataset_options(datasets, split_only=True), format_func=format_dataset_option
            )
            dataset = get_item_from_option(datasets, dataset_option)
            base_model_names = api_client.get_base_model_names()
            base_model_name = st.selectbox('Base model', options=base_model_names)
            use_imagenet_weights = st.checkbox('Use imagenet weights', value=True)
            base_model_trainable = st.checkbox('Base model trainable', value=True)
            include_relu_dense = st.checkbox('Include last RELU Dense', value=True)
            relu_dense_units = st.number_input('Last RELU Dense units', min_value=1, value=512)
        with col2:
            batch_size = st.select_slider('Batch size', options=[2 ** i for i in range(1, 12)], value=32)
            base_learning_rate = st.slider(
                'Base learning rate', min_value=0.0001, max_value=0.1, value=0.005, step=0.0001, format="%.4f"
            )
            reduce_learning_rate_patience = st.slider('Learning rate patience', min_value=1, max_value=10, value=2)
            epochs = st.slider('Epochs', min_value=1, value=25)
        create_button = st.form_submit_button('Train')
    if create_button:
        if not dataset:
            st.warning('You should select a dataset')
        else:
            try:
                dl_model = api_client.create_dl_model(DLModelIn(
                    dataset_id=dataset.id,
                    batch_size=batch_size,
                    base_model_name=base_model_name,
                    use_imagenet_weights=use_imagenet_weights,
                    base_model_trainable=base_model_trainable,
                    include_relu_dense=include_relu_dense,
                    relu_dense_units=relu_dense_units,
                    base_learning_rate=base_learning_rate,
                    reduce_learning_rate_patience=reduce_learning_rate_patience,
                    epochs=epochs,
                ))
                st.success(f'Model training scheduled: {dl_model.id}')
            except RuntimeError as err:
                st.error(err)

    st.markdown('### Tensorflow Dashboard')
    st_tensorboard(CONFIG.dashboard_url)
