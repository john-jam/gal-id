import streamlit as st
from gicommon.models.learning import Evaluation, EvaluationIn
from giwebapp.config import CONFIG
from giwebapp.api import ApiClient
from giwebapp.utils import get_item_from_option, get_dl_models_options, format_dl_model_option

api_client = ApiClient(CONFIG.api_url)


def render_evaluation(evaluation: Evaluation):
    st.markdown('### Model evaluation')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            'Train Accuracy',
            value="%.4f" % evaluation.train_accuracy,
            delta="%.4f" % (evaluation.train_accuracy - evaluation.test_accuracy)
        )
    with col2:
        st.metric(
            'Train Loss',
            value="%.4f" % evaluation.train_loss,
            delta="%.4f" % (evaluation.train_loss - evaluation.test_loss),
            delta_color='inverse'
        )
    with col3:
        st.metric(
            'Test Accuracy',
            value="%.4f" % evaluation.test_accuracy,
            delta="%.4f" % (evaluation.test_accuracy - evaluation.train_accuracy)
        )
    with col4:
        st.metric(
            'Loss',
            value="%.4f" % evaluation.test_loss,
            delta="%.4f" % (evaluation.test_loss - evaluation.train_loss),
            delta_color='inverse'
        )


def render():
    st.title('Evaluation')

    st.markdown('### Evaluate the model')

    col1, col2 = st.columns(2)

    with col1:
        dl_models = api_client.get_all_dl_models()
        dl_model_option = st.selectbox(
            'Model', options=get_dl_models_options(dl_models), format_func=format_dl_model_option
        )
        dl_model = get_item_from_option(dl_models, dl_model_option)
    with col2:
        batch_size = st.select_slider('Batch size', options=[2 ** i for i in range(1, 12)], value=32)

    if dl_model:
        if dl_model.trained:
            if dl_model.evaluation:
                evaluate_button = st.button('Re-evaluate')
            else:
                evaluate_button = st.button('Evaluate')
            if evaluate_button:
                try:
                    with st.spinner('Evaluating...'):
                        dl_model = api_client.post_evaluation(dl_model.id, EvaluationIn(batch_size=batch_size))
                        render_evaluation(dl_model.evaluation)
                except RuntimeError as err:
                    st.error(err)
            elif dl_model.evaluation:
                render_evaluation(dl_model.evaluation)

        st.markdown('### Model info')

        st.json(dl_model.json())
