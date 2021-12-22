import requests
from json import JSONDecodeError
from urllib.parse import urljoin
from gicommon.routes import PREPROCESSING_PREFIX, LEARNING_PREFIX, DATASETS_PATH, DL_MODELS_PATH, EVALUATION_PATH, \
    PREDICTION_PATH, EXPORT_PATH, IMPORT_PATH, BASE_MODEL_NAMES_PATH
from gicommon.models.preprocessing import DatasetIn, DatasetOut
from gicommon.models.learning import DLModelIn, DLModelOut, PredictionIn, PredictionOut, EvaluationIn


class ApiClient:
    def __init__(self, api_url):
        self._api_url = api_url

    def get_url(self, paths):
        return urljoin(self._api_url, '/'.join(path.strip('/') for path in paths))

    def get_datasets(self):
        url = self.get_url([PREPROCESSING_PREFIX, DATASETS_PATH])
        response = requests.get(url)
        if response.status_code == 200:
            return [DatasetOut.parse_obj(item) for item in response.json()]
        else:
            raise RuntimeError(format_error(response))

    def create_dataset(self, dataset: DatasetIn):
        url = self.get_url([PREPROCESSING_PREFIX, DATASETS_PATH])
        response = requests.post(url, data=dataset.json())
        if response.status_code == 201:
            return DatasetOut.parse_obj(response.json())
        else:
            raise RuntimeError(format_error(response))

    def get_dl_models(self):
        url = self.get_url([LEARNING_PREFIX, DL_MODELS_PATH])
        response = requests.get(url)
        if response.status_code == 200:
            return [DLModelOut.parse_obj(item) for item in response.json()]
        else:
            raise RuntimeError(format_error(response))

    def create_dl_model(self, dl_model: DLModelIn):
        url = self.get_url([LEARNING_PREFIX, DL_MODELS_PATH])
        response = requests.post(url, data=dl_model.json())
        if response.status_code == 201:
            return DLModelOut.parse_obj(response.json())
        else:
            raise RuntimeError(format_error(response))

    def get_base_model_names(self):
        url = self.get_url([LEARNING_PREFIX, BASE_MODEL_NAMES_PATH])
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(format_error(response))

    def evaluate_dl_model(self, dl_model_id, evaluation: EvaluationIn):
        url = self.get_url([LEARNING_PREFIX, EVALUATION_PATH.format(dl_model_id)])
        response = requests.post(url, data=evaluation.json())
        if response.status_code == 200:
            return DLModelOut.parse_obj(response.json())
        else:
            raise RuntimeError(format_error(response))

    def predict_image(self, dl_model_id, prediction: PredictionIn):
        url = self.get_url([LEARNING_PREFIX, PREDICTION_PATH.format(dl_model_id)])
        response = requests.post(url, data=prediction.json())
        if response.status_code == 200:
            return PredictionOut.parse_obj(response.json())
        else:
            raise RuntimeError(format_error(response))

    def export_dl_model(self, dl_model_id):
        url = self.get_url([LEARNING_PREFIX, EXPORT_PATH.format(dl_model_id)])
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(format_error(response))

    def import_dl_models(self):
        url = self.get_url([LEARNING_PREFIX, IMPORT_PATH])
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(format_error(response))


def format_error(err):
    try:
        json_err = err.json()
        return json_err['detail']
    except JSONDecodeError:
        return f'Api Error: {err.content}'
