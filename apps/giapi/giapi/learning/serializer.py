import os
from zipfile import ZipFile, ZIP_DEFLATED
from gicommon.models.learning import DLModel
from giapi.config import CONFIG
from giapi.learning.model import DLModelManager


class DLModelSerializer:
    def __init__(self, compression=ZIP_DEFLATED):
        self._compression = compression
        self._model_file_name = 'model.json'

    @staticmethod
    def get_base_path():
        return os.path.join(CONFIG.data_path, 'export')

    @staticmethod
    def get_export_path(dl_model: DLModel):
        return os.path.join(DLModelSerializer.get_base_path(), f'{dl_model.id}.zip')

    def get_dl_models(self):
        # Retrieve all DLModels
        dl_models = []
        for root, dirs, files in os.walk(DLModelSerializer.get_base_path()):
            for file in files:
                export_path = os.path.join(str(root), str(file))
                with ZipFile(export_path, 'r', compression=self._compression) as zip_file:
                    with zip_file.open(self._model_file_name, 'r') as json_model_file:
                        json_model = json_model_file.read().decode('UTF-8')
                        dl_models.append(DLModel.parse_raw(json_model))
        return dl_models

    def load(self, dl_model: DLModel):
        # Define the input location
        export_path = DLModelSerializer.get_export_path(dl_model)

        # Extract all the files except the model json representation
        with ZipFile(export_path, 'r', compression=self._compression) as zip_file:
            zip_files = zip_file.namelist()
            for file in zip_files:
                if file == self._model_file_name:
                    continue
                zip_file.extract(file, CONFIG.data_path)

    def dump(self, dl_model: DLModel):
        # Define the output location
        export_path = DLModelSerializer.get_export_path(dl_model)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Create the db model json representation
        json_model = dl_model.json()

        # Export as a zip archive
        with ZipFile(export_path, 'w', compression=self._compression) as zip_file:
            # Export the weights
            save_path = DLModelManager.get_save_path(dl_model)
            out_save_path = os.path.join(os.path.basename(os.path.dirname(save_path)), os.path.basename(save_path))
            zip_file.write(save_path, arcname=out_save_path)

            # Export the logs
            logs_path = DLModelManager.get_logs_path(dl_model)
            for root, dirs, files in os.walk(logs_path):
                for file in files:
                    log_path = os.path.join(str(root), str(file))
                    out_folders = str(root).split(os.sep)
                    num_folders_to_select = len(out_folders) - (len(logs_path.split(os.sep)) - 2)
                    out_log_path = os.path.join(*out_folders[-num_folders_to_select:], str(file))
                    zip_file.write(log_path, arcname=out_log_path)

            # Export the db representation
            with zip_file.open(self._model_file_name, 'w') as json_model_file:
                json_model_file.write(json_model.encode('UTF-8'))
