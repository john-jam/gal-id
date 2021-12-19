import os
from decouple import config
from pydantic import BaseModel

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))


class Settings(BaseModel):
    debug = config('GI_DEBUG', cast=bool, default=False)
    api_url = config('GI_API_URL', default='http://127.0.0.1:8000')
    dashboard_url = config('GI_DASHBOARD_URL', default='http://127.0.0.1:6006')
    assets_path = config('GI_ASSETS_PATH', default=os.path.join(MODULE_PATH, os.path.pardir, 'assets'))
    data_path = config('GI_DATA_PATH', default=os.path.join(MODULE_PATH, *[os.path.pardir for _ in range(3)], 'data'))
    lsv_cutout_url = "https://www.legacysurvey.org/viewer/cutout.jpg"
    lsv_ra_param = 'ra'
    lsv_dec_param = 'dec'
    lsv_layer_param = 'layer'
    lsv_zoom_param = 'zoom'


CONFIG = Settings()
