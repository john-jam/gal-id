import os
from decouple import config
from pydantic import BaseModel

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))


class Settings(BaseModel):
    debug = config('GI_DEBUG', cast=bool, default=False)
    mongo_uri = config('GI_MONGO_URI''', default='mongodb://localhost:27017')
    data_path = config(
        'GI_DATA_PATH',
        default=os.path.abspath(os.path.join(MODULE_PATH, *[os.path.pardir for _ in range(3)], 'data'))
    )
    galaxy10_decals_url = 'https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5'
    galaxy10_decals_num_categories = 10
    enable_gpu = config('GI_ENABLE_GPU', cast=bool, default=False)
    gpu_memory_limit = config('GI_GPU_MEMORY_LIMIT', cast=int, default=5120)


CONFIG = Settings()
