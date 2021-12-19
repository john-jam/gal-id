import os
from urllib.request import urlretrieve
from giapi.config import CONFIG
from giapi.logging import logger


def fetch_galaxy10_decals():
    downloads_path = os.path.join(CONFIG.data_path, 'downloads')
    galaxy10_decals_path = os.path.join(downloads_path, 'galaxy10_decals.h5')

    os.makedirs(downloads_path, exist_ok=True)

    if not os.path.isfile(galaxy10_decals_path):
        logger.info('Galaxy10 DECals file not found locally, downloading...')
        urlretrieve(CONFIG.galaxy10_decals_url, galaxy10_decals_path)
    return galaxy10_decals_path
