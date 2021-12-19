import os
from PIL import Image
from giwebapp.config import CONFIG

DECALS_LABELS = {
    0: 'Disturbed Galaxies',
    1: 'Merging Galaxies',
    2: 'Round Smooth Galaxies',
    3: 'In-between Round Smooth Galaxies',
    4: 'Cigar Shaped Smooth Galaxies',
    5: 'Barred Spiral Galaxies',
    6: 'Unbarred Tight Spiral Galaxies',
    7: 'Unbarred Loose Spiral Galaxies',
    8: 'Edge-on Galaxies without Bulge',
    9: 'Edge-on Galaxies with Bulge',
}


def get_home_image():
    return Image.open(os.path.join(CONFIG.assets_path, 'home.png'))


def get_favicon_image():
    return Image.open(os.path.join(CONFIG.assets_path, 'favicon.png'))


def get_decals_sample_image(category_id):
    print(category_id)
    return Image.open(os.path.join(CONFIG.assets_path, 'decals_samples', f'{category_id}.jpg'))
