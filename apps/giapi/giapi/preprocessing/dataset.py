import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from gicommon.models.preprocessing import Dataset
from giapi.preprocessing.download import fetch_galaxy10_decals
from giapi.config import CONFIG
from giapi.logging import logger


class DatasetManager:
    def __init__(self, dataset: Dataset, batch_size=500):
        self._dataset = dataset
        self._batch_size = batch_size

    def get_dataset_path(self, is_h5=False):
        formatted_date = self._dataset.created_at.strftime('%Y-%m-%d-%H-%M-%S')
        return os.path.join(CONFIG.data_path, 'datasets',
                            f'{formatted_date}_{str(self._dataset.id)}{".h5" if is_h5 else ""}')

    def get_train_test_paths(self):
        dataset_path = self.get_dataset_path()
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')
        return train_path, test_path

    def get_memory_iterator(self, generator: ImageDataGenerator, x, y, batch_size):
        return generator.flow(
            x,
            y,
            batch_size=batch_size,
            shuffle=True
        )

    def get_disk_iterator(self, generator: ImageDataGenerator, dir_path, batch_size):
        return generator.flow_from_directory(
            dir_path,
            class_mode='categorical',
            batch_size=batch_size,
            target_size=(self._dataset.image_size, self._dataset.image_size),
            shuffle=True
        )

    def get_iterators(self, train_generator: ImageDataGenerator, test_generator: ImageDataGenerator, batch_size):
        if self._dataset.compressed:
            # Return memory iterators
            h5py_file_path = self.get_dataset_path(is_h5=True)
            with h5py.File(h5py_file_path, 'r') as h5py_file:
                x_train = np.array(h5py_file['train']['images'])
                x_test = np.array(h5py_file['test']['images'])
                y_train = to_categorical(np.array(h5py_file['train']['categories']),
                                         CONFIG.galaxy10_decals_num_categories)
                y_test = to_categorical(np.array(h5py_file['test']['categories']),
                                        CONFIG.galaxy10_decals_num_categories)
            train_it = self.get_memory_iterator(train_generator, x_train, y_train, batch_size)
            test_it = self.get_memory_iterator(test_generator, x_test, y_test, batch_size)
        else:
            # Return disks iterators
            train_path, test_path = self.get_train_test_paths()
            train_it = self.get_disk_iterator(train_generator, train_path, batch_size)
            test_it = self.get_disk_iterator(test_generator, test_path, batch_size)
        return train_it, test_it

    def preprocess_images(self, images, original_size):
        # Resize the image if needed
        if self._dataset.image_size != original_size:
            images = np.array([
                np.array(Image.fromarray(image).resize((self._dataset.image_size, self._dataset.image_size)))
                for image in images
            ])
        return images

    def create_categories_dataset(self, group, idxs_num):
        group.create_dataset(
            'categories', shape=(0,), maxshape=(idxs_num,),
            dtype='uint8', chunks=True, compression="gzip"
        )

    def create_images_dataset(self, group, image_shape, idxs_num):
        group.create_dataset(
            'images', shape=(0, *image_shape), maxshape=(idxs_num, *image_shape),
            dtype='uint8', chunks=True, compression="gzip"
        )

    def save_batch_in_group(self, group, idxs, batch_idxs, images, categories):
        images_dataset = group['images']
        categories_dataset = group['categories']

        images = np.array([
            image for idx, image in enumerate(images) if batch_idxs[idx] in idxs
        ])
        images_dataset.resize((images_dataset.shape[0] + len(images), *images_dataset.shape[1:]))
        images_dataset[-len(images):] = images

        categories = np.array([
            category for idx, category in enumerate(categories) if batch_idxs[idx] in idxs
        ])
        categories_dataset.resize((categories_dataset.shape[0] + len(categories),))
        categories_dataset[-len(categories):] = categories

    def save_batch(self, train_idxs, test_idxs, batch_idxs, categories, images):
        logger.info(
            f'Saving items from {batch_idxs[0]} to {batch_idxs[-1]} for dataset {self._dataset.id} {"(h5)" if self._dataset.compressed else "(disk)"}'
        )

        if not self._dataset.compressed:
            # Save images in files and folders by category
            train_path, test_path = self.get_train_test_paths()
            for idx in range(len(batch_idxs)):
                ref_idx = batch_idxs[idx]
                category = categories[idx]
                image = images[idx]

                # Define where to save the current image according to the splits
                image_base_path = train_path if ref_idx in train_idxs else test_path

                # Save the image
                image_path = os.path.join(image_base_path, str(category))
                os.makedirs(image_path, exist_ok=True)
                image_file_path = os.path.join(image_path, f'{ref_idx}.jpg')
                Image.fromarray(image).save(image_file_path, 'JPEG')
        else:
            # Save the images and categories in h5 file
            h5py_file_path = self.get_dataset_path(is_h5=True)
            os.makedirs(os.path.dirname(h5py_file_path), exist_ok=True)
            with h5py.File(h5py_file_path, 'a') as h5py_file_out:
                # Create groups and datasets if not exists
                train_group = h5py_file_out.require_group("train")
                test_group = h5py_file_out.require_group("test")
                if 'images' not in train_group:
                    image_shape = (self._dataset.image_size, self._dataset.image_size, 3)
                    self.create_images_dataset(train_group, image_shape, len(train_idxs))
                    self.create_images_dataset(test_group, image_shape, len(test_idxs))
                    self.create_categories_dataset(train_group, len(train_idxs))
                    self.create_categories_dataset(test_group, len(test_idxs))

                self.save_batch_in_group(train_group, train_idxs, batch_idxs, images, categories)
                self.save_batch_in_group(test_group, test_idxs, batch_idxs, images, categories)

    def split_idxs(self, y):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self._dataset.test_size,
                                          random_state=self._dataset.random_state)
        for train_idxs, test_idxs in splitter.split(np.zeros(len(y)), y):
            return train_idxs, test_idxs

    def split(self):
        # Open the original dataset
        galaxy10_decals_path = fetch_galaxy10_decals()
        with h5py.File(galaxy10_decals_path, 'r') as h5py_file_in:
            # Load all of our categories in memory to stratify
            categories = np.array(h5py_file_in['ans'])
            # Load images references only for now
            images = h5py_file_in['images']
            # Define the original images shapes
            original_size = images[0].shape[0]

            # Split the images into a train/test datasets
            train_idxs, test_idxs = self.split_idxs(categories)

            # Create batches of images to save h5py loading time
            all_idxs = np.arange(len(categories))
            idx_size = len(all_idxs)
            for start_idx in range(0, idx_size, self._batch_size):
                end_idx = min(start_idx + self._batch_size, idx_size)

                # Load the indexes, images and categories for this batch
                batch_idxs = all_idxs[start_idx:end_idx]
                batch_categories = categories[start_idx:end_idx]
                batch_images = np.array(images[start_idx:end_idx])

                # Preprocess and save each images
                batch_images = self.preprocess_images(batch_images, original_size)
                self.save_batch(train_idxs, test_idxs, batch_idxs, batch_categories, batch_images)

        logger.info(f'Dataset {self._dataset.id} split')
